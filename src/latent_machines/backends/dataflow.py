"""Backends for ``dataflow/v1``: conventional, scalar-machine, and compiled.

Every backend returns an ``Outcome`` for one input vector: the final event,
the canonical values of the evaluated nodes, the logical work (nodes
evaluated, as the machine counts it), and the residue before rounding.
Each is written independently of the oracle; only the program format and
the word width are shared.

- ``python_int``: conventional evaluation. Evaluates every node in node
  order with unbounded Python integers, then finds the first out-of-range
  node in evaluation order. Its own level computation, no tensors.
- ``ScalarDelta``: a scalar machine. One instruction per node in evaluation
  order; operands live in E00's delta-rule memory (``memory/v1`` words of
  ``width`` bits holding two's-complement patterns), read and written one
  memory action per tick, with a host integer ALU between reads and the
  write. Arithmetic is not on the tensor path; storage is.
- ``CompiledGraph``: one batched tensor step per level, columns are input
  vectors. Lowerings:

  - ``gather``: the direct compiled graph. Operands are gathered by index;
    every node in a level computes all three unit types (linear, product,
    and ``a + relu(b - a)`` for max) and a mask selects one, as a dense
    layer would.
  - ``weights``: program-specialized weights. Wiring and constants are
    baked into dense matrices: ``Wl V``, ``Wr V``, ``Wc V`` per level, and
    ``Win x + b`` for the leaves. Same unit computation as ``gather``.
  - ``dispatch``: conditional units. Nodes are routed, statically per
    program, to one expert per op type with a fixed capacity per round;
    an expert runs more rounds when a level has more nodes than its
    capacity, so no node is dropped. Only the needed op is computed.
  - ``grouped``: ``dispatch`` with unlimited capacity, used with int64 as
    the optimized conventional baseline.

  Overflow is checked after each level; the first out-of-range row in
  evaluation order, per column, is that column's fault. Rows of a column
  after its fault are computed but never decoded.

``decode``, ``encode``, and ``residue`` exist for conformance and snapshots.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from latent_machines.backends.delta import DeltaMemory
from latent_machines.machines.dataflow import (
    DataflowEvent,
    DataflowState,
    FaultReason,
    Program,
)
from latent_machines.machines.events import EventKind
from latent_machines.machines.memory import Memory, Read, Write


@dataclass(frozen=True)
class Outcome:
    event: DataflowEvent
    values: tuple[int | None, ...]  # canonical: evaluated nodes only
    work: int  # nodes evaluated, the machine's unit
    residue: float = 0.0
    ticks: int = 0  # backend steps: memory actions and ALU ops, or levels


def _depths(program: Program) -> list[int]:
    depth: list[int] = []
    for node in program.nodes:
        depth.append(max((depth[a] + 1 for a in node.args), default=0))
    return depth


def evaluation_order(program: Program) -> list[int]:
    depth = _depths(program)
    return sorted(range(len(program.nodes)), key=lambda i: (depth[i], i))


def _fault(node: int) -> DataflowEvent:
    return DataflowEvent(EventKind.FAULT, node=node, reason=FaultReason.OVERFLOW)


# ---------- conventional ----------


def python_int(program: Program, inputs: Sequence[int], width: int) -> Outcome:
    low, high = -(1 << (width - 1)), (1 << (width - 1)) - 1
    raw: list[int] = []
    for node in program.nodes:
        args = [raw[a] for a in node.args]
        match node.op:
            case "input":
                raw.append(inputs[node.value])  # type: ignore[index]
            case "const":
                raw.append(node.value)  # type: ignore[arg-type]
            case "add":
                raw.append(args[0] + args[1])
            case "sub":
                raw.append(args[0] - args[1])
            case "mul":
                raw.append(args[0] * args[1])
            case "max":
                raw.append(args[0] if args[0] >= args[1] else args[1])
            case "sum":
                raw.append(sum(args))
    values: list[int | None] = [None] * len(raw)
    for work, index in enumerate(evaluation_order(program), start=1):
        if not low <= raw[index] <= high:
            return Outcome(_fault(index), tuple(values), work)
        values[index] = raw[index]
    outputs = tuple(raw[o] for o in program.outputs)
    return Outcome(
        DataflowEvent(EventKind.OUTPUT, outputs=outputs), tuple(values), len(raw)
    )


# ---------- scalar machine on delta-rule memory ----------


@dataclass(frozen=True)
class ScalarDelta:
    """One node per instruction, operands in a delta-rule ``memory/v1``."""

    width: int = 16
    dtype: torch.dtype = torch.float32

    def run(
        self,
        program: Program,
        inputs: Sequence[int],
        snapshot_at: int | None = None,
    ) -> tuple[Outcome, str]:
        """Execute; at instruction ``snapshot_at`` round-trip the memory.

        Returns the outcome and the snapshot result: ``agrees``, ``differs``,
        or ``not reached``.
        """
        memory = Memory(capacity=len(program.nodes), word_bits=self.width)
        backend = DeltaMemory(memory, self.dtype)
        state = backend.initial_state()
        low, high = -(1 << (self.width - 1)), (1 << (self.width - 1)) - 1
        mask = (1 << self.width) - 1
        ticks = residue = 0
        snapshot = "not reached"

        def read(address: int) -> int:
            nonlocal ticks
            transition = backend.step(state, Read(address))
            ticks += 1
            word = transition.event.value
            assert word is not None
            return word - (1 << self.width) if word > high else word

        order = evaluation_order(program)
        event: DataflowEvent | None = None
        work = 0
        for pc, index in enumerate(order):
            if pc == snapshot_at:
                saved = json.loads(json.dumps(memory.snapshot(backend.decode(state))))
                restored = backend.encode(memory.restore(saved))
                same = torch.equal(restored.matrix, state.matrix)
                snapshot = "agrees" if same else "differs"
                state = restored
            node = program.nodes[index]
            if node.op == "input":
                value = inputs[node.value]  # type: ignore[index]
                ticks += 1
            elif node.op == "const":
                value = node.value  # type: ignore[assignment]
                ticks += 1
            else:
                args = [read(a) for a in node.args]
                ticks += max(1, len(args) - 1)  # ALU ops
                match node.op:
                    case "add":
                        value = args[0] + args[1]
                    case "sub":
                        value = args[0] - args[1]
                    case "mul":
                        value = args[0] * args[1]
                    case "max":
                        value = max(args)
                    case _:
                        value = sum(args)
            work += 1
            assert value is not None
            if not low <= value <= high:
                event = _fault(index)
                break
            state = backend.step(state, Write(index, value & mask)).next_state
            ticks += 1
            residue = max(residue, backend.residue(state))
        if event is None:
            outputs = tuple(read(o) for o in program.outputs)
            event = DataflowEvent(EventKind.OUTPUT, outputs=outputs)
        words = backend.decode(state).words
        evaluated = set(order[: work - (event.kind is EventKind.FAULT)])
        values = tuple(
            (w - (1 << self.width) if w > high else w) if i in evaluated else None
            for i, w in enumerate(words)
        )
        return Outcome(event, values, work, residue, ticks), snapshot

    @staticmethod
    def state_bytes(program: Program, width: int = 16, itemsize: int = 4) -> int:
        return len(program.nodes) * width * itemsize


# ---------- compiled graph ----------

UNITS = ("linear", "mul", "max")
EXPERTS = ("add", "sub", "mul", "max", "sum")


@dataclass
class _Level:
    nodes: torch.Tensor  # (s,) node indices in evaluation order
    left: torch.Tensor  # (s,) argument indices, 0 where unused
    right: torch.Tensor
    is_mul: torch.Tensor  # (s,) bool
    is_max: torch.Tensor
    lin_rows: torch.Tensor  # linear terms: row in level, argument, coefficient
    lin_cols: torch.Tensor
    lin_coef: torch.Tensor
    dense: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None  # Wl, Wr, Wc
    # op -> rounds of (rows in level, first operands, second operands); for
    # "sum", the operands are (row within round, argument) pairs instead.
    experts: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
    pads: int  # padded expert slots


@dataclass
class TensorState:
    level: int  # next level to evaluate
    values: torch.Tensor  # (nodes, batch)
    fault: torch.Tensor  # (batch,) faulting node or -1
    work: torch.Tensor  # (batch,) nodes evaluated


class CompiledGraph:
    """A program lowered to one tensor step per level."""

    def __init__(
        self,
        program: Program,
        width: int = 16,
        dtype: torch.dtype = torch.float32,
        lowering: str = "gather",
        capacity: int | None = None,
    ) -> None:
        if lowering not in ("gather", "weights", "dispatch", "grouped"):
            raise ValueError(f"unknown lowering {lowering!r}")
        if lowering == "dispatch" and (capacity is None or capacity < 1):
            raise ValueError("dispatch needs a positive capacity")
        self.program, self.width, self.dtype = program, width, dtype
        self.lowering = lowering
        self.capacity = capacity if lowering == "dispatch" else None
        self.low, self.high = -(1 << (width - 1)), (1 << (width - 1)) - 1
        started = time.perf_counter()
        self._compile()
        self.compile_seconds = time.perf_counter() - started

    # ----- lowering -----

    def _compile(self) -> None:
        program, n = self.program, len(self.program.nodes)
        depth = _depths(program)
        order = evaluation_order(program)
        self.order = order
        self.position = {node: i for i, node in enumerate(order)}
        by_level: list[list[int]] = [[] for _ in range(max(depth) + 1)]
        for node in order:
            by_level[depth[node]].append(node)
        self.level_nodes = by_level
        long = torch.long
        leaves = by_level[0]
        inputs = [i for i in leaves if program.nodes[i].op == "input"]
        consts = [i for i in leaves if program.nodes[i].op == "const"]
        self.input_nodes = torch.tensor(inputs, dtype=long)
        self.input_index = torch.tensor(
            [program.nodes[i].value for i in inputs], dtype=long
        )
        self.const_nodes = torch.tensor(consts, dtype=long)
        self.const_values = torch.tensor(
            [program.nodes[i].value for i in consts], dtype=self.dtype
        )
        self.leaf_nodes = torch.tensor(leaves, dtype=long)
        self.leaf_inputs = torch.tensor([leaves.index(i) for i in inputs], dtype=long)
        self.leaf_consts = torch.tensor([leaves.index(i) for i in consts], dtype=long)
        if self.lowering == "weights":
            w_in = torch.zeros(len(leaves), program.inputs, dtype=self.dtype)
            bias = torch.zeros(len(leaves), 1, dtype=self.dtype)
            for row, i in enumerate(leaves):
                node = program.nodes[i]
                if node.op == "input":
                    w_in[row, node.value] = 1
                else:
                    bias[row, 0] = node.value  # type: ignore[assignment]
            self.leaf_weights: tuple[torch.Tensor, torch.Tensor] | None = (w_in, bias)
        else:
            self.leaf_weights = None
        self.levels = [self._lower_level(nodes, n) for nodes in by_level[1:]]

    def _lower_level(self, nodes: list[int], n: int) -> _Level:
        program, s = self.program, len(nodes)
        left, right, is_mul, is_max = [], [], [], []
        rows, cols, coefs = [], [], []
        for row, i in enumerate(nodes):
            node = program.nodes[i]
            binary = node.op in ("mul", "max")
            left.append(node.args[0] if binary else 0)
            right.append(node.args[1] if binary else 0)
            is_mul.append(node.op == "mul")
            is_max.append(node.op == "max")
            if node.op in ("add", "sub", "sum"):
                signs = [1, -1] if node.op == "sub" else [1] * len(node.args)
                for arg, sign in zip(node.args, signs, strict=True):
                    rows.append(row)
                    cols.append(arg)
                    coefs.append(sign)
        long = torch.long
        dense = None
        if self.lowering == "weights":
            wl = torch.zeros(s, n, dtype=self.dtype)
            wr = torch.zeros(s, n, dtype=self.dtype)
            wc = torch.zeros(s, n, dtype=self.dtype)
            for row in range(s):
                if is_mul[row] or is_max[row]:
                    wl[row, left[row]] = 1
                    wr[row, right[row]] = 1
            for row, col, coef in zip(rows, cols, coefs, strict=True):
                wc[row, col] += coef
            dense = (wl, wr, wc)
        experts: dict[str, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        pads = 0
        if self.lowering in ("dispatch", "grouped"):
            for op in EXPERTS:
                members = [r for r, i in enumerate(nodes) if program.nodes[i].op == op]
                if not members:
                    continue
                size = self.capacity or len(members)
                experts[op] = []
                for k in range(0, len(members), size):
                    chunk = members[k : k + size]
                    pads += size - len(chunk)
                    args = [program.nodes[nodes[r]].args for r in chunk]
                    if op == "sum":
                        first = [j for j, a in enumerate(args) for _ in a]
                        second = [x for a in args for x in a]
                    else:
                        # Padded slots compute on node 0 and are discarded.
                        padding = [(0, 0)] * (size - len(chunk))
                        first = [a[0] for a in args + padding]
                        second = [a[1] for a in args + padding]
                    experts[op].append(
                        (
                            torch.tensor(chunk, dtype=long),
                            torch.tensor(first, dtype=long),
                            torch.tensor(second, dtype=long),
                        )
                    )
        return _Level(
            torch.tensor(nodes, dtype=long),
            torch.tensor(left, dtype=long),
            torch.tensor(right, dtype=long),
            torch.tensor(is_mul),
            torch.tensor(is_max),
            torch.tensor(rows, dtype=long),
            torch.tensor(cols, dtype=long),
            torch.tensor(coefs, dtype=self.dtype),
            dense,
            experts,
            pads,
        )

    # ----- execution -----

    def initial_state(self, batch: int) -> TensorState:
        return TensorState(
            0,
            torch.zeros(len(self.program.nodes), batch, dtype=self.dtype),
            torch.full((batch,), -1, dtype=torch.long),
            torch.zeros(batch, dtype=torch.long),
        )

    @property
    def num_levels(self) -> int:
        return 1 + len(self.levels)

    def step(self, state: TensorState, x: torch.Tensor) -> TensorState:
        """Evaluate the next level for every column of ``x`` (inputs, batch)."""
        values = state.values.clone()
        if state.level == 0:
            rows = self.leaf_nodes
            if self.leaf_weights is not None:
                w_in, bias = self.leaf_weights
                out = w_in @ x + bias
            else:
                out = torch.empty(len(rows), x.shape[1], dtype=self.dtype)
                out[self.leaf_inputs] = x[self.input_index]
                out[self.leaf_consts] = self.const_values[:, None]
        else:
            level = self.levels[state.level - 1]
            rows = level.nodes
            out = self._evaluate(level, values)
        values[rows] = out
        fault, work = state.fault.clone(), state.work.clone()
        bad = (out > self.high) | (out < self.low) | torch.isnan(out)
        running = fault < 0
        first = bad.int().argmax(dim=0)
        hit = bad.any(dim=0) & running
        done = sum(len(nodes) for nodes in self.level_nodes[: state.level])
        fault[hit] = rows[first[hit]]
        work[hit] = done + first[hit] + 1
        work[running & ~hit] = done + len(rows)
        return TensorState(state.level + 1, values, fault, work)

    def _evaluate(self, level: _Level, values: torch.Tensor) -> torch.Tensor:
        batch = values.shape[1]
        if level.experts:
            out = torch.empty(len(level.nodes), batch, dtype=self.dtype)
            for op, rounds in level.experts.items():
                for members, first, second in rounds:
                    result = self._expert(op, first, second, values, self.capacity)
                    out[members] = result[: len(members)]
            return out
        if level.dense is not None:
            wl, wr, wc = level.dense
            a, b, linear = wl @ values, wr @ values, wc @ values
        else:
            a, b = values[level.left], values[level.right]
            linear = torch.zeros(len(level.nodes), batch, dtype=self.dtype)
            linear.index_add_(
                0, level.lin_rows, values[level.lin_cols] * level.lin_coef[:, None]
            )
        product = a * b
        maximum = a + torch.clamp(b - a, min=0)
        return torch.where(
            level.is_mul[:, None],
            product,
            torch.where(level.is_max[:, None], maximum, linear),
        )

    def _expert(
        self,
        op: str,
        first: torch.Tensor,
        second: torch.Tensor,
        values: torch.Tensor,
        capacity: int | None,
    ) -> torch.Tensor:
        """One expert round; the result has ``capacity`` rows when set."""
        if op == "sum":
            rows = capacity or int(first.max()) + 1
            out = torch.zeros(rows, values.shape[1], dtype=self.dtype)
            return out.index_add_(0, first, values[second])
        a, b = values[first], values[second]
        if op == "add":
            return a + b
        if op == "sub":
            return a - b
        if op == "mul":
            return a * b
        return a + torch.clamp(b - a, min=0)

    def run(self, inputs: Sequence[Sequence[int]]) -> TensorState:
        x = torch.tensor(inputs, dtype=self.dtype).T.contiguous()
        state = self.initial_state(len(inputs))
        while state.level < self.num_levels:
            state = self.step(state, x)
        return state

    # ----- decoding, for conformance only -----

    def outcome(self, state: TensorState, column: int) -> Outcome:
        canonical, residue = self.decode(state, column)
        fault = int(state.fault[column])
        if fault >= 0:
            event = _fault(fault)
        else:
            outputs = tuple(canonical.values[o] for o in self.program.outputs)
            event = DataflowEvent(EventKind.OUTPUT, outputs=outputs)  # type: ignore[arg-type]
        return Outcome(
            event, canonical.values, int(state.work[column]), residue, state.level
        )

    def decode(self, state: TensorState, column: int) -> tuple[DataflowState, float]:
        """Canonical state of one column and its residue before rounding."""
        fault = int(state.fault[column])
        stop = self.position[fault] if fault >= 0 else len(self.order)
        done = sum(len(nodes) for nodes in self.level_nodes[: state.level])
        evaluated = self.order[: min(stop, done)]
        column_values = state.values[:, column].to(torch.float64)
        values: list[int | None] = [None] * len(self.program.nodes)
        residue = 0.0
        if evaluated:
            picked = column_values[torch.tensor(evaluated)]
            if not torch.isfinite(picked).all():
                raise ValueError("non-finite value in an evaluated node")
            rounded = picked.round()
            residue = (picked - rounded).abs().max().item()
            for node, value in zip(evaluated, rounded.tolist(), strict=True):
                values[node] = int(value)
        finished = fault >= 0 or state.level == self.num_levels
        return DataflowState(state.level, tuple(values), finished), residue

    def encode(self, canonical: DataflowState) -> TensorState:
        """A one-column state holding ``canonical``; no faulted states."""
        if canonical.finished:
            raise ValueError("only running states are encoded")
        state = self.initial_state(1)
        for node, value in enumerate(canonical.values):
            if value is not None:
                state.values[node, 0] = value
        done = sum(len(nodes) for nodes in self.level_nodes[: canonical.level])
        work = state.work + done
        return TensorState(canonical.level, state.values, state.fault, work)

    # ----- sizes -----

    def weight_bytes(self) -> int:
        """Bytes of the compiled program: indices, masks, weights, constants."""
        tensors = [self.input_nodes, self.input_index, self.const_nodes]
        tensors.append(self.const_values)
        if self.leaf_weights is not None:
            tensors += list(self.leaf_weights)
        for level in self.levels:
            if level.dense is not None:
                tensors += [*level.dense, level.is_mul, level.is_max]
            elif level.experts:
                tensors += [
                    t for rounds in level.experts.values() for r in rounds for t in r
                ]
                tensors += [level.nodes]
            else:
                tensors += [level.nodes, level.left, level.right, level.is_mul]
                tensors += [level.is_max, level.lin_rows, level.lin_cols]
                tensors += [level.lin_coef]
        return sum(t.numel() * t.element_size() for t in tensors)

    def units_per_evaluation(self) -> int:
        """Unit results computed per column, padding and unselected units included."""
        units = len(self.level_nodes[0])
        for level in self.levels:
            if level.experts:
                units += sum(
                    self.capacity or len(members)
                    for rounds in level.experts.values()
                    for members, _, _ in rounds
                )
            else:
                units += len(UNITS) * len(level.nodes)
        return units

    def padded_slots(self) -> int:
        return sum(level.pads for level in self.levels)

    def state_bytes(self, batch: int = 1) -> int:
        itemsize = torch.tensor([], dtype=self.dtype).element_size()
        return len(self.program.nodes) * batch * itemsize
