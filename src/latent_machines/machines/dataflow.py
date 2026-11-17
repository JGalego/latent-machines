"""Integer dataflow graphs: the reference oracle for E04.

Specification ``dataflow/v1``:

- Program: a tuple of nodes and a tuple of output node indices. A node is
  ``input`` (reads input element ``value``), ``const`` (the integer
  ``value``), ``add``, ``sub``, ``mul``, ``max`` (two arguments each), or
  ``sum`` (one or more arguments). Arguments name earlier nodes, so node
  order is a topological order; a node may use the same argument twice.
  Programs that break these rules, or whose constants are out of range, are
  rejected with ``ValueError`` before any evaluation.
- Words: signed integers of ``width`` bits, ``-2**(width-1)`` to
  ``2**(width-1) - 1``. Arithmetic is exact; a ``sum`` is checked on its
  final value only, so its partial sums are unbounded.
- Levels: leaves (``input`` and ``const``) are level 0; any other node is
  one more than its deepest argument. Evaluation order is by level, then by
  node index within a level. The tick is one level.
- Faults: the first node in evaluation order whose value is out of range,
  an input element included, raises ``fault: overflow`` naming that node.
  Values are never wrapped or clamped. Nodes after it are not evaluated and
  the machine is finished. Stepping a finished machine raises ``fault:
  finished`` and evaluates nothing.
- Events: a level that completes without a fault yields ``continue``; the
  last level yields ``output`` carrying the output values in output order.
- Work: one unit per node evaluated, the faulting node included.
- Snapshots: at level boundaries, the safepoints. A snapshot holds the next
  level and the value of every evaluated node (``None`` for the others).

Inputs must be a sequence of exactly ``program.inputs`` ints; anything else
is outside the schema and raises ``TypeError`` or ``ValueError``.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from latent_machines.machines.events import EventKind

VERSION = "dataflow/v1"
LEAVES = ("input", "const")
BINARY = ("add", "sub", "mul", "max")
OPS = (*LEAVES, *BINARY, "sum")


@dataclass(frozen=True, slots=True)
class Node:
    op: str
    args: tuple[int, ...] = ()
    value: int | None = None  # input index for "input", the constant for "const"


@dataclass(frozen=True)
class Program:
    nodes: tuple[Node, ...]
    outputs: tuple[int, ...]
    inputs: int  # length of the input vector


class FaultReason(Enum):
    OVERFLOW = "overflow"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True)
class DataflowEvent:
    kind: EventKind
    outputs: tuple[int, ...] | None = None  # for OUTPUT
    node: int | None = None  # the faulting node, for an overflow FAULT
    reason: FaultReason | None = None


@dataclass(frozen=True, slots=True)
class DataflowState:
    level: int  # next level to evaluate
    values: tuple[int | None, ...]
    finished: bool = False


@dataclass(frozen=True, slots=True)
class Transition:
    next_state: DataflowState
    event: DataflowEvent
    work: int


@dataclass(frozen=True)
class Dataflow:
    """The ``dataflow/v1`` machine for words of ``width`` bits."""

    width: int = 16

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 2:
            raise ValueError(f"width must be an int of at least 2, got {self.width!r}")

    @property
    def minimum(self) -> int:
        return -(1 << (self.width - 1))

    @property
    def maximum(self) -> int:
        return (1 << (self.width - 1)) - 1

    def in_range(self, value: int) -> bool:
        return self.minimum <= value <= self.maximum

    def validate(self, program: Program) -> None:
        if not program.nodes:
            raise ValueError("a program needs at least one node")
        if type(program.inputs) is not int or program.inputs < 0:
            raise ValueError(f"inputs must be a non-negative int: {program.inputs!r}")
        for index, node in enumerate(program.nodes):
            if node.op not in OPS:
                raise ValueError(f"node {index}: unknown op {node.op!r}")
            if any(type(a) is not int or not 0 <= a < index for a in node.args):
                raise ValueError(f"node {index}: arguments must name earlier nodes")
            if node.op in LEAVES:
                if node.args or type(node.value) is not int:
                    raise ValueError(f"node {index}: a leaf takes an int value only")
                if node.op == "input" and not 0 <= node.value < program.inputs:
                    raise ValueError(f"node {index}: no input {node.value}")
                if node.op == "const" and not self.in_range(node.value):
                    raise ValueError(f"node {index}: constant out of range")
            elif node.value is not None:
                raise ValueError(f"node {index}: only leaves carry a value")
            elif node.op in BINARY and len(node.args) != 2:
                raise ValueError(f"node {index}: {node.op} takes two arguments")
            elif node.op == "sum" and not node.args:
                raise ValueError(f"node {index}: sum needs an argument")
        if not program.outputs or any(
            type(o) is not int or not 0 <= o < len(program.nodes)
            for o in program.outputs
        ):
            raise ValueError("outputs must name nodes")

    def levels(self, program: Program) -> tuple[tuple[int, ...], ...]:
        """Node indices per level, in evaluation order."""
        depth: list[int] = []
        for node in program.nodes:
            depth.append(1 + max((depth[a] for a in node.args), default=-1))
        return tuple(
            tuple(i for i, d in enumerate(depth) if d == level)
            for level in range(max(depth) + 1)
        )

    def initial_state(self, program: Program) -> DataflowState:
        self.validate(program)
        return DataflowState(0, (None,) * len(program.nodes))

    def step(
        self, program: Program, inputs: Sequence[int], state: DataflowState
    ) -> Transition:
        """Evaluate the next level."""
        self._check_inputs(program, inputs)
        if state.finished:
            event = DataflowEvent(EventKind.FAULT, reason=FaultReason.FINISHED)
            return Transition(state, event, 0)
        levels = self.levels(program)
        values = list(state.values)
        work = 0
        for index in levels[state.level]:
            node = program.nodes[index]
            value = self._apply(node, values, inputs)
            work += 1
            if not self.in_range(value):
                event = DataflowEvent(
                    EventKind.FAULT, node=index, reason=FaultReason.OVERFLOW
                )
                done = DataflowState(state.level + 1, tuple(values), finished=True)
                return Transition(done, event, work)
            values[index] = value
        level = state.level + 1
        if level == len(levels):
            outputs = tuple(values[o] for o in program.outputs)
            assert all(v is not None for v in outputs)
            event = DataflowEvent(EventKind.OUTPUT, outputs=outputs)  # type: ignore[arg-type]
            return Transition(DataflowState(level, tuple(values), True), event, work)
        event = DataflowEvent(EventKind.CONTINUE)
        return Transition(DataflowState(level, tuple(values)), event, work)

    def run(
        self, program: Program, inputs: Sequence[int]
    ) -> tuple[DataflowEvent, DataflowState, int]:
        """Step to the end; return the last event, final state, and total work."""
        state, work = self.initial_state(program), 0
        while True:
            transition = self.step(program, inputs, state)
            state, work = transition.next_state, work + transition.work
            if state.finished:
                return transition.event, state, work

    def snapshot(self, state: DataflowState) -> dict[str, Any]:
        return {
            "version": VERSION,
            "width": self.width,
            "level": state.level,
            "values": list(state.values),
            "finished": state.finished,
        }

    def restore(self, snapshot: Mapping[str, Any], program: Program) -> DataflowState:
        """Rebuild a state for ``program``, rejecting anything malformed."""
        if set(snapshot) != {"version", "width", "level", "values", "finished"}:
            raise ValueError(f"snapshot keys are {sorted(snapshot)}")
        if snapshot["version"] != VERSION or snapshot["width"] != self.width:
            raise ValueError("snapshot is for another machine")
        level, values, finished = (
            snapshot["level"],
            snapshot["values"],
            snapshot["finished"],
        )
        levels = self.levels(program)
        if type(level) is not int or not 0 <= level <= len(levels):
            raise ValueError(f"snapshot level {level!r} out of range")
        if type(finished) is not bool:
            raise ValueError("snapshot finished must be a bool")
        if not isinstance(values, list) or len(values) != len(program.nodes):
            raise ValueError("snapshot values must list every node")
        evaluated = {i for lv in levels[:level] for i in lv}
        for index, value in enumerate(values):
            if value is None:
                # Only a fault leaves an evaluated level incomplete.
                if index in evaluated and not finished:
                    raise ValueError(f"snapshot value {index} is missing")
            elif index not in evaluated:
                raise ValueError(f"snapshot value {index} precedes its level")
            elif type(value) is not int or not self.in_range(value):
                raise ValueError(f"snapshot value {index} is {value!r}")
        return DataflowState(level, tuple(values), finished)

    def _check_inputs(self, program: Program, inputs: Sequence[int]) -> None:
        if len(inputs) != program.inputs:
            raise ValueError(f"expected {program.inputs} inputs, got {len(inputs)}")
        for value in inputs:
            if type(value) is not int:
                raise TypeError(f"inputs must be int, got {type(value).__name__}")

    @staticmethod
    def _apply(node: Node, values: list[int | None], inputs: Sequence[int]) -> int:
        if node.op == "input":
            assert node.value is not None
            return inputs[node.value]
        if node.op == "const":
            assert node.value is not None
            return node.value
        args = [values[a] for a in node.args]
        assert all(a is not None for a in args)
        if node.op == "sum":
            return sum(args)  # type: ignore[arg-type]
        a, b = args
        assert a is not None and b is not None
        match node.op:
            case "add":
                return a + b
            case "sub":
                return a - b
            case "mul":
                return a * b
            case "max":
                return max(a, b)
        raise AssertionError(node.op)
