"""Dense tensor backends for the E05 fixed-point machines.

Each round of ``reach/v1``, ``minplus/v1``, and ``chain-datalog/v1`` is
lowered to dense tensor reductions over ``n x n`` matrices:

- Boolean semiring (reachability, Datalog joins): relations are 0/1
  matrices; a join is a matrix product, and ``clamp(x, max=1)`` maps the
  integer count of witnesses back to {0, 1}. Union is ``clamp(a + b, max=1)``.
  A chain body ``r1 ; ... ; rk`` is a product of matrices, transposed for an
  inverted atom.
- Tropical (min-plus) semiring (shortest paths): ``D' = min(D, min_k D[:, k]
  + W[k, :])`` by broadcasting ``D[:, :, None] + W[None, :, :]`` and taking
  the minimum over ``k``. Unreachable is ``+inf``, absent edges are ``+inf``,
  and ``inf + w = inf`` in IEEE arithmetic.

The number format is a parameter; no format-specific range check is made, so
integers beyond a format's exact range (256 for bfloat16, 2048 for float16)
round, and the decoded state then disagrees with the oracle. That is the
behaviour E05 measures, not a defect hidden by the backend.

Host code checks halted and the round budget, compares the new tensor with
the old one to detect the fixed point, and applies the distance bound to
finite entries. ``decode``, ``encode``, and ``residue`` exist for conformance
checks and snapshots only.
"""

from dataclasses import dataclass

import torch

from latent_machines.machines.events import Event, EventKind
from latent_machines.machines.graph import (
    ChainDatalog,
    DatalogState,
    DistanceState,
    GraphFault,
    Iterate,
    Reachability,
    RelationState,
    ShortestPaths,
    Transition,
)

WORK_PER_ACTION = 1


@dataclass(frozen=True)
class TensorState:
    tensors: torch.Tensor  # (relations, n, n), never modified in place
    rounds: int = 0
    halted: bool = False


def _boolean(pairs: frozenset[tuple[int, int]], n: int, dtype, device) -> torch.Tensor:
    matrix = torch.zeros(n, n, dtype=dtype, device=device)
    if pairs:
        index = torch.tensor(sorted(pairs), device=device)
        matrix[index[:, 0], index[:, 1]] = 1
    return matrix


def _pairs(matrix: torch.Tensor) -> frozenset[tuple[int, int]]:
    _check_boolean(matrix)
    return frozenset(map(tuple, (matrix > 0.5).nonzero().tolist()))


def _check_boolean(matrix: torch.Tensor) -> None:
    if _boolean_residue(matrix) >= 0.5:
        raise ValueError("relation entry is not near 0 or 1")


def _boolean_residue(matrix: torch.Tensor) -> float:
    x = matrix.double()
    return float(torch.minimum(x.abs(), (x - 1).abs()).max())


def _union(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.clamp(a + b, max=1)


class _TensorFixedPoint:
    machine: object
    dtype: torch.dtype

    def step(self, state: TensorState, action: Iterate) -> Transition:
        if not isinstance(action, Iterate):
            raise TypeError(f"not an Iterate action: {action!r}")
        if state.halted:
            return self._fault(state, GraphFault.HALTED)
        if state.rounds >= self.machine.max_rounds:
            return self._fault(state, GraphFault.BUDGET_EXHAUSTED)
        candidate = self._round(state.tensors)
        if self._exceeds_bound(candidate):
            return self._fault(state, GraphFault.DISTANCE_BOUND_EXCEEDED)
        same = torch.equal(candidate, state.tensors)
        event = Event(EventKind.HALT if same else EventKind.CONTINUE)
        return Transition(
            TensorState(candidate, state.rounds + 1, same), event, WORK_PER_ACTION
        )

    def _exceeds_bound(self, candidate: torch.Tensor) -> bool:
        return False

    def _round(self, tensors: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def _fault(state: TensorState, reason: GraphFault) -> Transition:
        return Transition(state, Event(EventKind.FAULT, reason=reason), WORK_PER_ACTION)


class TensorReachability(_TensorFixedPoint):
    """``reach/v1``: ``R' = clamp(R + R A, max=1)``."""

    def __init__(self, machine: Reachability, dtype: torch.dtype, device="cpu") -> None:
        self.machine, self.dtype, self.device = machine, dtype, device
        self.adjacency = _boolean(machine.edges, machine.n, dtype, device)

    def initial_state(self) -> TensorState:
        return self.encode(self.machine.initial_state())

    def _round(self, tensors: torch.Tensor) -> torch.Tensor:
        relation = tensors[0]
        return _union(relation, relation @ self.adjacency)[None]

    def decode(self, state: TensorState) -> RelationState:
        return RelationState(_pairs(state.tensors[0]), state.rounds, state.halted)

    def encode(self, state: RelationState) -> TensorState:
        n = self.machine.n
        matrix = _boolean(state.relation, n, self.dtype, self.device)
        return TensorState(matrix[None], state.rounds, state.halted)

    def residue(self, state: TensorState) -> float:
        return _boolean_residue(state.tensors)

    def padded_work(self) -> int:
        """Multiply-adds per round of the dense product."""
        return self.machine.n**3


class TensorMinPlus(_TensorFixedPoint):
    """``minplus/v1``: ``D' = min(D, min_k D[:, k] + W[k, :])``."""

    def __init__(
        self, machine: ShortestPaths, dtype: torch.dtype, device="cpu"
    ) -> None:
        self.machine, self.dtype, self.device = machine, dtype, device
        n = machine.n
        weights = torch.full((n, n), torch.inf, dtype=dtype, device=device)
        for (u, v), w in machine.weights.items():
            weights[u, v] = w
        self.weights = weights

    def initial_state(self) -> TensorState:
        return self.encode(self.machine.initial_state())

    def _round(self, tensors: torch.Tensor) -> torch.Tensor:
        distances = tensors[0]
        through = (distances[:, :, None] + self.weights[None, :, :]).amin(dim=1)
        return torch.minimum(distances, through)[None]

    def _exceeds_bound(self, candidate: torch.Tensor) -> bool:
        finite = torch.isfinite(candidate)
        return bool((candidate[finite] > self.machine.max_distance).any())

    def decode(self, state: TensorState) -> DistanceState:
        matrix = state.tensors[0].double()
        if self.residue(state) >= 0.5:
            raise ValueError("distance is not near an integer")
        rounded = matrix.round()
        rows = tuple(
            tuple(None if d == float("inf") else int(d) for d in row)
            for row in rounded.tolist()
        )
        return DistanceState(rows, state.rounds, state.halted)

    def encode(self, state: DistanceState) -> TensorState:
        data = [
            [float("inf") if d is None else float(d) for d in row]
            for row in state.distances
        ]
        matrix = torch.tensor(data, dtype=self.dtype, device=self.device)
        return TensorState(matrix[None], state.rounds, state.halted)

    def residue(self, state: TensorState) -> float:
        x = state.tensors.double()
        finite = x[torch.isfinite(x)]
        if (x == -torch.inf).any() or torch.isnan(x).any():
            return float("inf")
        return float((finite - finite.round()).abs().max()) if finite.numel() else 0.0

    def padded_work(self) -> int:
        """Additions and comparisons per round of the dense broadcast."""
        return self.machine.n**3


class TensorDatalog(_TensorFixedPoint):
    """``chain-datalog/v1``: each rule a product of 0/1 matrices, clamped."""

    def __init__(self, machine: ChainDatalog, dtype: torch.dtype, device="cpu") -> None:
        self.machine, self.dtype, self.device = machine, dtype, device
        n = machine.n
        self.edb = {
            name: _boolean(pairs, n, dtype, device)
            for name, pairs in machine.edb.items()
        }

    def initial_state(self) -> TensorState:
        return self.encode(self.machine.initial_state())

    def _matrix(self, tensors: torch.Tensor, name: str, inverted: bool) -> torch.Tensor:
        idb = self.machine.idb
        matrix = tensors[idb.index(name)] if name in idb else self.edb[name]
        return matrix.T if inverted else matrix

    def _round(self, tensors: torch.Tensor) -> torch.Tensor:
        heads = {name: tensors[i] for i, name in enumerate(self.machine.idb)}
        for rule in self.machine.rules:
            atoms = [self._matrix(tensors, a.relation, a.inverted) for a in rule.body]
            result = atoms[0]
            for atom in atoms[1:]:
                result = torch.clamp(result @ atom, max=1)
            heads[rule.head] = _union(heads[rule.head], result)
        return torch.stack([heads[name] for name in self.machine.idb])

    def decode(self, state: TensorState) -> DatalogState:
        relations = tuple(_pairs(matrix) for matrix in state.tensors)
        return DatalogState(relations, state.rounds, state.halted)

    def encode(self, state: DatalogState) -> TensorState:
        n = self.machine.n
        stacked = torch.stack(
            [_boolean(r, n, self.dtype, self.device) for r in state.relations]
        )
        return TensorState(stacked, state.rounds, state.halted)

    def residue(self, state: TensorState) -> float:
        return _boolean_residue(state.tensors)

    def padded_work(self) -> int:
        """Multiply-adds per round: one n^3 product per join."""
        joins = sum(len(rule.body) - 1 for rule in self.machine.rules)
        return joins * self.machine.n**3
