"""Fixed-point graph and logic machines: the reference oracles for E05.

Three specifications share one execution model. A machine is built from one
validated problem (a graph or a Datalog database); its state is a relation
or distance table that grows or shrinks monotonically, one synchronous round
per ``Iterate`` action, until a round changes nothing.

Shared semantics:

- Action: ``Iterate()`` runs one synchronous (Jacobi) round: every entry of
  the next state is computed from the previous state only.
- Order of checks in a step: a halted machine faults ``halted``; a machine
  that has run ``max_rounds`` rounds faults ``budget_exhausted``; otherwise
  the round is computed. ``minplus/v1`` then faults
  ``distance_bound_exceeded`` if any finite entry of the new table exceeds
  ``max_distance``. Every fault leaves the state unchanged, so an exhausted
  or faulted machine returns its last state, which is not certified as a
  fixed point.
- Events: ``continue`` when the round changed the state, ``halt`` when it
  did not (the fixed point is reached and the machine halts), ``fault``
  otherwise. The confirming round counts in ``rounds``.
- Work: one unit per action, faults included. Edges or tuples processed are
  measured separately by the experiment, not counted as work.
- Snapshots: a JSON-compatible dict with the specification version, a digest
  of the problem, the round count, the halted flag, and the canonical state.

``reach/v1``: vertices ``0..n-1`` and a set of directed edges (self-loops
allowed). The state is the relation ``R`` of pairs ``(u, v)``; initially the
identity, so every vertex reaches itself. A round adds ``(u, w)`` for each
``(u, v)`` in ``R`` and edge ``(v, w)``. The fixed point is the reflexive
transitive closure. ``max_rounds = n`` always suffices.

``minplus/v1``: edges carry integer weights ``0 <= w <= max_weight``;
parallel edges are allowed and the smallest weight applies. Negative
weights, and so negative cycles, are rejected when the problem is built.
The state is the all-pairs table ``D``: ``0`` on the diagonal and ``None``
(the infinity sentinel, "unreachable") elsewhere. A round sets
``D'[u][v] = min(D[u][v], min over edges (k, v, w) of D[u][k] + w)``. The
bound applies to every stored, possibly tentative, distance; values are never
saturated. ``max_rounds = n`` always suffices.

``chain-datalog/v1``: a positive Datalog program over binary relations on
the domain ``0..n-1``. Each rule is a chain ``h(x0, xk) :- r1(x0, x1), ...,
rk(x(k-1), xk)``, where a body atom may use a relation inverted (``r^-1(a, b)
= r(b, a)``). Extensional relations are fixed; intentional relations start
empty, and a round evaluates every rule on the previous round's relations
(naive evaluation) and unions the results into the heads. Each non-final
round adds at least one tuple, so ``max_rounds = n * n * len(idb) + 1``
always suffices.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from latent_machines.machines.events import Event, EventKind

WORK_PER_ACTION = 1


@dataclass(frozen=True, slots=True)
class Iterate:
    pass


class GraphFault(Enum):
    HALTED = "halted"
    BUDGET_EXHAUSTED = "budget_exhausted"
    DISTANCE_BOUND_EXCEEDED = "distance_bound_exceeded"


type Pair = tuple[int, int]
type Table = tuple[tuple[int | None, ...], ...]


@dataclass(frozen=True, slots=True)
class RelationState:
    relation: frozenset[Pair]
    rounds: int = 0
    halted: bool = False


@dataclass(frozen=True, slots=True)
class DistanceState:
    distances: Table
    rounds: int = 0
    halted: bool = False


@dataclass(frozen=True, slots=True)
class DatalogState:
    relations: tuple[frozenset[Pair], ...]  # one per intentional relation
    rounds: int = 0
    halted: bool = False


@dataclass(frozen=True, slots=True)
class Transition:
    next_state: Any
    event: Event
    work: int


_CONTINUE = Event(EventKind.CONTINUE)
_HALT = Event(EventKind.HALT)


def _fault(state: Any, reason: GraphFault) -> Transition:
    return Transition(state, Event(EventKind.FAULT, reason=reason), WORK_PER_ACTION)


def _digest(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _check_vertex(n: int, vertex: object) -> None:
    if type(vertex) is not int or not 0 <= vertex < n:
        raise ValueError(f"vertex {vertex!r} is not in 0..{n - 1}")


def _check_size(n: object) -> None:
    if type(n) is not int or n < 1:
        raise ValueError(f"n must be a positive int, got {n!r}")


class _FixedPoint:
    """Shared step order: halted, budget, round, bound, fixed point."""

    version: str
    max_rounds: int

    def step(self, state: Any, action: Iterate) -> Transition:
        if not isinstance(action, Iterate):
            raise TypeError(f"not a {self.version} action: {action!r}")
        if state.halted:
            return _fault(state, GraphFault.HALTED)
        if state.rounds >= self.max_rounds:
            return _fault(state, GraphFault.BUDGET_EXHAUSTED)
        candidate = self._round(state)
        if self._exceeds_bound(candidate):
            return _fault(state, GraphFault.DISTANCE_BOUND_EXCEEDED)
        same = self._content(candidate) == self._content(state)
        next_state = self._with(candidate, state.rounds + 1, same)
        return Transition(next_state, _HALT if same else _CONTINUE, WORK_PER_ACTION)

    def _exceeds_bound(self, candidate: Any) -> bool:
        return False

    def _round(self, state: Any) -> Any:
        raise NotImplementedError

    def _content(self, state: Any) -> Any:
        raise NotImplementedError

    def _with(self, candidate: Any, rounds: int, halted: bool) -> Any:
        raise NotImplementedError

    def _snapshot_header(self, state: Any) -> dict[str, Any]:
        return {
            "version": self.version,
            "problem": self.digest,
            "rounds": state.rounds,
            "halted": state.halted,
        }

    def _restore_header(self, snapshot: Mapping[str, Any], keys: set[str]) -> None:
        if set(snapshot) != {"version", "problem", "rounds", "halted", *keys}:
            raise ValueError(f"snapshot keys are {sorted(snapshot)}")
        if snapshot["version"] != self.version or snapshot["problem"] != self.digest:
            raise ValueError("snapshot is for another machine or problem")
        rounds, halted = snapshot["rounds"], snapshot["halted"]
        if type(rounds) is not int or not 0 <= rounds <= self.max_rounds:
            raise ValueError(f"snapshot rounds is {rounds!r}")
        if type(halted) is not bool:
            raise ValueError(f"snapshot halted is {halted!r}")

    @property
    def digest(self) -> str:
        raise NotImplementedError


def _pairs_from_json(n: int, data: object) -> frozenset[Pair]:
    if not isinstance(data, list):
        raise ValueError("relation must be a list of pairs")
    pairs = set()
    for item in data:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"relation entry {item!r} is not a pair")
        for vertex in item:
            _check_vertex(n, vertex)
        pairs.add((item[0], item[1]))
    return frozenset(pairs)


class Reachability(_FixedPoint):
    """``reach/v1`` on one directed graph."""

    version = "reach/v1"

    def __init__(self, n: int, edges: Sequence[Pair]) -> None:
        _check_size(n)
        for u, v in edges:
            _check_vertex(n, u)
            _check_vertex(n, v)
        self.n = n
        self.edges = frozenset(edges)
        self.max_rounds = n
        self.successors: list[list[int]] = [[] for _ in range(n)]
        for u, v in sorted(self.edges):
            self.successors[u].append(v)

    @property
    def digest(self) -> str:
        return _digest({"n": self.n, "edges": sorted(self.edges)})

    def initial_state(self) -> RelationState:
        return RelationState(frozenset((v, v) for v in range(self.n)))

    def _round(self, state: RelationState) -> frozenset[Pair]:
        grown = set(state.relation)
        for u, v in state.relation:
            for w in self.successors[v]:
                grown.add((u, w))
        return frozenset(grown)

    def _content(self, state: Any) -> frozenset[Pair]:
        return state if isinstance(state, frozenset) else state.relation

    def _with(self, candidate: frozenset[Pair], rounds: int, halted: bool) -> Any:
        return RelationState(candidate, rounds, halted)

    def snapshot(self, state: RelationState) -> dict[str, Any]:
        return self._snapshot_header(state) | {
            "relation": [list(p) for p in sorted(state.relation)]
        }

    def restore(self, snapshot: Mapping[str, Any]) -> RelationState:
        self._restore_header(snapshot, {"relation"})
        return RelationState(
            _pairs_from_json(self.n, snapshot["relation"]),
            snapshot["rounds"],
            snapshot["halted"],
        )


class ShortestPaths(_FixedPoint):
    """``minplus/v1`` on one weighted directed graph."""

    version = "minplus/v1"

    def __init__(
        self,
        n: int,
        edges: Sequence[tuple[int, int, int]],
        max_weight: int,
        max_distance: int,
    ) -> None:
        _check_size(n)
        for name, bound in (("max_weight", max_weight), ("max_distance", max_distance)):
            if type(bound) is not int or bound < 0:
                raise ValueError(f"{name} must be a nonnegative int, got {bound!r}")
        weights: dict[Pair, int] = {}
        for u, v, w in edges:
            _check_vertex(n, u)
            _check_vertex(n, v)
            if type(w) is not int or not 0 <= w <= max_weight:
                raise ValueError(
                    f"edge ({u}, {v}) weight {w!r} is not in 0..{max_weight}"
                )
            weights[u, v] = min(w, weights.get((u, v), w))
        self.n, self.max_weight, self.max_distance = n, max_weight, max_distance
        self.weights = weights  # smallest weight per (u, v)
        self.max_rounds = n
        self.incoming: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for (u, v), w in sorted(weights.items()):
            self.incoming[v].append((u, w))

    @property
    def digest(self) -> str:
        return _digest(
            {
                "n": self.n,
                "max_weight": self.max_weight,
                "max_distance": self.max_distance,
                "edges": sorted([u, v, w] for (u, v), w in self.weights.items()),
            }
        )

    def initial_state(self) -> DistanceState:
        return DistanceState(
            tuple(
                tuple(0 if u == v else None for v in range(self.n))
                for u in range(self.n)
            )
        )

    def _round(self, state: DistanceState) -> Table:
        rows = []
        for row in state.distances:
            new_row = []
            for v in range(self.n):
                best = row[v]
                for k, w in self.incoming[v]:
                    if row[k] is not None and (best is None or row[k] + w < best):
                        best = row[k] + w
                new_row.append(best)
            rows.append(tuple(new_row))
        return tuple(rows)

    def _exceeds_bound(self, candidate: Table) -> bool:
        return any(
            d is not None and d > self.max_distance for r in candidate for d in r
        )

    def _content(self, state: Any) -> Table:
        return state if isinstance(state, tuple) else state.distances

    def _with(self, candidate: Table, rounds: int, halted: bool) -> Any:
        return DistanceState(candidate, rounds, halted)

    def snapshot(self, state: DistanceState) -> dict[str, Any]:
        return self._snapshot_header(state) | {
            "distances": [list(row) for row in state.distances]
        }

    def restore(self, snapshot: Mapping[str, Any]) -> DistanceState:
        self._restore_header(snapshot, {"distances"})
        rows = snapshot["distances"]
        if not isinstance(rows, list) or len(rows) != self.n:
            raise ValueError(f"snapshot distances must have {self.n} rows")
        table = []
        for row in rows:
            if not isinstance(row, list) or len(row) != self.n:
                raise ValueError(f"snapshot distance rows must have {self.n} entries")
            for d in row:
                if d is not None and (
                    type(d) is not int or not 0 <= d <= self.max_distance
                ):
                    raise ValueError(f"snapshot distance {d!r} is out of range")
            table.append(tuple(row))
        return DistanceState(tuple(table), snapshot["rounds"], snapshot["halted"])


@dataclass(frozen=True)
class Atom:
    relation: str
    inverted: bool = False


@dataclass(frozen=True)
class Rule:
    head: str
    body: tuple[Atom, ...]


class ChainDatalog(_FixedPoint):
    """``chain-datalog/v1``: one program over one extensional database."""

    version = "chain-datalog/v1"

    def __init__(
        self,
        n: int,
        rules: Sequence[Rule],
        edb: Mapping[str, Sequence[Pair]],
    ) -> None:
        _check_size(n)
        self.n = n
        self.edb = {name: frozenset(pairs) for name, pairs in sorted(edb.items())}
        for pairs in self.edb.values():
            for u, v in pairs:
                _check_vertex(n, u)
                _check_vertex(n, v)
        self.idb = tuple(sorted({rule.head for rule in rules}))
        self.rules = tuple(rules)
        for rule in self.rules:
            if rule.head in self.edb:
                raise ValueError(f"rule head {rule.head!r} is extensional")
            if not rule.body:
                raise ValueError(f"rule for {rule.head!r} has an empty body")
            for atom in rule.body:
                if atom.relation not in self.edb and atom.relation not in self.idb:
                    raise ValueError(f"unknown relation {atom.relation!r}")
        self.max_rounds = n * n * len(self.idb) + 1

    @property
    def digest(self) -> str:
        return _digest(
            {
                "n": self.n,
                "rules": [
                    [r.head, [[a.relation, a.inverted] for a in r.body]]
                    for r in self.rules
                ],
                "edb": {k: sorted(v) for k, v in self.edb.items()},
            }
        )

    def initial_state(self) -> DatalogState:
        return DatalogState(tuple(frozenset() for _ in self.idb))

    def relation(self, state: DatalogState, name: str) -> frozenset[Pair]:
        if name in self.edb:
            return self.edb[name]
        return state.relations[self.idb.index(name)]

    def _round(self, state: DatalogState) -> tuple[frozenset[Pair], ...]:
        heads = {name: set(state.relations[i]) for i, name in enumerate(self.idb)}
        for rule in self.rules:
            heads[rule.head] |= compose_chain(
                [self._atom(state, atom) for atom in rule.body]
            )
        return tuple(frozenset(heads[name]) for name in self.idb)

    def _atom(self, state: DatalogState, atom: Atom) -> frozenset[Pair]:
        pairs = self.relation(state, atom.relation)
        return frozenset((b, a) for a, b in pairs) if atom.inverted else pairs

    def _content(self, state: Any) -> tuple[frozenset[Pair], ...]:
        return state if isinstance(state, tuple) else state.relations

    def _with(self, candidate: tuple, rounds: int, halted: bool) -> Any:
        return DatalogState(candidate, rounds, halted)

    def snapshot(self, state: DatalogState) -> dict[str, Any]:
        return self._snapshot_header(state) | {
            "relations": {
                name: [list(p) for p in sorted(state.relations[i])]
                for i, name in enumerate(self.idb)
            }
        }

    def restore(self, snapshot: Mapping[str, Any]) -> DatalogState:
        self._restore_header(snapshot, {"relations"})
        relations = snapshot["relations"]
        if not isinstance(relations, dict) or sorted(relations) != list(self.idb):
            raise ValueError(f"snapshot relations must be {list(self.idb)}")
        return DatalogState(
            tuple(_pairs_from_json(self.n, relations[name]) for name in self.idb),
            snapshot["rounds"],
            snapshot["halted"],
        )


def compose_chain(atoms: Sequence[frozenset[Pair]]) -> set[Pair]:
    """Relational composition ``a1 ; a2 ; ... ; ak`` of binary relations."""
    result: set[Pair] = set(atoms[0])
    for atom in atoms[1:]:
        successors: dict[int, list[int]] = {}
        for a, b in atom:
            successors.setdefault(a, []).append(b)
        result = {(x, z) for x, y in result for z in successors.get(y, ())}
    return result
