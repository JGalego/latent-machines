"""Conventional sparse algorithms for the E05 fixed-point machines.

These compute the fixed point directly, without the machines' synchronous
rounds, and serve as the conventional baseline and as an independent check
of the oracles' final states:

- ``closure``: breadth-first search from every vertex (``reach/v1``).
- ``all_pairs_dijkstra``: Dijkstra with a binary heap from every vertex
  (``minplus/v1``; weights are nonnegative by specification).
- ``seminaive``: semi-naive evaluation of a chain-Datalog program, which
  joins only the tuples that are new since the previous iteration
  (``chain-datalog/v1``).

Each returns the result plus a count of edges or tuples processed, the
sparse work measure that E05 compares with dense padded work. They are
written from the problem description only, not from the oracle code.
"""

import heapq
from collections import deque
from collections.abc import Mapping, Sequence

from latent_machines.machines.graph import Pair, Rule


def closure(n: int, edges: frozenset[Pair]) -> tuple[frozenset[Pair], int]:
    successors: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        successors[u].append(v)
    pairs, scanned = set(), 0
    for source in range(n):
        seen = {source}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for v in successors[u]:
                scanned += 1
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        pairs.update((source, v) for v in seen)
    return frozenset(pairs), scanned


def reachable_from(
    n: int, successors: Sequence[Sequence[int]], source: int
) -> list[bool]:
    """Single-source BFS, used by the timing benchmark."""
    seen = [False] * n
    seen[source] = True
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in successors[u]:
            if not seen[v]:
                seen[v] = True
                queue.append(v)
    return seen


def dijkstra(
    n: int, adjacency: Sequence[Sequence[tuple[int, int]]], source: int
) -> tuple[list[int | None], int]:
    distances: list[int | None] = [None] * n
    distances[source] = 0
    heap = [(0, source)]
    relaxed = 0
    done = [False] * n
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        for v, w in adjacency[u]:
            relaxed += 1
            current = distances[v]
            if current is None or d + w < current:
                distances[v] = d + w
                heapq.heappush(heap, (d + w, v))
    return distances, relaxed


def all_pairs_dijkstra(
    n: int, weights: Mapping[Pair, int]
) -> tuple[tuple[tuple[int | None, ...], ...], int]:
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for (u, v), w in weights.items():
        adjacency[u].append((v, w))
    rows, relaxed = [], 0
    for source in range(n):
        row, count = dijkstra(n, adjacency, source)
        rows.append(tuple(row))
        relaxed += count
    return tuple(rows), relaxed


def _join(
    left: set[Pair] | frozenset[Pair], right: set[Pair] | frozenset[Pair]
) -> tuple[set[Pair], int]:
    index: dict[int, list[int]] = {}
    for a, b in right:
        index.setdefault(a, []).append(b)
    out, joined = set(), 0
    for x, y in left:
        for z in index.get(y, ()):
            joined += 1
            out.add((x, z))
    return out, joined


def seminaive(
    rules: Sequence[Rule],
    edb: Mapping[str, frozenset[Pair]],
) -> tuple[dict[str, frozenset[Pair]], int]:
    """Least fixed point of a positive chain-Datalog program."""
    idb = sorted({rule.head for rule in rules})
    full: dict[str, set[Pair]] = {name: set() for name in idb}
    delta: dict[str, set[Pair]] = {name: set() for name in idb}
    joined = 0

    def atom(name: str, inverted: bool, source: Mapping[str, set[Pair]]) -> set[Pair]:
        pairs = edb[name] if name in edb else source[name]
        return {(b, a) for a, b in pairs} if inverted else set(pairs)

    def evaluate(rule: Rule, delta_at: int | None) -> set[Pair]:
        nonlocal joined
        parts = []
        for i, a in enumerate(rule.body):
            source = delta if i == delta_at else full
            parts.append(atom(a.relation, a.inverted, source))
        result = parts[0]
        for part in parts[1:]:
            result, count = _join(result, part)
            joined += count
        return result

    # First iteration: rules evaluated on empty intentional relations.
    for rule in rules:
        delta[rule.head] |= evaluate(rule, None)
    while any(delta.values()):
        for name in idb:
            full[name] |= delta[name]
        new: dict[str, set[Pair]] = {name: set() for name in idb}
        for rule in rules:
            for i, a in enumerate(rule.body):
                if a.relation in full and delta[a.relation]:
                    new[rule.head] |= evaluate(rule, i)
        delta = {name: new[name] - full[name] for name in idb}
    return {name: frozenset(full[name]) for name in idb}, joined
