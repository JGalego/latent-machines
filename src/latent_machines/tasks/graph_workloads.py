"""Seeded problem generators for E05's graph and logic machines.

Each workload yields fresh, validated machines. Vertex labels are shuffled so
that chains and components do not line up with matrix indices.

``reach/v1``:
- ``sparse``: 48 vertices, edge probability 1.5/n, about 10% self-loops;
  many vertices are isolated or reach nothing.
- ``dense``: 32 vertices, edge probability 0.25.
- ``chain``: a 64-vertex path plus a few self-loops; the closure needs 63
  growing rounds.
- ``components``: four dense clusters of 10 plus 8 isolated vertices, with no
  edges between clusters.

``minplus/v1`` (``max_weight`` 63, ``max_distance`` 2047, an 11-bit table):
- ``unit``: 40 vertices, probability 0.1, all weights 1, so many shortest
  paths tie; includes parallel edges and self-loops.
- ``weighted``: 40 vertices, probability 0.1, weights 0-15, zero-weight
  edges and self-loops included.
- ``long``: a 64-vertex path with weights 16-31 plus six forward shortcuts
  of weight 63; distances reach about 1,950, beyond bfloat16's exact integers (256) but
  within float16's (2048).
- ``bound``: a 64-vertex path with weights 32-63, whose distances exceed
  ``max_distance``, so the machine faults part-way.

``chain-datalog/v1`` with the ``ancestry`` program over ``par(child,
parent)``: ``anc`` is the transitive closure of ``par``, and ``sg`` (same
generation) holds for two individuals with a common parent, or whose parents
are in ``sg``:
- ``forest``: 40 individuals in 5 generations, 1-2 parents each.
- ``lineage``: one line of 40 descendants; ``anc`` needs 39 growing rounds.
- ``cyclic``: 24 individuals with random ``par`` at probability 0.08, which
  contains cycles that a real family tree would not.
"""

import random
from collections.abc import Callable, Iterator

from latent_machines.machines.graph import (
    Atom,
    ChainDatalog,
    Reachability,
    Rule,
    ShortestPaths,
)

MAX_WEIGHT = 63
MAX_DISTANCE = 2047

ANCESTRY = (
    Rule("anc", (Atom("par"),)),
    Rule("anc", (Atom("anc"), Atom("par"))),
    Rule("sg", (Atom("par"), Atom("par", inverted=True))),
    Rule("sg", (Atom("par"), Atom("sg"), Atom("par", inverted=True))),
)


def _random_edges(rng: random.Random, n: int, p: float) -> list[tuple[int, int]]:
    return [(u, v) for u in range(n) for v in range(n) if rng.random() < p]


def _path(rng: random.Random, n: int) -> list[int]:
    order = list(range(n))
    rng.shuffle(order)
    return order


def reach_problem(rng: random.Random, workload: str) -> Reachability:
    if workload == "sparse":
        n = 48
        edges = [(u, v) for u, v in _random_edges(rng, n, 1.5 / n) if u != v]
        edges += [(v, v) for v in range(n) if rng.random() < 0.1]
    elif workload == "dense":
        n = 32
        edges = _random_edges(rng, n, 0.25)
    elif workload == "chain":
        n = 64
        order = _path(rng, n)
        edges = list(zip(order, order[1:], strict=False))
        edges += [(v, v) for v in rng.sample(range(n), 4)]
    elif workload == "components":
        n = 48
        labels = _path(rng, n)
        edges = []
        for c in range(4):
            members = labels[10 * c : 10 * c + 10]
            edges += [(u, v) for u in members for v in members if rng.random() < 0.3]
    else:
        raise ValueError(f"unknown reach workload {workload!r}")
    return Reachability(n, edges)


def minplus_problem(rng: random.Random, workload: str) -> ShortestPaths:
    edges: list[tuple[int, int, int]]
    if workload in ("unit", "weighted"):
        n = 40
        top = 1 if workload == "unit" else 15
        low = 1 if workload == "unit" else 0
        edges = [(u, v, rng.randint(low, top)) for u, v in _random_edges(rng, n, 0.1)]
        edges += [(u, v, rng.randint(low, top)) for u, v, _ in rng.sample(edges, 3)]
        edges += [(v, v, rng.randint(low, top)) for v in rng.sample(range(n), 3)]
    elif workload in ("long", "bound"):
        n = 64
        low, high = (16, 31) if workload == "long" else (32, 63)
        order = _path(rng, n)
        edges = [
            (u, v, rng.randint(low, high))
            for u, v in zip(order, order[1:], strict=False)
        ]
        if workload == "long":
            for _ in range(6):  # forward shortcuts, often shorter than the path
                i = rng.randrange(n - 3)
                j = rng.randrange(i + 2, n)
                edges.append((order[i], order[j], MAX_WEIGHT))
    else:
        raise ValueError(f"unknown minplus workload {workload!r}")
    return ShortestPaths(n, edges, MAX_WEIGHT, MAX_DISTANCE)


def datalog_problem(rng: random.Random, workload: str) -> ChainDatalog:
    if workload == "forest":
        n, generations = 40, 5
        labels = _path(rng, n)
        size = n // generations
        par = []
        for g in range(1, generations):
            parents = labels[(g - 1) * size : g * size]
            for child in labels[g * size : (g + 1) * size]:
                par += [(child, p) for p in rng.sample(parents, rng.randint(1, 2))]
    elif workload == "lineage":
        n = 40
        order = _path(rng, n)
        par = list(zip(order[1:], order, strict=False))
    elif workload == "cyclic":
        n = 24
        par = _random_edges(rng, n, 0.08)
    else:
        raise ValueError(f"unknown datalog workload {workload!r}")
    return ChainDatalog(n, ANCESTRY, {"par": par})


GENERATORS: dict[str, tuple[Callable, tuple[str, ...]]] = {
    "reach": (reach_problem, ("sparse", "dense", "chain", "components")),
    "minplus": (minplus_problem, ("unit", "weighted", "long", "bound")),
    "datalog": (datalog_problem, ("forest", "lineage", "cyclic")),
}


def problems(machine: str, workload: str, count: int, seed: int) -> Iterator:
    """``count`` problems, reproducible from ``(seed, machine, workload)``."""
    generate, workloads = GENERATORS[machine]
    if workload not in workloads:
        raise ValueError(f"unknown {machine} workload {workload!r}")
    rng = random.Random(f"{seed}:{machine}:{workload}")
    for _ in range(count):
        yield generate(rng, workload)
