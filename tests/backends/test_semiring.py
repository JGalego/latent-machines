"""Dense tensor backends and conventional algorithms for E05 machines."""

import random

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends import graph_native  # noqa: E402
from latent_machines.backends.semiring import (  # noqa: E402
    TensorDatalog,
    TensorMinPlus,
    TensorReachability,
)
from latent_machines.experiments import lockstep  # noqa: E402
from latent_machines.machines.events import EventKind  # noqa: E402
from latent_machines.machines.graph import Iterate, ShortestPaths  # noqa: E402
from latent_machines.tasks import graph_workloads  # noqa: E402

BACKENDS = {
    "reach": TensorReachability,
    "minplus": TensorMinPlus,
    "datalog": TensorDatalog,
}


def rounds(machine) -> int:
    state, count = machine.initial_state(), 0
    while True:
        count += 1
        t = machine.step(state, Iterate())
        state = t.next_state
        if t.event.kind is not EventKind.CONTINUE:
            return count


def check(machine, backend, snapshot_at: int) -> dict:
    n = rounds(machine)
    return lockstep.check(
        machine,
        backend,
        [Iterate()] * n,
        name="t",
        snapshot_at=snapshot_at if snapshot_at >= 0 else -1,
        trace_limit=0,
        reproduce=lambda h: [],
    )["metrics"]


@pytest.mark.parametrize(
    ("kind", "workload"),
    [("reach", "sparse"), ("reach", "components"), ("minplus", "weighted"),
     ("minplus", "unit"), ("datalog", "forest"), ("datalog", "cyclic")],
)  # fmt: skip
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tensor_backend_conforms_every_round(kind, workload, dtype) -> None:
    for machine in graph_workloads.problems(kind, workload, 2, seed=3):
        m = check(machine, BACKENDS[kind](machine, dtype), rounds(machine) // 2)
        assert (m["classification"], m["snapshot_resume"]) == ("exact", "agrees")


def test_bfloat16_min_plus_loses_integers_above_256() -> None:
    machine = ShortestPaths(3, [(0, 1, 200), (1, 2, 57)], 255, 1000)
    m = check(machine, TensorMinPlus(machine, torch.bfloat16), -1)
    assert m["classification"] == "mismatch"  # 257 rounds to 256
    assert check(machine, TensorMinPlus(machine, torch.float16), -1)[
        "classification"
    ] == ("exact")


def test_tensor_bound_fault_leaves_state_unchanged() -> None:
    machine = ShortestPaths(3, [(0, 1, 6), (1, 2, 6)], 9, 10)
    backend = TensorMinPlus(machine, torch.float32)
    state = backend.step(backend.initial_state(), Iterate()).next_state
    t = backend.step(state, Iterate())
    assert t.event.kind is EventKind.FAULT and t.next_state is state


@pytest.mark.parametrize(
    ("kind", "workload"),
    [("reach", "dense"), ("minplus", "weighted"), ("datalog", "cyclic")],
)
def test_conventional_algorithms_match_the_oracles(kind, workload) -> None:
    for machine in graph_workloads.problems(kind, workload, 2, seed=5):
        state = machine.initial_state()
        for _ in range(rounds(machine)):
            state = machine.step(state, Iterate()).next_state
        if kind == "reach":
            assert graph_native.closure(machine.n, machine.edges)[0] == state.relation
        elif kind == "minplus":
            expected = graph_native.all_pairs_dijkstra(machine.n, machine.weights)[0]
            assert expected == state.distances
        else:
            expected = graph_native.seminaive(machine.rules, machine.edb)[0]
            for i, name in enumerate(machine.idb):
                assert expected[name] == state.relations[i]


def test_single_source_bfs_matches_closure_row() -> None:
    rng = random.Random(0)
    n = 30
    edges = frozenset((rng.randrange(n), rng.randrange(n)) for _ in range(45))
    successors: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        successors[u].append(v)
    seen = graph_native.reachable_from(n, successors, 0)
    closure, _ = graph_native.closure(n, edges)
    assert {v for v in range(n) if seen[v]} == {v for u, v in closure if u == 0}
