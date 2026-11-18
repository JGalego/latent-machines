"""E05: graph, semiring, and logic execution.

Checks three fixed-point machines (``reach/v1``, ``minplus/v1``, and
``chain-datalog/v1``; see ``machines/graph.py``) in two ways:

- Dense tensor backends (``backends/semiring.py``) run in lockstep with the
  integer oracle, one synchronous round per action, in each number format.
  Every round compares the event, work, and decoded relation or distance
  table, including unreachable entries; one problem per variant is
  snapshotted mid-run and restored through JSON.
- Conventional sparse algorithms (``backends/graph_native.py``: BFS,
  Dijkstra, semi-naive Datalog) compute each fixed point directly; their
  results must equal the oracle's final state. These are the ``conventional``
  variants, with dtype ``python-int``.

A timing benchmark then compares single-source BFS and Dijkstra with dense
and sparse tensor iterations on larger sparse random graphs, checking each
timed result against the conventional answer.

    uv run --locked python -m latent_machines.experiments.e05
    uv run --locked python -m latent_machines.experiments.e05 --horizon 200 --seed 1

Exit status: 0 when every variant conforms, 1 on any mismatch or snapshot
disagreement, 2 if the run itself failed.
"""

import argparse
import json
import random
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any

import torch

from latent_machines.backends import graph_native
from latent_machines.backends.semiring import (
    TensorDatalog,
    TensorMinPlus,
    TensorReachability,
)
from latent_machines.experiments import lockstep
from latent_machines.machines.events import EventKind
from latent_machines.machines.graph import Iterate
from latent_machines.tasks import graph_workloads

EXPERIMENT = "E05"
MODULE = "latent_machines.experiments.e05"
DTYPES = ("float32", "float64", "bfloat16", "float16")
CONVENTIONAL = "python-int"
MACHINES = tuple(graph_workloads.GENERATORS)
BACKENDS = {
    "reach": TensorReachability,
    "minplus": TensorMinPlus,
    "datalog": TensorDatalog,
}
VERSIONS = {"reach": "reach/v1", "minplus": "minplus/v1", "datalog": "chain-datalog/v1"}
ACCEPTANCE = (
    "every round's event, work, and decoded relation or distance table (including "
    "unreachable entries) equals the oracle's; a mid-run snapshot restored through "
    "JSON equals the live tensors; conventional sparse algorithms reach the same "
    "fixed points"
)
NOT_MEASURED = [
    "scalar-machine execution: no instruction-level graph interpreter exists yet",
    "learned GNNs or embedding-space reasoning: out of scope for exact kernels",
    "sparse tensor kernels in lockstep: the sparse tensor path is timed in the "
    "benchmark only, with its final answer checked",
    "GPU and other devices; batching several problems into one tensor",
    "squaring-based closure (log rounds) and Datalog programs beyond chain rules",
]


@dataclass(frozen=True)
class Variant:
    machine: str
    workload: str
    dtype: str
    horizon: int  # problems per variant
    seed: int
    trace_limit: int

    @property
    def backend(self) -> str:
        return "conventional" if self.dtype == CONVENTIONAL else "dense-tensor"

    @property
    def name(self) -> str:
        return f"{self.backend}/{self.dtype}/{self.machine}/{self.workload}"


def reproduce_command(variant: Variant, problems: int) -> list[str]:
    return [
        "python", "-m", MODULE,
        "--machines", variant.machine, "--workloads", variant.workload,
        "--dtypes", variant.dtype,
        "--horizon", str(problems), "--seed", str(variant.seed), "--no-benchmark",
    ]  # fmt: skip


def run_oracle(machine: Any) -> tuple[Any, list[Any], float]:
    """Step the oracle to its terminal event; return final state and events."""
    state, events = machine.initial_state(), []
    started = time.perf_counter()
    while True:
        transition = machine.step(state, Iterate())
        state = transition.next_state
        events.append(transition.event)
        if transition.event.kind is not EventKind.CONTINUE:
            return state, events, time.perf_counter() - started


def density(machine: Any) -> float:
    n = machine.n
    if hasattr(machine, "weights"):
        return len(machine.weights) / n**2
    if hasattr(machine, "edges"):
        return len(machine.edges) / n**2
    return sum(len(p) for p in machine.edb.values()) / n**2


def run_conventional(variant: Variant, problems: list[Any]) -> dict[str, Any]:
    """Check each oracle fixed point against a conventional sparse algorithm."""
    disagreements, work, seconds, faults, failure = 0, 0, 0.0, 0, None
    tentative_faults = 0
    started = time.perf_counter()
    for index, machine in enumerate(problems):
        final, events, _ = run_oracle(machine)
        last = events[-1]
        t0 = time.perf_counter()
        if variant.machine == "reach":
            expected, count = graph_native.closure(machine.n, machine.edges)
            ok = last.kind is EventKind.HALT and final.relation == expected
        elif variant.machine == "minplus":
            expected, count = graph_native.all_pairs_dijkstra(
                machine.n, machine.weights
            )
            beyond = any(
                d is not None and d > machine.max_distance for r in expected for d in r
            )
            if last.kind is EventKind.HALT:
                ok = not beyond and final.distances == expected
            else:
                faults += 1
                tentative_faults += not beyond
                # A fault is correct when a true distance is beyond the bound, or
                # when a tentative distance was (the bound covers stored values).
                ok = last.kind is EventKind.FAULT
        else:
            expected, count = graph_native.seminaive(machine.rules, machine.edb)
            ok = last.kind is EventKind.HALT and all(
                final.relations[i] == expected[name]
                for i, name in enumerate(machine.idb)
            )
        seconds += time.perf_counter() - t0
        work += count
        if not ok:
            disagreements += 1
            if failure is None:
                failure = {
                    "variant": variant.name,
                    "problem_index": index,
                    "terminal_event": lockstep.event_json(last),
                    "reproduce": reproduce_command(variant, index + 1),
                }
    return {
        "metrics": {
            "variant": variant.name,
            "unit": "problem: final oracle state versus the conventional algorithm",
            "actions_checked": len(problems),
            "mismatches": disagreements,
            "first_mismatch": None if failure is None else failure["problem_index"],
            "max_residue": 0.0,
            "classification": "exact" if disagreements == 0 else "mismatch",
            "snapshot_resume": "not applicable",
            "end_to_end_seconds": round(time.perf_counter() - started, 3),
            "conventional_seconds": round(seconds, 4),
            "sparse_work": work,
            "sparse_work_unit": {
                "reach": "edges scanned by BFS from every vertex",
                "minplus": "edges relaxed by Dijkstra from every vertex",
                "datalog": "tuples joined by semi-naive evaluation",
            }[variant.machine],
            "bound_faults": faults if variant.machine == "minplus" else None,
            "bound_faults_on_tentative_only": (
                tentative_faults if variant.machine == "minplus" else None
            ),
        },
        "trace": [],
        "failure": failure,
        "snapshot": None,
    }


def run_tensor(variant: Variant, problems: list[Any]) -> dict[str, Any]:
    dtype = getattr(torch, variant.dtype)
    snapshot_problem = len(problems) // 2
    trace, failure, snapshot = [], None, None
    snapshot_resume = "not reached"
    mismatched, rounds, max_residue = 0, 0, 0.0
    padded_work = 0
    tensor_bytes = 0
    backend_seconds = oracle_seconds = 0.0
    terminal: dict[str, int] = {}
    first_round = None
    started = time.perf_counter()
    for index, machine in enumerate(problems):
        final, events, oracle_time = run_oracle(machine)
        oracle_seconds += oracle_time
        backend = BACKENDS[variant.machine](machine, dtype)
        result = lockstep.check(
            machine,
            backend,
            [Iterate()] * len(events),
            name=variant.name,
            snapshot_at=len(events) // 2 if index == snapshot_problem else -1,
            trace_limit=variant.trace_limit if index == 0 else 0,
            reproduce=lambda _, i=index: reproduce_command(variant, i + 1),
        )
        m = result["metrics"]
        for line in result["trace"]:
            trace.append(line | {"problem": index})
        max_residue = max(max_residue, m["max_residue"])
        rounds += m["actions_checked"]
        if index == snapshot_problem:
            snapshot_resume, snapshot = m["snapshot_resume"], result["snapshot"]
        if m["mismatches"]:
            mismatched += 1
            if failure is None:
                failure = result["failure"] | {"problem_index": index}
                first_round = m["first_mismatch"]
            continue
        kind = events[-1].kind.value
        if events[-1].reason is not None:
            kind += f":{events[-1].reason.value}"
        terminal[kind] = terminal.get(kind, 0) + 1
        padded_work += backend.padded_work() * len(events)
        tensor_bytes = max(
            tensor_bytes,
            sum(
                t.numel() * t.element_size()
                for t in vars(backend).values()
                if isinstance(t, torch.Tensor)
            )
            + sum(
                t.numel() * t.element_size()
                for t in getattr(backend, "edb", {}).values()
            )
            + lockstep.tensor_bytes(backend.initial_state()),
        )
        # Time the backend alone on the same deterministic path, and confirm
        # that the timed run ends in the oracle's final state.
        state = backend.initial_state()
        t0 = time.perf_counter()
        for _ in events:
            state = backend.step(state, Iterate()).next_state
        backend_seconds += time.perf_counter() - t0
        if backend.decode(state) != final:
            raise RuntimeError(f"timed path of problem {index} diverged")

    if failure is not None:
        classification = "mismatch"
    elif max_residue == 0.0:
        classification = "exact"
    else:
        classification = "conforms with residue"
    return {
        "metrics": {
            "variant": variant.name,
            "unit": "round: one synchronous fixed-point iteration checked in lockstep",
            "actions_checked": rounds,
            "problems": len(problems),
            "mismatches": mismatched,
            "mismatch_unit": "problems with a nonconforming round",
            "first_mismatch": None if failure is None else failure["problem_index"],
            "first_mismatch_round": first_round,
            "max_residue": max_residue,
            "classification": classification,
            "snapshot_resume": snapshot_resume,
            "end_to_end_seconds": round(time.perf_counter() - started, 3),
            "terminal_events": terminal,
            "backend_seconds": round(backend_seconds, 4),
            "oracle_seconds": round(oracle_seconds, 4),
            "padded_work": padded_work,
            "padded_work_unit": "dense multiply-adds or add-mins, conforming problems",
            "max_tensor_bytes": tensor_bytes,
        },
        "trace": trace,
        "failure": failure,
        "snapshot": snapshot,
    }


def run_variant(variant: Variant) -> dict[str, Any]:
    torch.set_num_threads(1)
    problems = list(
        graph_workloads.problems(
            variant.machine, variant.workload, variant.horizon, variant.seed
        )
    )
    if variant.dtype == CONVENTIONAL:
        result = run_conventional(variant, problems)
    else:
        result = run_tensor(variant, problems)
    sizes = sorted({p.n for p in problems})
    lengths = [len(run_oracle(p)[1]) for p in problems]
    result["metrics"] |= {
        "backend": variant.backend,
        "dtype": variant.dtype,
        "machine": VERSIONS[variant.machine],
        "workload": variant.workload,
        "horizon": variant.horizon,
        "vertices": sizes,
        "mean_density": round(statistics.fmean(density(p) for p in problems), 4),
        "mean_rounds_to_terminal": round(statistics.fmean(lengths), 2),
        "max_rounds_to_terminal": max(lengths),
    }
    return result


# ---------- timing benchmark ----------


def _median_time(function: Any, repeats: int = 3) -> tuple[Any, float]:
    times, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = function()
        times.append(time.perf_counter() - t0)
    return result, statistics.median(times)


def _frontier(start: torch.Tensor, step: Any) -> tuple[torch.Tensor, int]:
    """Grow a 0/1 reached vector by one edge per round until it is unchanged."""
    reached, rounds = start, 0
    while True:
        rounds += 1
        grown = torch.clamp(reached + step(reached), max=1)
        if torch.equal(grown, reached):
            return reached, rounds
        reached = grown


def _bellman_ford(
    init: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, int]:
    distances, rounds = init, 0
    while True:
        rounds += 1
        new = torch.minimum(distances, (distances[:, None] + weights).amin(dim=0))
        if torch.equal(new, distances):
            return distances, rounds
        distances = new


def bench_size(n: int, degree: float, seed: int) -> dict[str, Any]:
    """Single-source reachability and shortest paths on one sparse digraph."""
    rng = random.Random(f"{seed}:bench:{n}")
    edges = {(rng.randrange(n), rng.randrange(n)) for _ in range(int(degree * n))}
    weights = {e: rng.randint(1, 8) for e in edges}
    successors: list[list[int]] = [[] for _ in range(n)]
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for (u, v), w in sorted(weights.items()):
        successors[u].append(v)
        adjacency[u].append((v, w))
    source = 0
    expected_reach, bfs_time = _median_time(
        lambda: graph_native.reachable_from(n, successors, source)
    )
    (expected_dist, relaxed), dijkstra_time = _median_time(
        lambda: graph_native.dijkstra(n, adjacency, source)
    )
    index = torch.tensor(sorted(edges)).T
    dense = torch.zeros(n, n)
    dense[index[0], index[1]] = 1
    transposed = torch.sparse_coo_tensor(
        index.flip(0), torch.ones(len(edges)), (n, n), check_invariants=True
    ).coalesce()
    start = torch.zeros(n)
    start[source] = 1
    (dense_reach, rounds), dense_time = _median_time(
        lambda: _frontier(start, lambda r: r @ dense)
    )
    (sparse_reach, _), sparse_time = _median_time(
        lambda: _frontier(
            start, lambda r: torch.sparse.mm(transposed, r[:, None])[:, 0]
        )
    )
    weight_matrix = torch.full((n, n), torch.inf)
    for (u, v), w in weights.items():
        weight_matrix[u, v] = w
    init = torch.full((n,), torch.inf)
    init[source] = 0
    (dense_dist, bf_rounds), bf_time = _median_time(
        lambda: _bellman_ford(init, weight_matrix)
    )
    reach_ok = (dense_reach > 0.5).tolist() == expected_reach and (
        sparse_reach > 0.5
    ).tolist() == expected_reach
    decoded = [None if d == float("inf") else int(d) for d in dense_dist.tolist()]
    match = reach_ok and decoded == expected_dist
    print(
        f"bench n={n:<6} BFS {bfs_time * 1e3:8.2f} ms  "
        f"dense {dense_time * 1e3:8.2f} ms  sparse {sparse_time * 1e3:8.2f} ms  "
        f"Dijkstra {dijkstra_time * 1e3:8.2f} ms  dense BF {bf_time * 1e3:8.2f} ms  "
        f"match {match}",
        flush=True,
    )
    return {
        "vertices": n,
        "edges": len(edges),
        "density": len(edges) / n**2,
        "reachable": sum(expected_reach),
        "reach_rounds": rounds,
        "bfs_seconds": bfs_time,
        "dense_reach_seconds": dense_time,
        "sparse_tensor_reach_seconds": sparse_time,
        "dense_reach_padded_work": n * n * rounds,
        "bellman_ford_rounds": bf_rounds,
        "dijkstra_seconds": dijkstra_time,
        "dijkstra_edges_relaxed": relaxed,
        "dense_bellman_ford_seconds": bf_time,
        "dense_bellman_ford_padded_work": n * n * bf_rounds,
        "results_match": match,
    }


# ---------- command line ----------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="E05 conformance of graph, semiring, and Datalog tensor backends.",
    )
    parser.add_argument("--machines", default=",".join(MACHINES))
    parser.add_argument(
        "--workloads", default="all", help="comma-separated workloads, or all"
    )
    parser.add_argument("--dtypes", default=",".join((*DTYPES, CONVENTIONAL)))
    parser.add_argument("--horizon", type=int, default=40, help="problems per variant")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trace-limit", type=int, default=200, help="rounds traced for problem 0"
    )
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--bench-sizes", default="256,1024,4096")
    parser.add_argument("--bench-degree", type=float, default=4.0)
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--threads", type=int, default=2, help="torch threads")
    parser.add_argument("--out", default="runs", help="parent directory of the run")
    args = parser.parse_args(argv)
    for name in args.machines.split(","):
        if name not in MACHINES:
            parser.error(f"--machines: unknown {name!r}; choose from {list(MACHINES)}")
    for name in args.dtypes.split(","):
        if name not in (*DTYPES, CONVENTIONAL):
            parser.error(f"--dtypes: unknown {name!r}")
    if args.workloads != "all":
        known = {w for _, ws in graph_workloads.GENERATORS.values() for w in ws}
        for name in args.workloads.split(","):
            if name not in known:
                parser.error(f"--workloads: unknown {name!r}")
    if args.horizon < 1 or args.jobs < 1:
        parser.error("--horizon and --jobs must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    run_dir, source, root = lockstep.start(args.out, "e05")
    selected = None if args.workloads == "all" else set(args.workloads.split(","))
    variants = [
        Variant(machine, workload, dtype, args.horizon, args.seed, args.trace_limit)
        for machine in args.machines.split(",")
        for workload in graph_workloads.GENERATORS[machine][1]
        if selected is None or workload in selected
        for dtype in args.dtypes.split(",")
    ]
    header = lockstep.run_header(
        experiment=EXPERIMENT,
        module=MODULE,
        argv=argv,
        run_dir=run_dir,
        source=source,
        root=root,
        config=vars(args),
        device="cpu",
    )
    lockstep.records.write_json(
        run_dir / "run.json",
        header
        | {
            "seeds": {
                "seed": args.seed,
                "derivation": "random.Random(f'{seed}:{machine}:{workload}'); "
                "benchmark random.Random(f'{seed}:bench:{n}')",
            },
            "machines": {
                "reach/v1": "reflexive transitive closure, one edge per round",
                "minplus/v1": {
                    "max_weight": graph_workloads.MAX_WEIGHT,
                    "max_distance": graph_workloads.MAX_DISTANCE,
                    "infinity": "None in canonical state, +inf in tensors",
                },
                "chain-datalog/v1": {
                    "program": [
                        [r.head, [[a.relation, a.inverted] for a in r.body]]
                        for r in graph_workloads.ANCESTRY
                    ]
                },
            },
            "backends": {
                "dense-tensor": "backends.semiring: Boolean products clamped to 1; "
                "min-plus by broadcast and amin; eager PyTorch",
                "conventional": "backends.graph_native: BFS, Dijkstra, semi-naive",
                "dtypes": args.dtypes.split(","),
            },
            "variants": [asdict(v) | {"name": v.name} for v in variants],
        },
    )
    results, error = [], None
    try:
        results = lockstep.run_all(run_variant, variants, args.jobs)
    except Exception:
        error = traceback.format_exc()
    bench: list[dict[str, Any]] = []
    if error is None and not args.no_benchmark:
        try:
            bench = [
                bench_size(int(n), args.bench_degree, args.seed)
                for n in args.bench_sizes.split(",")
            ]
        except Exception:
            error = traceback.format_exc()
    code = lockstep.finish(
        run_dir,
        experiment=EXPERIMENT,
        acceptance=ACCEPTANCE,
        not_measured=NOT_MEASURED,
        variant_count=len(variants),
        results=results,
        error=error,
    )
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["benchmark"] = {
        "description": "single-source reachability and shortest paths from vertex 0 "
        "on random sparse digraphs; median of 3 timings; float32 tensors; "
        f"{args.threads} torch threads",
        "degree": args.bench_degree,
        "rows": bench,
    }
    if bench and not all(row["results_match"] for row in bench) and code == 0:
        metrics["status"], code = "failed", 1
    lockstep.records.write_json(metrics_path, metrics)
    return code


if __name__ == "__main__":
    sys.exit(main())
