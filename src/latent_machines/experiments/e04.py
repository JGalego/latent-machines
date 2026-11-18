"""E04: dataflow, specialization, and conditional units.

Evaluates seeded ``dataflow/v1`` programs (expression DAGs, a bounded dot
product, gather/map/reduce/scatter) on every backend in
``backends.dataflow`` and compares each evaluation with the oracle: the
event (outputs, or the faulting node), the work, and the canonical values
of every evaluated node. A variant is one backend, number format, and
workload. Unlike ``lockstep.check``, a variant does not stop at its first
mismatch; it counts mismatching evaluations and keeps the first as
evidence.

Snapshot check: on the first evaluation that completes without a fault,
stepped backends stop at the middle level (or instruction), round-trip the
canonical state through JSON, continue from the restored state, and must
equal the live state and the oracle's result.

Timing is separate from conformance: compilation, the first call after
compiling (cold), steady-state latency for one input vector, and
throughput for a batch, on the first programs of each workload.

    uv run --locked python -m latent_machines.experiments.e04

Exit status: 0 when every variant in a format declared exact passes, 1
otherwise, 2 if the run itself failed. bfloat16 and float16 variants are
precision probes: they are checked and reported the same way, but cannot
represent every 16-bit word, so they are not part of the acceptance.
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

from latent_machines.backends.dataflow import (
    CompiledGraph,
    Outcome,
    ScalarDelta,
    TensorState,
    evaluation_order,
    python_int,
)
from latent_machines.experiments import lockstep, records
from latent_machines.machines.dataflow import Dataflow, Program
from latent_machines.tasks import dataflow_workloads as workloads

EXPERIMENT = "E04"
MODULE = "latent_machines.experiments.e04"
WIDTH = workloads.WIDTH
# backend name -> (lowering, number formats); python-int and scalar-delta are special
BACKENDS: dict[str, tuple[str | None, tuple[str, ...]]] = {
    "python-int": (None, ("int",)),
    "torch-int64": ("grouped", ("int64",)),
    "scalar-delta": (None, ("float32",)),
    "compiled-graph": ("gather", ("float32", "float64", "bfloat16", "float16")),
    "specialized-weights": ("weights", ("float32", "float64", "bfloat16", "float16")),
    "expert-dispatch": ("dispatch", ("float32",)),
}
PROBES = ("bfloat16", "float16")
ACCEPTANCE = (
    "for every variant in int, int64, float32, or float64: zero mismatches in "
    "event (outputs or faulting node), work, and evaluated node values on every "
    "evaluation, and snapshot resume equal to the live state; bfloat16 and "
    "float16 variants are precision probes reported but not accepted"
)
NOT_MEASURED = [
    "a fused dot product (one Linear layer without per-product checks): it "
    "would skip the mul nodes' precise overflow faults unless input bounds "
    "are declared",
    "learned or input-dependent routing: expert dispatch is static per program",
    "GPU or other devices; timings are on a shared, busy CPU with 2 threads",
    "peak memory per variant",
]


@dataclass(frozen=True)
class Variant:
    backend: str
    dtype: str
    workload: str
    programs: int
    inputs: int
    seed: int
    capacity: int
    timing_programs: int
    timing_batch: int
    timing_repeats: int
    trace_limit: int

    @property
    def name(self) -> str:
        return f"{self.backend}/{self.dtype}/{self.workload}"


def workload_data(
    workload: str, programs: int, inputs: int, seed: int
) -> list[tuple[Program, list[list[int]]]]:
    rng = random.Random(f"{seed}:{workload}")
    data = []
    for _ in range(programs):
        program = workloads.program(workload, rng)
        data.append(
            (program, [workloads.inputs(workload, program, rng) for _ in range(inputs)])
        )
    return data


def outcome_json(o: Outcome) -> dict[str, Any]:
    event: dict[str, Any] = {"kind": o.event.kind.value}
    if o.event.outputs is not None:
        event["outputs"] = list(o.event.outputs)
    if o.event.node is not None:
        event |= {"node": o.event.node, "reason": o.event.reason.value}  # type: ignore[union-attr]
    return {"event": event, "work": o.work, "residue": o.residue}


def _same(actual: Outcome, expected: Outcome) -> bool:
    return (
        actual.event == expected.event
        and actual.work == expected.work
        and actual.values == expected.values
    )


def _tensor_states_equal(a: TensorState, b: TensorState) -> bool:
    return (
        a.level == b.level
        and torch.equal(a.values, b.values)
        and torch.equal(a.fault, b.fault)
        and torch.equal(a.work, b.work)
    )


def compiled(variant: Variant, program: Program) -> CompiledGraph:
    lowering = BACKENDS[variant.backend][0]
    assert lowering is not None
    return CompiledGraph(
        program, WIDTH, getattr(torch, variant.dtype), lowering, variant.capacity
    )


def evaluate_all(
    variant: Variant, program: Program, inputs: list[list[int]]
) -> list[Outcome | str]:
    """Outcomes per input vector, or a decode error message."""
    if variant.backend == "python-int":
        return [python_int(program, x, WIDTH) for x in inputs]
    if variant.backend == "scalar-delta":
        return [ScalarDelta(WIDTH).run(program, x)[0] for x in inputs]
    graph = compiled(variant, program)
    state = graph.run(inputs)
    results: list[Outcome | str] = []
    for column in range(len(inputs)):
        try:
            results.append(graph.outcome(state, column))
        except ValueError as error:
            results.append(str(error))
    return results


def snapshot_check(
    variant: Variant, machine: Dataflow, program: Program, x: list[int]
) -> str:
    """Round-trip the state at the middle safepoint.

    The restored state must equal the live one, and resuming from it must
    give the same outcome as an uninterrupted run of the same backend.
    Agreement with the oracle is checked separately.
    """
    if variant.backend == "python-int":
        return "not applicable"
    if variant.backend == "scalar-delta":
        middle = len(evaluation_order(program)) // 2
        live, _ = ScalarDelta(WIDTH).run(program, x)
        outcome, agreement = ScalarDelta(WIDTH).run(program, x, snapshot_at=middle)
        return agreement if _same(outcome, live) else "differs"
    graph = compiled(variant, program)
    tensor_x = torch.tensor([x], dtype=graph.dtype).T
    state = graph.initial_state(1)
    middle = max(1, graph.num_levels // 2)
    while state.level < middle:
        state = graph.step(state, tensor_x)
    canonical, _ = graph.decode(state, 0)
    if canonical.finished:
        return "not reached"
    saved = json.loads(json.dumps(machine.snapshot(canonical)))
    restored = graph.encode(machine.restore(saved, program))
    if not _tensor_states_equal(restored, state):
        return "differs"
    while restored.level < graph.num_levels:
        restored = graph.step(restored, tensor_x)
    live = graph.outcome(graph.run([x]), 0)
    return "agrees" if _same(graph.outcome(restored, 0), live) else "differs"


def _oracle(machine: Dataflow, program: Program, x: list[int]) -> Outcome:
    event, state, work = machine.run(program, x)
    values = state.values
    return Outcome(event, values, work)


def timing(
    variant: Variant, data: list[tuple[Program, list[list[int]]]], seed: int
) -> dict[str, Any]:
    """Compile, cold, steady batch-1, and batch throughput on the first programs."""
    rng = random.Random(f"{seed}:{variant.workload}:timing")
    rows = []
    for program, inputs in data[: variant.timing_programs]:
        batch = [
            workloads.inputs(variant.workload, program, rng)
            for _ in range(variant.timing_batch)
        ]
        rows.append(_time_program(variant, program, inputs[0], batch))
    compile_s, cold_s, latency_s, batch_s = zip(*rows, strict=True)
    return {
        "timing_programs": len(rows),
        "compile_seconds": statistics.mean(compile_s),
        "cold_seconds": statistics.mean(cold_s),
        "latency_seconds": statistics.mean(latency_s),
        "batch": variant.timing_batch,
        "batch_seconds": statistics.mean(batch_s),
        "throughput_evaluations_per_second": variant.timing_batch
        / statistics.mean(batch_s),
    }


def _time_program(
    variant: Variant, program: Program, x: list[int], batch: list[list[int]]
) -> tuple[float, float, float, float]:
    """Seconds to compile, for the first call, per steady call, and per batch."""
    repeats = variant.timing_repeats
    if variant.backend in ("python-int", "scalar-delta"):
        scalar = variant.backend == "scalar-delta"

        def run_one(vector: list[int]) -> object:
            if scalar:
                return ScalarDelta(WIDTH).run(program, vector)
            return python_int(program, vector, WIDTH)

        def run_many(vectors: list[list[int]]) -> object:
            return [run_one(vector) for vector in vectors]

        compile_s = _time(lambda: evaluation_order(program))
        cold_s = _time(lambda: run_one(x))
        if scalar:
            repeats = max(3, repeats // 10)
        steady = statistics.median(_time(lambda: run_one(x)) for _ in range(repeats))
        # The scalar machine is slow; time a sixteenth of the batch and scale.
        sample = batch[: max(1, len(batch) // (16 if scalar else 1))]
        many = statistics.median(_time(lambda: run_many(sample)) for _ in range(3))
        return compile_s, cold_s, steady, many * len(batch) / len(sample)
    graph = compiled(variant, program)
    cold_s = _time(lambda: graph.run([x]))
    steady = statistics.median(_time(lambda: graph.run([x])) for _ in range(repeats))
    many = statistics.median(_time(lambda: graph.run(batch)) for _ in range(5))
    return graph.compile_seconds, cold_s, steady, many


def _time(call: Any) -> float:
    started = time.perf_counter()
    call()
    return time.perf_counter() - started


def sizes(variant: Variant, data: list[tuple[Program, list[list[int]]]]) -> dict:
    """Mean static sizes over the workload's programs."""
    rows = []
    for program, _ in data:
        nodes = len(program.nodes)
        if variant.backend == "python-int":
            rows.append(
                {"nodes": nodes, "weight_bytes": 0, "state_bytes": None, "units": nodes}
            )
        elif variant.backend == "scalar-delta":
            rows.append(
                {
                    "nodes": nodes,
                    "weight_bytes": 0,
                    "state_bytes": ScalarDelta.state_bytes(program, WIDTH),
                    "units": nodes,
                }
            )
        else:
            graph = compiled(variant, program)
            rows.append(
                {
                    "nodes": nodes,
                    "levels": graph.num_levels,
                    "weight_bytes": graph.weight_bytes(),
                    "state_bytes": graph.state_bytes(1),
                    "units": graph.units_per_evaluation(),
                    "padded_slots": graph.padded_slots(),
                }
            )
    keys = rows[0].keys()
    return {
        f"mean_{key}": (
            None
            if any(r.get(key) is None for r in rows)
            else statistics.mean(r[key] for r in rows)
        )
        for key in keys
    }


def run_variant(variant: Variant) -> dict[str, Any]:
    torch.set_num_threads(2)
    machine = Dataflow(WIDTH)
    data = workload_data(
        variant.workload, variant.programs, variant.inputs, variant.seed
    )
    started = time.perf_counter()
    checked = mismatches = faults = 0
    first: int | None = None
    max_residue = 0.0
    ticks: list[int] = []
    trace: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    snapshot = "not reached"
    for p_index, (program, inputs) in enumerate(data):
        expected = [_oracle(machine, program, x) for x in inputs]
        actual = evaluate_all(variant, program, inputs)
        if snapshot == "not reached":
            ok = [i for i, e in enumerate(expected) if e.event.kind.value == "output"]
            if ok:
                snapshot = snapshot_check(variant, machine, program, inputs[ok[0]])
        for i_index, (want, got) in enumerate(zip(expected, actual, strict=True)):
            index = checked
            checked += 1
            faults += want.event.kind.value == "fault"
            good = isinstance(got, Outcome) and _same(got, want)
            if isinstance(got, Outcome):
                max_residue = max(max_residue, got.residue)
                ticks.append(got.ticks)
            if index < variant.trace_limit:
                trace.append(
                    {
                        "variant": variant.name,
                        "index": index,
                        "program": p_index,
                        "input": i_index,
                        "result": outcome_json(got)
                        if isinstance(got, Outcome)
                        else {"decode_error": got},
                    }
                )
            if not good:
                mismatches += 1
                if first is None:
                    first = index
                    failure = {
                        "variant": variant.name,
                        "evaluation": index,
                        "program_index": p_index,
                        "input_index": i_index,
                        "inputs": inputs[i_index],
                        "expected": outcome_json(want) | {"values": list(want.values)},
                        "actual": (
                            outcome_json(got) | {"values": list(got.values)}
                            if isinstance(got, Outcome)
                            else {"decode_error": got}
                        ),
                        "reproduce": [
                            "python",
                            "-m",
                            MODULE,
                            "--backends",
                            variant.backend,
                            "--dtypes",
                            variant.dtype,
                            "--workloads",
                            variant.workload,
                            "--programs",
                            str(p_index + 1),
                            "--inputs",
                            str(variant.inputs),
                            "--seed",
                            str(variant.seed),
                        ],  # fmt: skip
                    }
    seconds = time.perf_counter() - started
    if mismatches:
        classification = "mismatch"
    elif max_residue == 0.0:
        classification = "exact"
    else:
        classification = "conforms with residue"
    metrics = {
        "variant": variant.name,
        "machine": variant.backend,
        "backend": variant.backend,
        "workload": variant.workload,
        "dtype": variant.dtype,
        "precision_probe": variant.dtype in PROBES,
        "unit": "evaluation: one program on one input vector",
        "actions_checked": checked,
        "mismatches": mismatches,
        "first_mismatch": first,
        "max_residue": max_residue,
        "classification": classification,
        "snapshot_resume": snapshot,
        "oracle_faults": faults,
        "mean_backend_ticks": statistics.mean(ticks) if ticks else None,
        "end_to_end_seconds": round(seconds, 3),
        **sizes(variant, data),
        **timing(variant, data, variant.seed),
    }
    return {"metrics": metrics, "trace": trace, "failure": failure}


def amortization(metrics: list[dict[str, Any]]) -> None:
    """Evaluations needed to repay compilation against conventional baselines."""
    for m in metrics:
        baselines = {
            "python_int_batch1": next(
                (
                    b["latency_seconds"]
                    for b in metrics
                    if b["backend"] == "python-int" and b["workload"] == m["workload"]
                ),
                None,
            ),
            "torch_int64_batched": next(
                (
                    b["batch_seconds"] / b["batch"]
                    for b in metrics
                    if b["backend"] == "torch-int64" and b["workload"] == m["workload"]
                ),
                None,
            ),
        }
        own = {
            "python_int_batch1": m["latency_seconds"],
            "torch_int64_batched": m["batch_seconds"] / m["batch"],
        }
        for key, base in baselines.items():
            saving = None if base is None else base - own[key]
            m[f"amortize_evaluations_vs_{key}"] = (
                None
                if not saving or saving <= 0
                else round(m["compile_seconds"] / saving, 1)
            )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="E04 conformance and cost of dataflow backends.",
    )
    parser.add_argument("--backends", default=",".join(BACKENDS))
    parser.add_argument("--dtypes", default="all", help="comma list or 'all'")
    parser.add_argument("--workloads", default=",".join(workloads.WORKLOADS))
    parser.add_argument("--programs", type=int, default=20, help="per workload")
    parser.add_argument("--inputs", type=int, default=50, help="vectors per program")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=4, help="expert capacity")
    parser.add_argument("--timing-programs", type=int, default=5)
    parser.add_argument("--timing-batch", type=int, default=256)
    parser.add_argument("--timing-repeats", type=int, default=30)
    parser.add_argument("--trace-limit", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--out", default="runs")
    args = parser.parse_args(argv)
    for option, names, allowed in (
        ("--backends", args.backends, tuple(BACKENDS)),
        ("--workloads", args.workloads, workloads.WORKLOADS),
    ):
        for name in names.split(","):
            if name not in allowed:
                parser.error(f"{option}: unknown {name!r}; choose from {list(allowed)}")
    if min(args.programs, args.inputs, args.capacity, args.jobs) < 1:
        parser.error("sizes, capacity, and jobs must be positive")
    if min(args.timing_programs, args.timing_batch, args.timing_repeats) < 1:
        parser.error("timing sizes must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    wanted = None if args.dtypes == "all" else set(args.dtypes.split(","))
    variants = [
        Variant(
            backend=backend,
            dtype=dtype,
            workload=workload,
            programs=args.programs,
            inputs=args.inputs,
            seed=args.seed,
            capacity=args.capacity,
            timing_programs=args.timing_programs,
            timing_batch=args.timing_batch,
            timing_repeats=args.timing_repeats,
            trace_limit=args.trace_limit,
        )
        for workload in args.workloads.split(",")
        for backend in args.backends.split(",")
        for dtype in BACKENDS[backend][1]
        if wanted is None or dtype in wanted
    ]
    if not variants:
        raise SystemExit("no variants selected")
    run_dir, source, root = lockstep.start(args.out, "e04")
    header = lockstep.run_header(
        experiment=EXPERIMENT,
        module=MODULE,
        argv=argv,
        run_dir=run_dir,
        source=source,
        root=root,
        config={**vars(args), "horizon": args.programs * args.inputs, "width": WIDTH},
        device="cpu",
    )
    header["environment"]["torch_threads_per_variant"] = 2
    records.write_json(
        run_dir / "run.json",
        header
        | {
            "seeds": {
                "seed": args.seed,
                "derivation": "random.Random(f'{seed}:{workload}') for programs "
                "and inputs; f'{seed}:{workload}:timing' for timing batches",
            },
            "machine": {"dataflow/v1": {"width": WIDTH}},
            "backends": {
                "python-int": "conventional: unbounded Python ints, then the "
                "first out-of-range node in evaluation order",
                "torch-int64": "optimized conventional: one batched int64 op per "
                "op type per level",
                "scalar-delta": "scalar machine: one node per instruction, "
                "operands in the E00 delta-rule memory, host integer ALU",
                "compiled-graph": "direct compiled graph: gather by index, all "
                "units per node, masked select",
                "specialized-weights": "program-specialized dense weight matrices "
                "for wiring and constants",
                "expert-dispatch": f"static routing to op experts, capacity "
                f"{args.capacity} per round, padded",
            },
            "probes": list(PROBES),
            "variants": [asdict(v) | {"name": v.name} for v in variants],
        },
    )
    results: list[dict[str, Any]] = []
    error = None
    try:
        results = lockstep.run_all(run_variant, variants, args.jobs)
    except Exception:
        error = traceback.format_exc()
    metrics = [r["metrics"] for r in results]
    try:
        amortization(metrics)
        with (run_dir / "trace.jsonl").open("w") as handle:
            for result in results:
                for line in result["trace"]:
                    handle.write(json.dumps(line) + "\n")
        failures = [r["failure"] for r in results if r["failure"]]
        if failures:
            records.write_json(run_dir / "failure.json", failures)
    except Exception:
        error = (error or "") + traceback.format_exc()
    accepted = [m for m in metrics if not m["precision_probe"]]
    if error is not None or len(metrics) != len(variants):
        status = "error"
    elif all(
        m["mismatches"] == 0 and m["snapshot_resume"] != "differs" for m in accepted
    ):
        status = "passed"
    else:
        status = "failed"
    records.write_json(
        run_dir / "metrics.json",
        {
            "schema_version": records.SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": status,
            "finished_at": records.now(),
            "acceptance": ACCEPTANCE,
            "totals": {
                "variants": len(variants),
                "variants_completed": len(metrics),
                "actions_checked": sum(m["actions_checked"] for m in metrics),
                "mismatches": sum(m["mismatches"] for m in metrics),
                "mismatches_outside_probes": sum(m["mismatches"] for m in accepted),
                "max_residue": max((m["max_residue"] for m in metrics), default=None),
            },
            "variants": metrics,
            "not_measured": NOT_MEASURED,
            "error": error,
        },
    )
    for m in metrics:
        print(
            f"{m['variant']:<52} {m['actions_checked']:>6} evals "
            f"{m['mismatches']:>5} mismatches  snapshot {m['snapshot_resume']:<14} "
            f"compile {m['compile_seconds'] * 1e3:8.2f} ms  "
            f"batch-1 {m['latency_seconds'] * 1e6:9.1f} us  "
            f"{m['throughput_evaluations_per_second']:>10.0f} evals/s"
        )
    if error:
        print(error, file=sys.stderr)
    print(f"status: {status}\nrecord: {run_dir}")
    return {"passed": 0, "failed": 1}.get(status, 2)


if __name__ == "__main__":
    sys.exit(main())
