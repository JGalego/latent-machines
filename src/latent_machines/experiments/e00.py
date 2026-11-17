"""E00: memory and transition conformance.

Steps the ``memory/v1`` oracle and the eager delta-rule backend in lockstep
over each E00 workload and number format, checking every transition: equal
events and work, equal decoded state, and the backend's residue before
decoding. At ``snapshot_at``, each variant snapshots its decoded state,
restores it through JSON, and continues from the restored tensor, which must
equal the live one exactly.

Writes a run record under ``runs/`` following "Run records and results" in
EXPERIMENTS.md. The default run is the named 10^3-action sweep:

    uv run --locked python -m latent_machines.experiments.e00
    uv run --locked python -m latent_machines.experiments.e00 --horizon 1000000 --jobs 8

Exit status: 0 when every variant conforms, 1 on any mismatch or snapshot
disagreement, 2 if the run itself failed.
"""

import argparse
import hashlib
import json
import multiprocessing
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from latent_machines.backends.delta import DeltaMemory
from latent_machines.experiments import records
from latent_machines.machines.memory import (
    VERSION,
    Action,
    Event,
    Idle,
    Memory,
    Read,
    Write,
)
from latent_machines.tasks.memory_workloads import WORKLOADS, actions

EXPERIMENT = "E00"
BACKEND = "delta-eager"
DTYPES = ("float32", "float64", "bfloat16", "float16")
ACCEPTANCE = (
    "zero event, work, and decoded-state mismatches at every step, and a "
    "snapshot restored through JSON equal to the live tensor"
)
NOT_MEASURED = [
    "backend latency: end_to_end_seconds includes oracle stepping and checks",
    "latency versus capacity",
    "memory traffic",
    "index bytes and retained history: the backend keeps no history",
]


@dataclass(frozen=True)
class Variant:
    workload: str
    dtype: str
    capacity: int
    word_bits: int
    horizon: int
    snapshot_at: int
    seed: int
    trace_limit: int
    device: str

    @property
    def name(self) -> str:
        return f"{BACKEND}/{self.dtype}/{self.workload}"


def action_json(action: Action) -> dict[str, Any]:
    match action:
        case Read(address):
            return {"op": "read", "address": address}
        case Write(address, value):
            return {"op": "write", "address": address, "value": value}
        case Idle():
            return {"op": "idle"}
    return {"op": "halt"}


def event_json(event: Event) -> dict[str, Any]:
    data: dict[str, Any] = {"kind": event.kind.value}
    if event.value is not None:
        data["value"] = event.value
    if event.reason is not None:
        data["reason"] = event.reason.value
    return data


def reproduce_command(variant: Variant, horizon: int) -> list[str]:
    return [
        "python", "-m", "latent_machines.experiments.e00",
        "--workloads", variant.workload, "--dtypes", variant.dtype,
        "--capacity", str(variant.capacity), "--word-bits", str(variant.word_bits),
        "--horizon", str(horizon), "--snapshot-at", str(variant.snapshot_at),
        "--seed", str(variant.seed),
    ]  # fmt: skip


def run_variant(variant: Variant) -> dict[str, Any]:
    """Run one workload and number format in lockstep; return its evidence."""
    torch.set_num_threads(1)
    machine = Memory(variant.capacity, variant.word_bits)
    backend = DeltaMemory(machine, getattr(torch, variant.dtype), variant.device)
    rng = random.Random(f"{variant.seed}:{variant.workload}")

    oracle = machine.initial_state()
    state = backend.initial_state()
    digest = hashlib.sha256()
    trace: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    snapshot: dict[str, Any] | None = None
    snapshot_resume = "not reached"
    max_residue = 0.0
    steps = work = 0

    started = time.perf_counter()
    for index, action in enumerate(
        actions(variant.workload, machine, variant.horizon, rng)
    ):
        digest.update(json.dumps(action_json(action)).encode() + b"\n")
        if index == variant.snapshot_at:
            snapshot = json.loads(json.dumps(machine.snapshot(backend.decode(state))))
            restored = backend.encode(machine.restore(snapshot))
            same = torch.equal(restored.matrix, state.matrix)
            agrees = same and restored.halted == state.halted
            snapshot_resume = "agrees" if agrees else "differs"
            state = restored

        expected = machine.step(oracle, action)
        actual = backend.step(state, action)
        residue = backend.residue(actual.next_state)
        try:
            decoded = backend.decode(actual.next_state)
            decode_error = None
        except ValueError as error:
            decoded, decode_error = None, str(error)
        max_residue = max(max_residue, residue)
        steps += 1
        work += actual.work
        ok = (
            actual.event == expected.event
            and actual.work == expected.work
            and decoded == expected.next_state
        )

        if index < variant.trace_limit or not ok:
            trace.append(
                {
                    "variant": variant.name,
                    "index": index,
                    "action": action_json(action),
                    "event": event_json(actual.event),
                    "work": actual.work,
                    "residue": residue,
                }
            )
        if not ok:
            expected_words = expected.next_state.words
            failure = {
                "variant": variant.name,
                "action_index": index,
                "action": action_json(action),
                "expected": {
                    "event": event_json(expected.event),
                    "work": expected.work,
                    "state": machine.snapshot(expected.next_state),
                },
                "actual": {
                    "event": event_json(actual.event),
                    "work": actual.work,
                    "decoded_state": (
                        machine.snapshot(decoded) if decoded is not None else None
                    ),
                    "decode_error": decode_error,
                    "residue": residue,
                },
                "differing_slots": (
                    [
                        address
                        for address, word in enumerate(decoded.words)
                        if word != expected_words[address]
                    ]
                    if decoded is not None
                    else None
                ),
                "dtype": variant.dtype,
                "reproduce": reproduce_command(variant, index + 1),
            }
            break
        oracle, state = expected.next_state, actual.next_state
    seconds = time.perf_counter() - started

    if failure is not None:
        classification = "mismatch"
    elif max_residue == 0.0:
        classification = "exact"
    else:
        classification = "conforms with residue"
    matrix = state.matrix
    return {
        "metrics": {
            "variant": variant.name,
            "backend": BACKEND,
            "dtype": variant.dtype,
            "workload": variant.workload,
            "horizon": variant.horizon,
            "actions_checked": steps,
            "mismatches": 0 if failure is None else 1,
            "first_mismatch": None if failure is None else failure["action_index"],
            "max_residue": max_residue,
            "classification": classification,
            "snapshot_resume": snapshot_resume,
            "ticks_per_action": work / steps if steps else None,
            "state_tensor_bytes": matrix.numel() * matrix.element_size(),
            "logical_state_bits": variant.capacity * variant.word_bits,
            "end_to_end_seconds": round(seconds, 3),
            "action_sha256": digest.hexdigest(),
        },
        "trace": trace,
        "failure": failure,
        "snapshot": snapshot,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m latent_machines.experiments.e00",
        description="E00 memory conformance of the eager delta-rule backend.",
    )
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--word-bits", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1000, help="actions per variant")
    parser.add_argument(
        "--snapshot-at", type=int, help="action index to snapshot (default: horizon/2)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--dtypes", default=",".join(DTYPES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--trace-limit", type=int, default=1000, help="transitions traced per variant"
    )
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--out", default="runs", help="parent directory of the run")
    args = parser.parse_args(argv)
    for name in args.workloads.split(","):
        if name not in WORKLOADS:
            parser.error(f"unknown workload {name!r}; choose from {sorted(WORKLOADS)}")
    for name in args.dtypes.split(","):
        if name not in DTYPES:
            parser.error(f"unknown dtype {name!r}; choose from {list(DTYPES)}")
    if args.horizon < 1 or args.jobs < 1:
        parser.error("--horizon and --jobs must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    root = records.repo_root(Path(__file__).resolve().parent)
    source, patch = records.source(root)
    run_dir = records.new_run_dir(Path(args.out), "e00", source.get("revision"))
    if patch:
        (run_dir / "source.patch").write_bytes(patch)

    variants = [
        Variant(
            workload=workload,
            dtype=dtype,
            capacity=args.capacity,
            word_bits=args.word_bits,
            horizon=args.horizon,
            snapshot_at=(
                args.horizon // 2 if args.snapshot_at is None else args.snapshot_at
            ),
            seed=args.seed,
            trace_limit=args.trace_limit,
            device=args.device,
        )
        for dtype in args.dtypes.split(",")
        for workload in args.workloads.split(",")
    ]
    lock = root / "uv.lock" if root else None
    records.write_json(
        run_dir / "run.json",
        {
            "schema_version": records.SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "run_id": run_dir.name,
            "command": ["python", "-m", "latent_machines.experiments.e00", *argv],
            "started_at": records.now(),
            "source": source,
            "config": {**vars(args), "snapshot_at": variants[0].snapshot_at},
            "seeds": {
                "seed": args.seed,
                "derivation": "random.Random(f'{seed}:{workload}') per workload",
            },
            "machine": {
                "version": VERSION,
                "capacity": args.capacity,
                "word_bits": args.word_bits,
            },
            "backend": {
                "name": BACKEND,
                "module": "latent_machines.backends.delta",
                "kernel": "eager PyTorch, one matmul and two outer products per step",
                "key_normalization": "x / max(||x||, 1e-6)",
                "gates": "alpha = 1; beta = 1 on writes, 0 otherwise",
                "output_decoding": "threshold each bit at 0.5",
                "dtypes": args.dtypes.split(","),
            },
            "environment": {
                **records.environment(),
                "torch": torch.__version__,
                "torch_threads_per_variant": 1,
                "deterministic": torch.are_deterministic_algorithms_enabled(),
                "device": args.device,
                "uv_lock_sha256": records.file_sha256(lock) if lock else None,
            },
            "variants": [
                asdict(variant) | {"name": variant.name} for variant in variants
            ],
        },
    )

    results: list[dict[str, Any]] = []
    error = None
    try:
        if args.jobs > 1:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(args.jobs, mp_context=context) as pool:
                results = list(pool.map(run_variant, variants))
        else:
            results = [run_variant(variant) for variant in variants]
        _write_evidence(run_dir, results)
    except Exception:
        error = traceback.format_exc()

    metrics = [result["metrics"] for result in results]
    if error is not None:
        status = "error"
    elif all(
        m["mismatches"] == 0 and m["snapshot_resume"] != "differs" for m in metrics
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
                "max_residue": max((m["max_residue"] for m in metrics), default=None),
            },
            "variants": metrics,
            "not_measured": NOT_MEASURED,
            "error": error,
        },
    )
    _print_summary(metrics, status, run_dir, error)
    return {"passed": 0, "failed": 1}.get(status, 2)


def _write_evidence(run_dir: Path, results: list[dict[str, Any]]) -> None:
    with (run_dir / "trace.jsonl").open("w") as trace:
        for result in results:
            for line in result["trace"]:
                trace.write(json.dumps(line) + "\n")
    failures = [result["failure"] for result in results if result["failure"]]
    if failures:
        records.write_json(run_dir / "failure.json", failures)
    snapshots = run_dir / "snapshots"
    for result in results:
        if result["snapshot"] is not None:
            snapshots.mkdir(exist_ok=True)
            m = result["metrics"]
            path = snapshots / f"{m['dtype']}-{m['workload']}.json"
            records.write_json(path, result["snapshot"])


def _print_summary(
    metrics: list[dict[str, Any]], status: str, run_dir: Path, error: str | None
) -> None:
    for m in metrics:
        print(
            f"{m['variant']:<34} {m['actions_checked']:>9} actions  "
            f"{m['mismatches']} mismatches  residue {m['max_residue']:.3g}  "
            f"snapshot {m['snapshot_resume']:<11} {m['end_to_end_seconds']:>8.2f}s"
        )
    if error:
        print(error, file=sys.stderr)
    print(f"status: {status}\nrecord: {run_dir}")


if __name__ == "__main__":
    sys.exit(main())
