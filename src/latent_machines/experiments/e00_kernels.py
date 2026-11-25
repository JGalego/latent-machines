"""E00 follow-up: chunked and fused delta-rule kernels.

The first E00 run stepped a recurrent PyTorch update once per action at 16
slots. This follow-up runs the same workloads at larger capacities through
kernels that process many actions at once, and counts every disagreement with
the integer oracle (see "Amendment 2026-11-25: chunked and fused kernels,
pre-registered" under E00 in EXPERIMENTS.md):

- ``recurrent``: the first run's backend, as a reference (``delta.py``);
- ``chunked-4``, ``chunked-16``, ``chunked-64``: the WY chunked form in
  PyTorch (``delta_chunked.py``), on CPU;
- ``fla-chunk``, ``fla-recurrent``: flash-linear-attention's fused kernels
  (``fla_kernels.py``), on a GPU.

A chunked or fused kernel exposes its state only at chunk or segment ends, so
those variants compare every action's event and the decoded state at each
boundary. The recurrent variant compares both after every action. Mismatches
are counted, not stopped at: a variant reports the number of actions whose
event differed and the number of boundaries whose state differed.

    uv run --locked python -m latent_machines.experiments.e00_kernels --jobs 4
    uv run --locked python -m latent_machines.experiments.e00_kernels \\
        --kernels fla-chunk,fla-recurrent --dtypes float32,bfloat16 --device cuda

Exit status: 0 when the run completes, whatever it found; 2 if it failed.
Mismatches are results here, not failures. A kernel that cannot run is
recorded as an error in that variant.
"""

import argparse
import hashlib
import json
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any

import torch

from latent_machines.backends import delta_chunked as chunked
from latent_machines.backends import fla_kernels
from latent_machines.experiments import e00, lockstep, records
from latent_machines.machines.memory import VERSION, Memory
from latent_machines.tasks.memory_workloads import WORKLOADS, actions

EXPERIMENT = "E00"
MODULE = "latent_machines.experiments.e00_kernels"
KERNELS = (
    "recurrent",
    "chunked-4",
    "chunked-16",
    "chunked-64",
    *fla_kernels.KERNELS,
)
DEFAULT_KERNELS = ("recurrent", "chunked-4", "chunked-16", "chunked-64")
DTYPES = ("float64", "float32", "bfloat16", "float16")
DEFAULT_DTYPES = ("float64", "float32", "bfloat16")
CAPACITIES = (64, 512, 4096)
PREDICTION_HORIZON = 100_000
ACCEPTANCE = (
    "count event mismatches per action and decoded-state mismatches at each "
    "chunk or segment end against the integer oracle; no pass or fail"
)
NOT_MEASURED = [
    "kernel latency: end_to_end_seconds includes oracle stepping and checks",
    "state inside a chunk or segment: a chunked or fused kernel does not expose it",
    "gates away from their endpoints and keys that are not one-hot",
    "snapshot and resume (not applicable to chunk-boundary checks)",
]


@dataclass(frozen=True)
class Variant:
    kernel: str
    dtype: str
    capacity: int
    word_bits: int
    workload: str
    horizon: int
    seed: int
    device: str
    normalize_in_kernel: bool = False
    segment: int = fla_kernels.SEGMENT

    @property
    def name(self) -> str:
        norm = "/kernel-norm" if self.normalize_in_kernel else ""
        return f"{self.kernel}{norm}/{self.dtype}/c{self.capacity}/{self.workload}"


def _chunk_size(kernel: str) -> int:
    return int(kernel.rsplit("-", 1)[1])


def run_variant(variant: Variant) -> dict[str, Any]:
    """Run one kernel, number format, capacity, and workload; return its result."""
    torch.set_num_threads(1)
    try:
        if variant.kernel == "recurrent":
            result = _run_recurrent(variant)
        else:
            result = _run_chunked(variant)
    except Exception:
        result = _error_result(variant, traceback.format_exc())
    m = result["metrics"]
    note = m["error"].strip().splitlines()[-1][:160] if "error" in m else ""
    print(
        f"{variant.name:<52} {m['classification']:<22} "
        f"{m['mismatches']} mismatches, {m['end_to_end_seconds']}s {note}",
        file=sys.stderr,
        flush=True,
    )
    return result


def _base_metrics(variant: Variant) -> dict[str, Any]:
    return {
        "variant": variant.name,
        "kernel": variant.kernel,
        "dtype": variant.dtype,
        "capacity": variant.capacity,
        "word_bits": variant.word_bits,
        "workload": variant.workload,
        "horizon": variant.horizon,
        "normalize_in_kernel": variant.normalize_in_kernel,
        "logical_state_bits": variant.capacity * variant.word_bits,
    }


def _error_result(variant: Variant, error: str) -> dict[str, Any]:
    metrics = _base_metrics(variant) | {
        "actions_checked": 0,
        "mismatches": 0,
        "event_mismatches": None,
        "state_mismatches": None,
        "first_mismatch": None,
        "max_residue": 0.0,
        "classification": "error",
        "snapshot_resume": "not applicable",
        "ticks_per_action": None,
        "state_tensor_bytes": 0,
        "end_to_end_seconds": 0.0,
        "error": error,
    }
    return {"metrics": metrics, "trace": [], "failure": None, "snapshot": None}


def _run_recurrent(variant: Variant) -> dict[str, Any]:
    result = e00.run_variant(
        e00.Variant(
            workload=variant.workload,
            dtype=variant.dtype,
            capacity=variant.capacity,
            word_bits=variant.word_bits,
            horizon=variant.horizon,
            snapshot_at=variant.horizon // 2,
            seed=variant.seed,
            trace_limit=0,
            device=variant.device,
        )
    )
    result["metrics"] = _base_metrics(variant) | {
        k: v
        for k, v in result["metrics"].items()
        if k not in ("variant", "backend", "workload", "dtype", "horizon")
    }
    result["metrics"] |= {
        "event_mismatches": None,
        "state_mismatches": None,
        "granularity": "every action",
    }
    result["trace"] = []
    return result


def _backend_chunks(variant: Variant, machine: Memory, lowered: chunked.Lowered):
    dtype = getattr(torch, variant.dtype)
    if variant.kernel.startswith("chunked-"):
        backend = chunked.ChunkedDeltaMemory(
            machine, _chunk_size(variant.kernel), dtype, variant.device
        )
        return backend.run(lowered)
    return fla_kernels.run_fla(
        machine,
        lowered,
        variant.kernel,
        dtype,
        normalize_in_kernel=variant.normalize_in_kernel,
        segment=variant.segment,
        device=variant.device,
    )


def _run_chunked(variant: Variant) -> dict[str, Any]:
    machine = Memory(variant.capacity, variant.word_bits)
    rng = random.Random(f"{variant.seed}:{variant.workload}")
    sequence = list(actions(variant.workload, machine, variant.horizon, rng))
    digest = hashlib.sha256()
    for action in sequence:
        digest.update(json.dumps(lockstep.action_json(action)).encode() + b"\n")
    lowered = chunked.lower_sequence(machine, sequence)

    oracle = machine.initial_state()
    event_mismatches = state_mismatches = 0
    first: int | None = None
    failure: dict[str, Any] | None = None
    max_residue = 0.0
    state_bytes = 0
    started = time.perf_counter()
    for chunk in _backend_chunks(variant, machine, lowered):
        events = chunked.chunk_events(lowered, chunk)
        for event, i in zip(events, range(chunk.start, chunk.stop), strict=True):
            expected = machine.step(oracle, sequence[i])
            oracle = expected.next_state
            if event != expected.event:
                event_mismatches += 1
                if first is None:
                    first = i
                    failure = _failure(variant, i, sequence[i], expected, event)
        max_residue = max(
            max_residue, chunked.residue(chunk.state.float(), chunk.outputs.float())
        )
        state_bytes = chunk.state.numel() * chunk.state.element_size()
        try:
            decoded = chunked.word_tensor(chunk.state.T).cpu()
            same = torch.equal(decoded, torch.tensor(oracle.words))
        except ValueError:
            same = False
        if not same:
            state_mismatches += 1
            if first is None:
                first = chunk.stop - 1
                failure = _failure(variant, first, sequence[first], expected, None)
    seconds = time.perf_counter() - started

    mismatches = event_mismatches + state_mismatches
    if mismatches:
        classification = "mismatch"
    elif max_residue == 0.0:
        classification = "exact"
    else:
        classification = "conforms with residue"
    metrics = _base_metrics(variant) | {
        "actions_checked": len(sequence),
        "mismatches": mismatches,
        "event_mismatches": event_mismatches,
        "state_mismatches": state_mismatches,
        "first_mismatch": first,
        "max_residue": max_residue,
        "classification": classification,
        "snapshot_resume": "not applicable",
        "ticks_per_action": 1.0,
        "state_tensor_bytes": state_bytes,
        "granularity": (
            f"every action's event; state every {_boundary(variant)} actions"
        ),
        "end_to_end_seconds": round(seconds, 3),
        "action_sha256": digest.hexdigest(),
    }
    return {"metrics": metrics, "trace": [], "failure": failure, "snapshot": None}


def _boundary(variant: Variant) -> int:
    if variant.kernel.startswith("chunked-"):
        return _chunk_size(variant.kernel)
    return variant.segment


def _failure(variant, index, action, expected, actual_event) -> dict[str, Any]:
    return {
        "variant": variant.name,
        "action_index": index,
        "action": lockstep.action_json(action),
        "expected": {"event": lockstep.event_json(expected.event)},
        "actual": {
            "event": lockstep.event_json(actual_event) if actual_event else None
        },
    }


# ---------- predictions ----------


def evaluate_predictions(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """The three pre-registered predictions from a run's variant metrics."""

    def select(**where: Any) -> list[dict[str, Any]]:
        return [
            m for m in metrics if all(m[key] in value for key, value in where.items())
        ]

    def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "variants": len(rows),
            "completed": sum(m["classification"] != "error" for m in rows),
            "with_mismatches": sum(m["mismatches"] > 0 for m in rows),
            "mismatches": sum(m["mismatches"] for m in rows),
            "errors": sum(m["classification"] == "error" for m in rows),
        }

    chunk_kernels = ("chunked-4", "chunked-16", "chunked-64")
    exact = select(kernel=chunk_kernels, dtype=("float64", "float32"))
    low = select(kernel=chunk_kernels, dtype=("bfloat16",))
    long_low = [m for m in low if m["horizon"] >= PREDICTION_HORIZON]
    fused = select(kernel=fla_kernels.KERNELS, dtype=("bfloat16",))
    one, two, three = counts(exact), counts(long_low), counts(fused)
    return {
        "1_chunked_exact_in_float64_and_float32": {
            **one,
            "met": bool(one["completed"])
            and one["mismatches"] == 0
            and one["errors"] == 0,
        },
        "2_chunked_bfloat16_has_a_mismatch": {
            **two,
            "met": two["with_mismatches"] > 0,
        },
        "3_fused_bfloat16_has_a_mismatch": {
            **three,
            "met": three["with_mismatches"] > 0 if three["completed"] else None,
            "note": None if three["completed"] else "no fused variant completed",
        },
    }


# ---------- run ----------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="E00 follow-up: chunked and fused delta-rule kernels.",
    )
    parser.add_argument("--kernels", default=",".join(DEFAULT_KERNELS))
    parser.add_argument("--dtypes", default=",".join(DEFAULT_DTYPES))
    parser.add_argument("--capacities", default=",".join(map(str, CAPACITIES)))
    parser.add_argument("--word-bits", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=PREDICTION_HORIZON)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument(
        "--kernel-norm",
        choices=("outside", "inside", "both"),
        default="outside",
        help="fla kernels: where keys and queries are L2-normalized",
    )
    parser.add_argument("--segment", type=int, default=fla_kernels.SEGMENT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--out", default="runs", help="parent directory of the run")
    args = parser.parse_args(argv)
    for option, allowed in (
        (args.kernels, KERNELS),
        (args.dtypes, DTYPES),
        (args.workloads, tuple(WORKLOADS)),
    ):
        for name in option.split(","):
            if name not in allowed:
                parser.error(f"unknown {name!r}; choose from {list(allowed)}")
    if args.horizon < 1 or args.jobs < 1 or args.segment < 1:
        parser.error("--horizon, --jobs, and --segment must be positive")
    return args


def make_variants(args: argparse.Namespace) -> list[Variant]:
    variants = []
    for kernel in args.kernels.split(","):
        fused = kernel in fla_kernels.KERNELS
        norms = {"outside": [False], "inside": [True], "both": [False, True]}
        for inside in norms[args.kernel_norm] if fused else [False]:
            for dtype in args.dtypes.split(","):
                for capacity in map(int, args.capacities.split(",")):
                    for workload in args.workloads.split(","):
                        variants.append(
                            Variant(
                                kernel=kernel,
                                dtype=dtype,
                                capacity=capacity,
                                word_bits=args.word_bits,
                                workload=workload,
                                horizon=args.horizon,
                                seed=args.seed,
                                device=args.device,
                                normalize_in_kernel=inside,
                                segment=args.segment,
                            )
                        )
    return variants


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    run_dir, source, root = lockstep.start(args.out, "e00-kernels")
    variants = make_variants(args)
    header = lockstep.run_header(
        experiment=EXPERIMENT,
        module=MODULE,
        argv=argv,
        run_dir=run_dir,
        source=source,
        root=root,
        config=vars(args),
        device=args.device,
    )
    header["environment"]["fla"] = fla_kernels.package_versions()
    records.write_json(
        run_dir / "run.json",
        header
        | {
            "seeds": {
                "seed": args.seed,
                "derivation": "random.Random(f'{seed}:{workload}') per workload",
            },
            "machine": {"version": VERSION, "word_bits": args.word_bits},
            "variants": [asdict(v) | {"name": v.name} for v in variants],
        },
    )

    results, error = [], None
    try:
        results = lockstep.run_all(run_variant, variants, args.jobs)
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
    return _annotate(run_dir, results, error, code)


def _annotate(run_dir, results, error, code: int) -> int:
    """Replace the pass/fail status, which does not apply here, with a summary."""
    if error is not None:
        return 2
    path = run_dir / "metrics.json"
    metrics = json.loads(path.read_text())
    rows = [r["metrics"] for r in results]
    metrics["status"] = (
        "completed with errors"
        if any(m["classification"] == "error" for m in rows)
        else "completed"
    )
    metrics["predictions"] = evaluate_predictions(rows)
    records.write_json(path, metrics)
    print(json.dumps(metrics["predictions"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
