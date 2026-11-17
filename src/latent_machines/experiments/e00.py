"""E00: memory and transition conformance.

Steps the ``memory/v1`` oracle and the eager delta-rule backend in lockstep
(see ``lockstep.check``) over each E00 workload and number format, checking
every transition and a JSON snapshot restore at ``snapshot_at``.

Writes a run record under ``runs/`` following "Run records and results" in
EXPERIMENTS.md. The default run is the named 10^3-action sweep:

    uv run --locked python -m latent_machines.experiments.e00
    uv run --locked python -m latent_machines.experiments.e00 --horizon 1000000 --jobs 8

Exit status: 0 when every variant conforms, 1 on any mismatch or snapshot
disagreement, 2 if the run itself failed.
"""

import argparse
import random
import sys
import traceback
from dataclasses import asdict, dataclass
from typing import Any

import torch

from latent_machines.backends.delta import DeltaMemory
from latent_machines.experiments import lockstep
from latent_machines.machines.memory import VERSION, Memory
from latent_machines.tasks.memory_workloads import WORKLOADS, actions

EXPERIMENT = "E00"
MODULE = "latent_machines.experiments.e00"
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


def reproduce_command(variant: Variant, horizon: int) -> list[str]:
    return [
        "python", "-m", MODULE,
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
    result = lockstep.check(
        machine,
        backend,
        actions(variant.workload, machine, variant.horizon, rng),
        name=variant.name,
        snapshot_at=variant.snapshot_at,
        trace_limit=variant.trace_limit,
        reproduce=lambda horizon: reproduce_command(variant, horizon),
    )
    result["metrics"] |= {
        "backend": BACKEND,
        "dtype": variant.dtype,
        "workload": variant.workload,
        "horizon": variant.horizon,
        "logical_state_bits": variant.capacity * variant.word_bits,
    }
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
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
    run_dir, source, root = lockstep.start(args.out, "e00")
    snapshot_at = args.horizon // 2 if args.snapshot_at is None else args.snapshot_at
    variants = [
        Variant(
            workload=workload,
            dtype=dtype,
            capacity=args.capacity,
            word_bits=args.word_bits,
            horizon=args.horizon,
            snapshot_at=snapshot_at,
            seed=args.seed,
            trace_limit=args.trace_limit,
            device=args.device,
        )
        for dtype in args.dtypes.split(",")
        for workload in args.workloads.split(",")
    ]
    header = lockstep.run_header(
        experiment=EXPERIMENT,
        module=MODULE,
        argv=argv,
        run_dir=run_dir,
        source=source,
        root=root,
        config={**vars(args), "snapshot_at": snapshot_at},
        device=args.device,
    )
    lockstep.records.write_json(
        run_dir / "run.json",
        header
        | {
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
            "variants": [asdict(v) | {"name": v.name} for v in variants],
        },
    )

    results, error = [], None
    try:
        results = lockstep.run_all(run_variant, variants, args.jobs)
    except Exception:
        error = traceback.format_exc()
    return lockstep.finish(
        run_dir,
        experiment=EXPERIMENT,
        acceptance=ACCEPTANCE,
        not_measured=NOT_MEASURED,
        variant_count=len(variants),
        results=results,
        error=error,
    )


if __name__ == "__main__":
    sys.exit(main())
