"""E01: exact streaming machines.

Steps each E01 oracle and its compiled tensor backend in lockstep (see
``lockstep.check``) in every number format:

- ``transducer/v1`` targets (parity, a mod-5 counter over increments 0-2, and
  S3 permutation tracking) on the ReLU lookup circuit, fed random symbols.
- ``stack/v1`` on the delta-rule stack, driven by the bracket parser, random
  push/pop, and fill-and-drain boundary cycles.

Writes a run record under ``runs/`` following "Run records and results" in
EXPERIMENTS.md:

    uv run --locked python -m latent_machines.experiments.e01
    uv run --locked python -m latent_machines.experiments.e01 --horizon 1000000 --jobs 8

Exit status: 0 when every variant conforms, 1 on any mismatch or snapshot
disagreement, 2 if the run itself failed.
"""

import argparse
import math
import random
import sys
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

from latent_machines.backends.delta_stack import DeltaStack
from latent_machines.backends.lookup import LookupTransducer
from latent_machines.experiments import lockstep
from latent_machines.machines.stack import Stack
from latent_machines.machines.transducer import (
    Transducer,
    modular_counter,
    parity,
    permutation_tracker,
)
from latent_machines.tasks import stack_workloads, transducer_workloads

EXPERIMENT = "E01"
MODULE = "latent_machines.experiments.e01"
DTYPES = ("float32", "float64", "bfloat16", "float16")
TRANSDUCERS: dict[str, Callable[[], Transducer]] = {
    "parity": parity,
    "mod5-counter": lambda: modular_counter(5, 3),
    "s3-tracker": lambda: permutation_tracker(3),
}
MACHINES = (*TRANSDUCERS, "stack")
ACCEPTANCE = (
    "zero event, work, and decoded-state mismatches at every step, explicit "
    "overflow and underflow faults, and a snapshot restored through JSON equal "
    "to the live tensors"
)
NOT_MEASURED = [
    "backend latency: end_to_end_seconds includes oracle stepping and checks",
    "hidden-unit and output margins: residue covers the fed-back state only",
    "smooth (SwiGLU) and direct signed or rotation recurrences: not yet built",
    "longer inputs versus more states or deeper stacks as separate axes",
]


@dataclass(frozen=True)
class Variant:
    machine: str
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
    def backend(self) -> str:
        return "delta-stack" if self.machine == "stack" else "relu-lookup"

    @property
    def name(self) -> str:
        return f"{self.backend}/{self.dtype}/{self.machine}/{self.workload}"


def reproduce_command(variant: Variant, horizon: int) -> list[str]:
    return [
        "python", "-m", MODULE,
        "--machines", variant.machine, "--stack-workloads", variant.workload,
        "--dtypes", variant.dtype,
        "--stack-capacity", str(variant.capacity),
        "--word-bits", str(variant.word_bits),
        "--horizon", str(horizon), "--snapshot-at", str(variant.snapshot_at),
        "--seed", str(variant.seed),
    ]  # fmt: skip


def run_variant(variant: Variant) -> dict[str, Any]:
    """Run one machine, workload, and number format in lockstep."""
    torch.set_num_threads(1)
    dtype = getattr(torch, variant.dtype)
    rng = random.Random(f"{variant.seed}:{variant.machine}:{variant.workload}")
    machine: Any
    if variant.machine == "stack":
        machine = Stack(variant.capacity, variant.word_bits)
        backend: Any = DeltaStack(machine, dtype, variant.device)
        actions = stack_workloads.actions(
            variant.workload, machine, variant.horizon, rng
        )
        logical_bits = machine.capacity * machine.word_bits + math.ceil(
            math.log2(machine.capacity + 1)
        )
        stats = {
            "stored_weights": backend.shift_up.numel(),
            "nonzero_weights": int(backend.shift_up.count_nonzero()),
        }
    else:
        machine = TRANSDUCERS[variant.machine]()
        backend = LookupTransducer(machine, dtype, variant.device)
        actions = transducer_workloads.actions(machine, variant.horizon, rng)
        logical_bits = math.ceil(math.log2(machine.num_states))
        stats = backend.circuit_stats()

    result = lockstep.check(
        machine,
        backend,
        actions,
        name=variant.name,
        snapshot_at=variant.snapshot_at,
        trace_limit=variant.trace_limit,
        reproduce=lambda horizon: reproduce_command(variant, horizon),
    )
    result["metrics"] |= {
        "backend": variant.backend,
        "dtype": variant.dtype,
        "machine": variant.machine,
        "workload": variant.workload,
        "horizon": variant.horizon,
        "logical_state_bits": logical_bits,
        **stats,
    }
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="E01 conformance of compiled transducers and the delta-rule stack.",
    )
    parser.add_argument("--machines", default=",".join(MACHINES))
    parser.add_argument(
        "--stack-workloads", default=",".join(stack_workloads.WORKLOADS)
    )
    parser.add_argument("--stack-capacity", type=int, default=8)
    parser.add_argument("--word-bits", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1000, help="actions per variant")
    parser.add_argument(
        "--snapshot-at", type=int, help="action index to snapshot (default: horizon/2)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtypes", default=",".join(DTYPES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--trace-limit", type=int, default=1000, help="transitions traced per variant"
    )
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--out", default="runs", help="parent directory of the run")
    args = parser.parse_args(argv)
    for option, names, allowed in (
        ("--machines", args.machines, MACHINES),
        ("--stack-workloads", args.stack_workloads, tuple(stack_workloads.WORKLOADS)),
        ("--dtypes", args.dtypes, DTYPES),
    ):
        for name in names.split(","):
            if name not in allowed:
                parser.error(f"{option}: unknown {name!r}; choose from {list(allowed)}")
    if args.horizon < 1 or args.jobs < 1:
        parser.error("--horizon and --jobs must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    run_dir, source, root = lockstep.start(args.out, "e01")
    snapshot_at = args.horizon // 2 if args.snapshot_at is None else args.snapshot_at
    pairs = [
        (machine, workload)
        for machine in args.machines.split(",")
        for workload in (
            args.stack_workloads.split(",") if machine == "stack" else ["symbols"]
        )
    ]
    variants = [
        Variant(
            machine=machine,
            workload=workload,
            dtype=dtype,
            capacity=args.stack_capacity,
            word_bits=args.word_bits,
            horizon=args.horizon,
            snapshot_at=snapshot_at,
            seed=args.seed,
            trace_limit=args.trace_limit,
            device=args.device,
        )
        for dtype in args.dtypes.split(",")
        for machine, workload in pairs
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
    transducers = {
        name: {"states": t.num_states, "alphabet": t.alphabet_size, "digest": t.digest}
        for name, t in ((name, make()) for name, make in TRANSDUCERS.items())
    }
    lockstep.records.write_json(
        run_dir / "run.json",
        header
        | {
            "seeds": {
                "seed": args.seed,
                "derivation": "random.Random(f'{seed}:{machine}:{workload}')",
            },
            "machines": {
                "transducer/v1": transducers,
                "stack/v1": {
                    "capacity": args.stack_capacity,
                    "word_bits": args.word_bits,
                },
            },
            "backends": {
                "relu-lookup": "backends.lookup: z = relu(W_in [s; x] - 1), "
                "next = W_next z, output = W_out z; eager PyTorch",
                "delta-stack": "latent_machines.backends.delta_stack: delta-rule "
                "memory plus one-hot depth pointer moved by shift matrices",
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
