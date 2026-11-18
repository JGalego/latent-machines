"""E06: spatial and structured dynamical machines.

Two branches, recorded as separate runs and never pooled:

- ``--branch exact`` (default): ``cellular/v1`` rules on a grid, stepped in
  lockstep (``lockstep.check``) with the integer oracle. Backends are the
  conventional vectorized ``torch-int`` code and the compiled
  ``conv-lookup`` circuit (3x3 convolution plus ReLU lookup) in four
  floating-point formats. Acceptance: cell-for-cell equality at every step.
- ``--branch numerical``: ``lds/v1`` systems driven by input signals, run by
  the dense ``direct`` recurrence and by a diagonal SSM with ``sequential``
  and ``chunked`` schedules, measured against the float64 reference under
  per-dtype tolerances declared in the run configuration before running.

Scalar-machine execution (a cellular rule interpreted one cell at a time on
an E00 memory backend) is not implemented.

    uv run --locked python -m latent_machines.experiments.e06
    uv run --locked python -m latent_machines.experiments.e06 --branch numerical

Exit status: 0 when every variant meets its contract, 1 on any mismatch,
tolerance excess, or snapshot disagreement, 2 if the run itself failed.
"""

import argparse
import json
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any

import torch

from latent_machines.backends.cellular import (
    PHASES,
    ConvLookupCellular,
    TorchIntCellular,
)
from latent_machines.backends.ssm import DiagonalSSM, DirectLDS
from latent_machines.experiments import lockstep
from latent_machines.machines import lds
from latent_machines.machines.cellular import Advance, Set
from latent_machines.tasks import cellular_workloads

EXPERIMENT = "E06"
MODULE = "latent_machines.experiments.e06"
RULES = ("B3/S23", "B36/S23", "B2/S")
FLOAT_DTYPES = ("float32", "float64", "bfloat16", "float16")
CELL_BACKENDS = ("torch-int", "conv-lookup")
LDS_BACKENDS = ("direct", "ssm-sequential", "ssm-chunked")
LDS_DTYPES = ("float64", "float32", "bfloat16")
# Declared before the run: max |y - y_ref| / rms(y_ref) over the horizon.
TOLERANCES = {"float64": 1e-9, "float32": 1e-4, "bfloat16": 1e-2}
ERROR_HORIZONS = (100, 1000, 10000, 100000)

EXACT_ACCEPTANCE = (
    "cell-for-cell equality of grid, generation, event, and work with the "
    "cellular/v1 oracle after every action, zero residue before decoding, and "
    "a snapshot restored through JSON equal to the live tensors"
)
NUMERICAL_ACCEPTANCE = (
    "max over the horizon of |y - y_ref| / rms(y_ref) at most the declared "
    "tolerance for the dtype, against the float64 lds/v1 reference; snapshot "
    "and resume must reproduce the uninterrupted outputs bit for bit"
)
EXACT_NOT_MEASURED = [
    "scalar-machine execution of the rule on an E00 memory backend",
    "1D elementary rules and non-totalistic neighborhoods",
    "GPU and multi-device halo exchange; communication is counted, not timed",
    "speed claims: timings are eager CPU PyTorch at one thread with Python "
    "overhead per generation, not optimized kernels",
]
NUMERICAL_NOT_MEASURED = [
    "float16 and fused or hardware scan kernels",
    "input-dependent (selective) SSMs: these systems are time-invariant",
    "speed claims: the sequential schedules pay Python overhead per sample",
]


# ---------- exact branch ----------


@dataclass(frozen=True)
class CellVariant:
    rule: str
    workload: str
    backend: str
    dtype: str
    size: int
    horizon: int
    snapshot_at: int
    seed: int
    trace_limit: int

    @property
    def name(self) -> str:
        return f"{self.backend}/{self.dtype}/{self.rule}/{self.workload}"


def cell_reproduce(variant: CellVariant, horizon: int) -> list[str]:
    return [
        "python", "-m", MODULE, "--rules", variant.rule,
        "--workloads", variant.workload, "--backends", variant.backend,
        "--dtypes", variant.dtype, "--size", str(variant.size),
        "--horizon", str(horizon), "--snapshot-at", str(variant.snapshot_at),
        "--seed", str(variant.seed),
    ]  # fmt: skip


def run_cell_variant(variant: CellVariant) -> dict[str, Any]:
    torch.set_num_threads(1)
    rng = random.Random(f"{variant.seed}:{variant.rule}:{variant.workload}")
    machine = cellular_workloads.machine(
        variant.rule, variant.workload, variant.size, variant.size, rng
    )
    actions = list(cellular_workloads.actions(machine, variant.horizon, rng))
    if variant.backend == "torch-int":
        backend: Any = TorchIntCellular(machine)
        stats: dict[str, Any] = {}
    else:
        backend = ConvLookupCellular(machine, getattr(torch, variant.dtype))
        stats = backend.circuit_stats()
    result = lockstep.check(
        machine,
        backend,
        actions,
        name=variant.name,
        snapshot_at=variant.snapshot_at,
        trace_limit=variant.trace_limit,
        reproduce=lambda horizon: cell_reproduce(variant, horizon),
    )

    # Timing pass: the backend alone over the same actions, with phase timers.
    generations = sum(isinstance(a, Advance) for a in actions)
    backend.profile = True
    state = backend.initial_state()
    started = time.perf_counter()
    for action in actions:
        state = backend.step(state, action).next_state
    backend_seconds = time.perf_counter() - started
    oracle_actions = actions[: min(len(actions), 200)]
    oracle_state = machine.initial_state()
    started = time.perf_counter()
    for action in oracle_actions:
        oracle_state = machine.step(oracle_state, action).next_state
    oracle_seconds = time.perf_counter() - started
    oracle_generations = sum(isinstance(a, Advance) for a in oracle_actions)

    side = variant.size
    border = 2 * side + 2 * side + 4
    result["metrics"] |= {
        "unit": "action (one generation or one Set)",
        "backend": variant.backend,
        "dtype": variant.dtype,
        "machine": f"cellular/v1 {variant.rule}",
        "rule": variant.rule,
        "workload": variant.workload,
        "boundary": machine.boundary,
        "grid": [side, side],
        "horizon": variant.horizon,
        "generations": generations,
        "sets": sum(isinstance(a, Set) for a in actions),
        "final_population": sum(map(sum, backend.decode(state).cells)),
        "logical_state_bits": machine.cells,
        "cell_updates": generations * machine.cells,
        "backend_seconds": round(backend_seconds, 4),
        "cells_per_second": round(generations * machine.cells / backend_seconds)
        if backend_seconds
        else None,
        "phase_seconds": {k: round(v, 4) for k, v in backend.phase_seconds.items()},
        "oracle_cells_per_second": round(
            oracle_generations * machine.cells / oracle_seconds
        )
        if oracle_seconds and oracle_generations
        else None,
        "halo_cells_per_generation": border,
        "halo_cells_copied_per_generation": border
        if machine.boundary == "torus"
        else 0,
        "padded_cells_per_generation": (side + 2) ** 2,
        **stats,
    }
    return result


# ---------- numerical branch ----------


@dataclass(frozen=True)
class LdsVariant:
    system: str
    signal: str
    backend: str
    dtype: str
    horizon: int
    chunk: int
    seed: int
    trace_limit: int

    @property
    def name(self) -> str:
        return f"{self.backend}/{self.dtype}/{self.system}/{self.signal}"

    @property
    def tolerance(self) -> float:
        return TOLERANCES[self.dtype]


def lds_reproduce(variant: LdsVariant) -> list[str]:
    return [
        "python", "-m", MODULE, "--branch", "numerical",
        "--systems", variant.system, "--signals", variant.signal,
        "--backends", variant.backend, "--dtypes", variant.dtype,
        "--horizon", str(variant.horizon), "--chunk", str(variant.chunk),
        "--seed", str(variant.seed),
    ]  # fmt: skip


def make_lds(variant: LdsVariant, system: lds.LinearSystem) -> Any:
    dtype = getattr(torch, variant.dtype)
    if variant.backend == "direct":
        return DirectLDS(system, dtype)
    schedule = variant.backend.removeprefix("ssm-")
    return DiagonalSSM(system, dtype, schedule, variant.chunk)


def run_lds_variant(variant: LdsVariant) -> dict[str, Any]:
    torch.set_num_threads(1)
    system = lds.SYSTEMS[variant.system]()
    inputs = (
        lds.noise(variant.horizon, variant.seed)
        if variant.signal == "noise"
        else lds.step(variant.horizon)
    )
    started = time.perf_counter()
    expected = lds.reference(system, inputs)
    reference_seconds = time.perf_counter() - started
    signal = torch.tensor(inputs, dtype=torch.float64)

    backend = make_lds(variant, system)
    started = time.perf_counter()
    outputs = backend.run(signal).to(torch.float64)
    seconds = time.perf_counter() - started

    # Snapshot at a chunk boundary near the midpoint, restore into a fresh
    # backend, and continue; the outputs must equal the uninterrupted ones.
    half = (variant.horizon // 2) // variant.chunk * variant.chunk
    first = make_lds(variant, system)
    head = first.run(signal[:half])
    snapshot = json.loads(json.dumps(first.snapshot()))
    resumed = make_lds(variant, system)
    resumed.restore(snapshot)
    tail = resumed.run(signal[half:])
    agrees = torch.equal(torch.cat([head, tail]).to(torch.float64), outputs)

    reference = torch.tensor(expected, dtype=torch.float64)
    rms = reference.square().mean().sqrt().item()
    error = (outputs - reference).abs() / rms
    if not torch.isfinite(error).all():
        error = torch.where(torch.isfinite(error), error, torch.inf)
    over = (error > variant.tolerance).nonzero().flatten()
    first_over = int(over[0]) if len(over) else None
    max_error = error.max().item()
    failure = None
    if first_over is not None:
        failure = {
            "variant": variant.name,
            "action_index": first_over,
            "expected": {"output": expected[first_over]},
            "actual": {
                "output": outputs[first_over].item(),
                "relative_error": error[first_over].item(),
            },
            "tolerance": variant.tolerance,
            "reproduce": lds_reproduce(variant),
        }
    marks = sorted({0, *(int(10**k) for k in range(1, 7)), variant.horizon - 1})
    trace = [
        {
            "variant": variant.name,
            "index": i,
            "input": inputs[i],
            "reference": expected[i],
            "output": outputs[i].item(),
            "relative_error": error[i].item(),
        }
        for i in marks
        if i < variant.horizon
    ][: variant.trace_limit]
    within = first_over is None
    return {
        "metrics": {
            "variant": variant.name,
            "unit": "output sample",
            "actions_checked": variant.horizon,
            "mismatches": len(over),
            "first_mismatch": first_over,
            "max_residue": max_error,
            "classification": "within tolerance" if within else "exceeds tolerance",
            "snapshot_resume": "agrees" if agrees else "differs",
            "end_to_end_seconds": round(seconds, 3),
            "contract": "approximate",
            "error_metric": "max |y - y_ref| / rms(y_ref)",
            "tolerance": variant.tolerance,
            "max_relative_error": max_error,
            "error_by_horizon": {
                str(h): error[:h].max().item()
                for h in ERROR_HORIZONS
                if h <= variant.horizon
            },
            "backend": variant.backend,
            "dtype": variant.dtype,
            "machine": f"lds/v1 {system.name}",
            "system": variant.system,
            "workload": variant.signal,
            "state_dimension": system.dimension,
            "state_bytes": backend.state_bytes,
            "samples_per_second": round(variant.horizon / seconds) if seconds else None,
            "reference_samples_per_second": round(variant.horizon / reference_seconds)
            if reference_seconds
            else None,
            "eigenvector_condition": getattr(backend, "condition", None),
            "spectral_radius": max(
                abs(v) for v in torch.linalg.eigvals(torch.tensor(system.a)).tolist()
            ),
            "cast_spectral_radius": backend.cast_spectral_radius,
            "reference_rms": rms,
        },
        "trace": trace,
        "failure": failure,
        "snapshot": {"variant": variant.name, "at": half, "state": snapshot},
    }


# ---------- command line ----------


def _names(
    parser: argparse.ArgumentParser, option: str, text: str, allowed
) -> list[str]:
    names = text.split(",")
    for name in names:
        if name not in allowed:
            parser.error(f"{option}: unknown {name!r}; choose from {list(allowed)}")
    return names


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="E06: cellular rules (exact) and SSM filters (numerical).",
    )
    parser.add_argument("--branch", choices=("exact", "numerical"), default="exact")
    parser.add_argument("--rules", default=",".join(RULES))
    parser.add_argument("--workloads", default=",".join(cellular_workloads.WORKLOADS))
    parser.add_argument("--size", type=int, default=32, help="grid side (exact)")
    parser.add_argument("--systems", default=",".join(lds.SYSTEMS))
    parser.add_argument("--signals", default=",".join(lds.INPUTS))
    parser.add_argument("--chunk", type=int, default=64, help="SSM chunk length")
    parser.add_argument("--backends", help="default: all for the branch")
    parser.add_argument("--dtypes", help="default: all for the branch")
    parser.add_argument(
        "--horizon",
        type=int,
        help="actions (exact, default 1000) or samples (numerical, default 10000)",
    )
    parser.add_argument(
        "--snapshot-at", type=int, help="exact branch; default horizon/2"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-limit", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--out", default="runs")
    args = parser.parse_args(argv)
    exact = args.branch == "exact"
    if args.horizon is None:
        args.horizon = 1000 if exact else 10000
    args.backends = _names(
        parser,
        "--backends",
        args.backends or ",".join(CELL_BACKENDS if exact else LDS_BACKENDS),
        CELL_BACKENDS if exact else LDS_BACKENDS,
    )
    args.dtypes = _names(
        parser,
        "--dtypes",
        args.dtypes or ",".join(FLOAT_DTYPES if exact else LDS_DTYPES),
        FLOAT_DTYPES if exact else LDS_DTYPES,
    )
    args.rules = _names(parser, "--rules", args.rules, RULES)
    args.workloads = _names(
        parser, "--workloads", args.workloads, tuple(cellular_workloads.WORKLOADS)
    )
    args.systems = _names(parser, "--systems", args.systems, tuple(lds.SYSTEMS))
    args.signals = _names(parser, "--signals", args.signals, lds.INPUTS)
    if min(args.horizon, args.jobs, args.size, args.chunk) < 1:
        parser.error("--horizon, --jobs, --size, and --chunk must be positive")
    return args


def variants_for(args: argparse.Namespace) -> list[Any]:
    if args.branch == "exact":
        snapshot_at = (
            args.horizon // 2 if args.snapshot_at is None else args.snapshot_at
        )
        out: list[Any] = []
        for backend in args.backends:
            dtypes = ["uint8"] if backend == "torch-int" else args.dtypes
            for dtype in dtypes:
                for rule in args.rules:
                    for workload in args.workloads:
                        out.append(
                            CellVariant(
                                rule,
                                workload,
                                backend,
                                dtype,
                                args.size,
                                args.horizon,
                                snapshot_at,
                                args.seed,
                                args.trace_limit,
                            )  # fmt: skip
                        )
        return out
    return [
        LdsVariant(
            system,
            signal,
            backend,
            dtype,
            args.horizon,
            args.chunk,
            args.seed,
            args.trace_limit,
        )  # fmt: skip
        for backend in args.backends
        for dtype in args.dtypes
        for system in args.systems
        for signal in args.signals
    ]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    exact = args.branch == "exact"
    run_dir, source, root = lockstep.start(args.out, f"e06-{args.branch}")
    variants = variants_for(args)
    config = {k: v for k, v in vars(args).items() if k != "out"} | {"out": args.out}
    if exact:
        config["snapshot_at"] = variants[0].snapshot_at if variants else None
    else:
        config |= {
            "tolerances": TOLERANCES,
            "error_metric": "max |y - y_ref| / rms(y_ref) over the horizon",
            "failure_policy": (
                "run the full horizon; count samples over tolerance; record the "
                "first one as the failure; classify the variant as exceeding"
            ),
        }
    header = lockstep.run_header(
        experiment=EXPERIMENT,
        module=MODULE,
        argv=argv,
        run_dir=run_dir,
        source=source,
        root=root,
        config=config,
        device="cpu",
    )
    if exact:
        details: dict[str, Any] = {
            "machines": {
                "cellular/v1": {
                    "rules": args.rules,
                    "boundaries": {
                        w: cellular_workloads.WORKLOADS[w] for w in args.workloads
                    },
                    "grid": [args.size, args.size],
                }
            },
            "backends": {
                "torch-int": "backends.cellular.TorchIntCellular: int16 neighbor sums "
                "from shifted slices, uint8 table lookup",
                "conv-lookup": "backends.cellular.ConvLookupCellular: pad, 3x3 conv2d, "
                "hat one-hot of the count, ReLU alive/dead units, rule-weighted sum",
                "phases": list(PHASES),
            },
            "seeds": {
                "seed": args.seed,
                "derivation": "random.Random(f'{seed}:{rule}:{workload}') draws the "
                "starting grid, then the actions",
            },
        }
    else:
        details = {
            "systems": {
                name: {
                    "dimension": s.dimension,
                    "a": s.a,
                    "b": s.b,
                    "c": s.c,
                    "d": s.d,
                    "spectral_radius": max(
                        abs(v) for v in torch.linalg.eigvals(torch.tensor(s.a)).tolist()
                    ),
                }
                for name, s in ((n, lds.SYSTEMS[n]()) for n in args.systems)
            },
            "backends": {
                "direct": "backends.ssm.DirectLDS: dense x <- A x + B u in the dtype",
                "ssm-sequential": "backends.ssm.DiagonalSSM: modes from float64 eig, "
                "cast; one sample per step, complex as real pairs",
                "ssm-chunked": "backends.ssm.DiagonalSSM: chunked lower-triangular "
                "kernel, powers computed in float64 then cast",
            },
            "seeds": {
                "seed": args.seed,
                "derivation": "random.Random(f'{seed}:noise')",
            },
        }
    lockstep.records.write_json(
        run_dir / "run.json",
        header
        | details
        | {"variants": [asdict(v) | {"name": v.name} for v in variants]},
    )
    results, error = [], None
    try:
        results = lockstep.run_all(
            run_cell_variant if exact else run_lds_variant, variants, args.jobs
        )
    except Exception:
        error = traceback.format_exc()
    return lockstep.finish(
        run_dir,
        experiment=EXPERIMENT,
        acceptance=EXACT_ACCEPTANCE if exact else NUMERICAL_ACCEPTANCE,
        not_measured=EXACT_NOT_MEASURED if exact else NUMERICAL_NOT_MEASURED,
        variant_count=len(variants),
        results=results,
        error=error,
    )


if __name__ == "__main__":
    sys.exit(main())
