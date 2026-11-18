"""E07: optimization-defined state updates.

Runs each ``optimizer-memory/v1`` setting (see ``machines/optimizer_memory.py``)
three ways at once over every workload and number format:

- the tensor backend (``backends/optimizer_memory.py``), the implementation
  under test;
- the float64 reference (own-reference conformance). The pass condition is
  local: each backend step must match one reference step taken from the
  backend's own state, within the dtype's tolerance. Agreement along the
  reference's independent trajectory is reported, not required, because
  rounding differences can grow along it, and some settings amplify them;
- the ``memory/v1`` oracle, against which the backend's decoded readout is
  compared (RAM conformance). A RAM deviation is a finding about the
  optimizer setting, not a failure, except for the plain ``sgd`` setting,
  whose RAM exactness is the acceptance case.

Settings change one feature at a time from ``sgd`` (lr 1, sequential, no
momentum, decay, or normalization, linear model). Workloads are E00's
``memory/v1`` streams plus online linear regression with dense keys.

At ``snapshot_at`` the backend's full state (parameters, momentum, chunk-start
copy, step count) goes through JSON and must restore exactly. For linear
settings the run also restarts from the decoded ``memory/v1`` words alone and
reports whether that canonical snapshot reproduces the decoded behavior.

    uv run --locked python -m latent_machines.experiments.e07
    uv run --locked python -m latent_machines.experiments.e07 --dtypes bfloat16

Read events may differ from the reference only at threshold ties: in bits
whose reference value lies within tolerance of 0.5, where decoding is
discontinuous.

Exit status: 0 when every variant matches its own reference locally and
``sgd`` matches ``memory/v1`` exactly, 1 otherwise, 2 if the run failed.
"""

import argparse
import json
import math
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from itertools import islice
from typing import Any

import torch

from latent_machines.backends.optimizer_memory import OptimizerMemory, TensorState
from latent_machines.experiments import lockstep, records
from latent_machines.machines.events import Event, EventKind
from latent_machines.machines.memory import VERSION as MEMORY_VERSION
from latent_machines.machines.memory import Memory, Write
from latent_machines.machines.optimizer_memory import (
    VERSION,
    Optimizer,
    OptimizerMemoryReference,
    ReferenceState,
)
from latent_machines.tasks.memory_workloads import WORKLOADS, actions
from latent_machines.tasks.regression import Regression

EXPERIMENT = "E07"
MODULE = "latent_machines.experiments.e07"
DTYPES = ("float32", "float64", "bfloat16")
DEFAULT_DTYPES = ("float32", "float64")
REGRESSION = "regression"
ALL_WORKLOADS = (*WORKLOADS, REGRESSION)
# |tensor - reference| <= atol + rtol * |reference| for parameters and outputs.
TOLERANCE = {"float64": 1e-9, "float32": 1e-4, "bfloat16": 2e-2}
SETTINGS: dict[str, Optimizer] = {
    "sgd": Optimizer(),
    "lr-0.9": Optimizer(lr=0.9),
    "lr-0.5": Optimizer(lr=0.5),
    "chunk-4": Optimizer(chunk=4),
    "momentum-0.5": Optimizer(momentum=0.5),
    "decay-0.001": Optimizer(decay=1e-3),
    "grad-norm": Optimizer(normalize=True),
    "mlp": Optimizer(model="mlp"),
}
# Written before the first run: what each setting should do to RAM semantics.
EXPECTED = {
    "sgd": "exact delta replacement",
    "lr-0.9": "decodes correctly with residue up to 0.1: writes move 90% of the way",
    "lr-0.5": "deviates on the first bit change: a write lands on the 0.5 threshold",
    "chunk-4": "deviates when one slot is written twice in a chunk: corrections sum",
    "momentum-0.5": "overshoots and keeps moving after writes, even on reads and idles",
    "decay-0.001": "forgets during idle: a stored 1 falls below 0.5 after 693 steps",
    "grad-norm": "deviates when a write changes several bits: the step has unit norm",
    "mlp": "deviates from the first write: shared hidden units make writes interfere",
}
ACCEPTANCE = (
    "every backend step matches one step of the float64 reference taken from "
    "the backend's own state, within the dtype's tolerance (read events may "
    "differ only at threshold ties); the full-state snapshot restores exactly; "
    "and the sgd setting matches memory/v1 exactly on every memory workload. "
    "Agreement along the reference's independent trajectory is reported, not "
    "required"
)
NOT_MEASURED = [
    "backend latency in isolation: tensor_seconds times the eager per-step calls, "
    "end_to_end_seconds adds the reference and checks",
    "optimizer settings in combination: each setting changes one feature",
    "chunked kernels with intra-chunk correction (the WY or UT forms)",
    "learned or input-dependent learning rates and gates",
    "devices other than CPU",
]


@dataclass(frozen=True)
class Variant:
    setting: str
    workload: str
    dtype: str
    capacity: int
    word_bits: int
    horizon: int
    snapshot_at: int
    seed: int
    trace_limit: int

    @property
    def name(self) -> str:
        return f"{self.setting}/{self.dtype}/{self.workload}"


def reproduce_command(variant: Variant) -> list[str]:
    return [
        "python", "-m", MODULE,
        "--settings", variant.setting, "--workloads", variant.workload,
        "--dtypes", variant.dtype, "--capacity", str(variant.capacity),
        "--word-bits", str(variant.word_bits), "--horizon", str(variant.horizon),
        "--snapshot-at", str(variant.snapshot_at), "--seed", str(variant.seed),
    ]  # fmt: skip


def _excess(
    actual: torch.Tensor, expected: list[float], tol: float
) -> tuple[float, float]:
    """Largest ``|a - e| - tol * (1 + |e|)``, positive when out of tolerance,
    and the largest absolute error."""
    ref = torch.tensor(expected, dtype=torch.float64)
    got = actual.reshape(-1).to(torch.float64)
    diff = (got - ref).abs()
    if not torch.isfinite(diff).all():
        same = (got == ref) | (got.isnan() & ref.isnan())
        diff = torch.where(same, 0.0, math.inf)
    return (diff - tol * (1 + ref.abs())).max().item(), diff.max().item()


def compare(
    tensor: TensorState,
    ref: ReferenceState,
    outputs: tuple[torch.Tensor, list[float]] | None,
    tol: float,
) -> tuple[bool, float]:
    """Whether the backend is within tolerance of the reference, and the error."""
    worst, error = -math.inf, 0.0
    pairs = [(tensor.params[n], ref.params[n]) for n in ref.params]
    if outputs is not None:
        pairs.append(outputs)
    for actual, expected in pairs:
        excess, err = _excess(actual, expected, tol)
        worst, error = max(worst, excess), max(error, err)
    return worst <= 0, error


def event_agreement(
    actual: Event, expected: Event, reference_output: list[float] | None, tol: float
) -> str:
    """``same``, ``tie``, or ``differs``.

    Decoding thresholds at 0.5, so it is discontinuous there. Read events
    that differ only in bits whose reference value lies within tolerance of
    0.5 are threshold ties, not mismatches.
    """
    if actual == expected:
        return "same"
    if (
        reference_output is not None
        and actual.kind is expected.kind is EventKind.OUTPUT
        and actual.value is not None
        and expected.value is not None
    ):
        flipped = actual.value ^ expected.value
        if all(
            abs(reference_output[i] - 0.5) <= tol * 1.5
            for i in range(len(reference_output))
            if flipped >> i & 1
        ):
            return "tie"
    return "differs"


@dataclass
class Agreement:
    """Running comparison of the backend with the float64 reference."""

    tol: float
    mismatches: int = 0
    first_mismatch: int | None = None
    threshold_ties: int = 0
    max_error: float = 0.0

    def record(
        self,
        index: int,
        state: TensorState,
        ref: ReferenceState,
        outputs: tuple[torch.Tensor, list[float]] | None,
        events: tuple[Event, Event] | None = None,
    ) -> float:
        within, error = compare(state, ref, outputs, self.tol)
        self.max_error = max(self.max_error, error)
        agreement = "same"
        if events is not None:
            reference_output = outputs[1] if outputs else None
            agreement = event_agreement(*events, reference_output, self.tol)
        self.threshold_ties += agreement == "tie"
        if not within or agreement == "differs":
            self.mismatches += 1
            if self.first_mismatch is None:
                self.first_mismatch = index
        return error

    def summary(self) -> dict[str, Any]:
        return {
            "classification": (
                "within tolerance" if self.mismatches == 0 else "exceeds tolerance"
            ),
            "mismatches": self.mismatches,
            "first_mismatch": self.first_mismatch,
            "threshold_ties": self.threshold_ties,
            "max_error": self.max_error,
        }


def paired_outputs(
    actual: Any, expected: Any
) -> tuple[torch.Tensor, list[float]] | None:
    """The backend's and the reference's read outputs, when both read."""
    if actual.output is None or expected.output is None:
        return None
    return actual.output, expected.output


def words_of(readout: torch.Tensor) -> tuple[int, ...]:
    columns = (readout > 0.5).T.tolist()
    return tuple(sum(1 << i for i, bit in enumerate(c) if bit) for c in columns)


def residue_of(readout: torch.Tensor) -> float:
    r = readout.to(torch.float64)
    value = torch.minimum(r.abs(), (r - 1).abs()).max().item()
    return value if math.isfinite(value) else math.inf


def snapshot_roundtrip(
    backend: OptimizerMemory, state: TensorState
) -> tuple[TensorState, str, dict[str, Any]]:
    snapshot = json.loads(json.dumps(backend.snapshot(state)))
    restored = backend.restore(snapshot)
    same = (
        restored.steps == state.steps
        and restored.halted == state.halted
        and all(torch.equal(restored.params[n], state.params[n]) for n in state.params)
        and all(
            torch.equal(restored.momentum[n], state.momentum[n]) for n in state.momentum
        )
        and (restored.chunk_start is None) == (state.chunk_start is None)
        and all(
            torch.equal(restored.chunk_start[n], state.chunk_start[n])
            for n in (state.chunk_start or {})
        )
    )
    return restored, "agrees" if same else "differs", snapshot


def run_memory(variant: Variant) -> dict[str, Any]:
    optimizer = SETTINGS[variant.setting]
    tol = TOLERANCE[variant.dtype]
    machine = Memory(variant.capacity, variant.word_bits)
    reference = OptimizerMemoryReference(variant.capacity, variant.word_bits, optimizer)
    backend = OptimizerMemory(
        variant.capacity, variant.word_bits, optimizer, getattr(torch, variant.dtype)
    )
    rng = random.Random(f"{variant.seed}:{variant.workload}")
    stream = list(actions(variant.workload, machine, variant.horizon, rng))

    oracle, ref, state = (
        machine.initial_state(),
        reference.initial_state(),
        backend.initial_state(),
    )
    trace: list[dict[str, Any]] = []
    snapshot, snapshot_resume = None, "not reached"
    canonical_start: tuple[tuple[int, ...], int] | None = None
    live: list[tuple[Any, tuple[int, ...]]] = []
    local, trajectory = Agreement(tol), Agreement(tol)
    deviations = writes = 0
    first_deviation = first_margin_loss = None
    deviation = None
    max_residue = interference = 0.0
    tensor_seconds = 0.0
    started = time.perf_counter()
    readout = backend.readout(state)
    for index, action in enumerate(stream):
        if index == variant.snapshot_at:
            state, snapshot_resume, snapshot = snapshot_roundtrip(backend, state)
            canonical_start = (words_of(readout), state.steps)
        expected = machine.step(oracle, action)
        r = reference.step(ref, action)
        # One step of the reference from the backend's own (exact) state.
        one = reference.step(reference.restore(backend.snapshot(state)), action)
        tick = time.perf_counter()
        actual = backend.step(state, action)
        tensor_seconds += time.perf_counter() - tick
        before = readout
        readout = backend.readout(actual.next_state)

        error = local.record(
            index,
            actual.next_state,
            one.next_state,
            paired_outputs(actual, one),
            (actual.event, one.event),
        )
        trajectory.record(
            index,
            actual.next_state,
            r.next_state,
            paired_outputs(actual, r),
            (actual.event, r.event),
        )

        words = words_of(readout)
        residue = residue_of(readout)
        max_residue = max(max_residue, residue)
        if residue > 0 and first_margin_loss is None:
            first_margin_loss = index
        ram_ok = (
            actual.event == expected.event
            and words == expected.next_state.words
            and actual.next_state.halted == expected.next_state.halted
        )
        if not ram_ok:
            deviations += 1
            if first_deviation is None:
                first_deviation = index
                deviation = {
                    "action_index": index,
                    "action": lockstep.action_json(action),
                    "expected": {
                        "event": lockstep.event_json(expected.event),
                        "words": list(expected.next_state.words),
                    },
                    "actual": {
                        "event": lockstep.event_json(actual.event),
                        "words": list(words),
                        "readout_columns_changed": [
                            a
                            for a, (x, y) in enumerate(
                                zip(words, expected.next_state.words, strict=True)
                            )
                            if x != y
                        ],
                    },
                    "reproduce": reproduce_command(variant),
                }
        if isinstance(action, Write) and expected.event.kind is EventKind.CONTINUE:
            if actual.event == expected.event:
                writes += 1
                others = torch.cat(
                    [before[:, : action.address], before[:, action.address + 1 :]],
                    dim=1,
                ) - torch.cat(
                    [readout[:, : action.address], readout[:, action.address + 1 :]],
                    dim=1,
                )
                interference = max(interference, others.abs().max().item())
        if index >= variant.snapshot_at:
            live.append((actual.event, words))
        if index < variant.trace_limit:
            trace.append(
                {
                    "variant": variant.name,
                    "index": index,
                    "action": lockstep.action_json(action),
                    "event": lockstep.event_json(actual.event),
                    "reference_error": error,
                    "residue": residue,
                    "ram_conforms": ram_ok,
                }
            )
        oracle, ref, state = expected.next_state, r.next_state, actual.next_state

    canonical = "not reached"
    if canonical_start is not None:
        if optimizer.model != "linear":
            canonical = "not applicable"
        else:
            canonical = canonical_resume(
                backend, stream, variant.snapshot_at, canonical_start, live
            )

    if deviations == 0:
        ram = "exact" if max_residue == 0 else "conforms with residue"
    else:
        ram = "deviates"
    return {
        "metrics": {
            "variant": variant.name,
            "machine": VERSION,
            "setting": variant.setting,
            "optimizer": asdict(optimizer),
            "workload": variant.workload,
            "dtype": variant.dtype,
            "unit": (
                "memory/v1 action: backend parameters and read output compared "
                "with the float64 reference; decoded readout compared with memory/v1"
            ),
            "actions_checked": len(stream),
            **conformance_fields(local, trajectory, tol),
            "max_residue": max_residue,
            "ram_conformance": ram,
            "ram_expected": EXPECTED[variant.setting],
            "ram_deviations": deviations,
            "first_ram_deviation": first_deviation,
            "first_margin_loss": first_margin_loss,
            "ram_deviation": deviation,
            "snapshot_resume": snapshot_resume,
            "canonical_snapshot_resume": canonical,
            "max_interference": interference,
            "valid_writes": writes,
            "optimizer_steps": state.steps,
            "ticks_per_action": 1,
            "state_bytes": backend.state_bytes(state),
            "tensor_seconds": round(tensor_seconds, 4),
            "end_to_end_seconds": round(time.perf_counter() - started, 3),
        },
        "trace": trace,
        "snapshot": snapshot,
    }


def conformance_fields(
    local: Agreement, trajectory: Agreement, tol: float
) -> dict[str, Any]:
    """Local agreement is the pass condition; trajectory agreement is reported."""
    summary = local.summary()
    return {
        "classification": summary["classification"],
        "mismatches": summary["mismatches"],
        "first_mismatch": summary["first_mismatch"],
        "threshold_ties": summary["threshold_ties"],
        "max_reference_error": summary["max_error"],
        "tolerance": tol,
        "trajectory": trajectory.summary(),
    }


def canonical_resume(
    backend: OptimizerMemory,
    stream: list[Any],
    start: int,
    canonical: tuple[tuple[int, ...], int],
    live: list[tuple[Any, tuple[int, ...]]],
) -> str:
    """Restart from decoded words alone; compare decoded events and words."""
    words, steps = canonical
    state = backend.from_words(words, steps)
    for offset, action in enumerate(stream[start:]):
        transition = backend.step(state, action)
        state = transition.next_state
        if (transition.event, words_of(backend.readout(state))) != live[offset]:
            return f"differs at action {start + offset}"
    return "agrees"


def run_regression(variant: Variant) -> dict[str, Any]:
    optimizer = SETTINGS[variant.setting]
    tol = TOLERANCE[variant.dtype]
    dtype = getattr(torch, variant.dtype)
    keys, values = variant.capacity, variant.word_bits
    task = Regression(keys, values, variant.seed)
    reference = OptimizerMemoryReference(keys, values, optimizer)
    backend = OptimizerMemory(keys, values, optimizer, dtype)
    ref, state = reference.initial_state(), backend.initial_state()
    losses: list[float] = []
    trace: list[dict[str, Any]] = []
    snapshot, snapshot_resume = None, "not reached"
    local, trajectory, tensor_seconds = Agreement(tol), Agreement(tol), 0.0
    started = time.perf_counter()
    for index, (x, y) in enumerate(islice(task.samples(), variant.horizon)):
        if index == variant.snapshot_at:
            state, snapshot_resume, snapshot = snapshot_roundtrip(backend, state)
        # The key and target in the backend's dtype, so the local check
        # compares arithmetic, not input rounding.
        key, target = torch.tensor(x, dtype=dtype), torch.tensor(y, dtype=dtype)
        own = reference.restore(backend.snapshot(state))
        one_prediction = reference.predict(own.params, key.tolist())
        one = reference.update(own, key.tolist(), target.tolist())
        tick = time.perf_counter()
        prediction = backend.predict(state, key)
        state = backend.update(state, key, target)
        tensor_seconds += time.perf_counter() - tick
        ref_prediction = reference.predict(ref.params, x)
        ref = reference.update(ref, x, y)
        error_vector = prediction.to(torch.float64) - torch.tensor(
            y, dtype=torch.float64
        )
        losses.append(0.5 * error_vector.pow(2).sum().item())
        error = local.record(index, state, one, (prediction, one_prediction))
        trajectory.record(index, state, ref, (prediction, ref_prediction))
        if index < variant.trace_limit:
            trace.append(
                {
                    "variant": variant.name,
                    "index": index,
                    "online_loss": losses[-1],
                    "reference_error": error,
                }
            )
    window = max(1, min(100, len(losses) // 10))
    parameter_error = None
    if optimizer.model == "linear":
        target_map = torch.tensor(task.target_map(), dtype=torch.float64)
        w = state.params["W"].to(torch.float64)
        parameter_error = ((w - target_map).norm() / target_map.norm()).item()
    return {
        "metrics": {
            "variant": variant.name,
            "machine": VERSION,
            "setting": variant.setting,
            "optimizer": asdict(optimizer),
            "workload": variant.workload,
            "dtype": variant.dtype,
            "unit": (
                "regression sample: parameters and prediction compared with "
                "the float64 reference"
            ),
            "actions_checked": len(losses),
            **conformance_fields(local, trajectory, tol),
            "max_residue": None,
            "ram_conformance": "not applicable",
            "ram_expected": EXPECTED[variant.setting],
            "snapshot_resume": snapshot_resume,
            "canonical_snapshot_resume": "not applicable",
            "online_loss_first": sum(losses[:window]) / window,
            "online_loss_last": sum(losses[-window:]) / window,
            "loss_window": window,
            "relative_parameter_error": parameter_error,
            "finite": all(math.isfinite(v) for v in losses),
            "optimizer_steps": state.steps,
            "ticks_per_action": 1,
            "state_bytes": backend.state_bytes(state),
            "tensor_seconds": round(tensor_seconds, 4),
            "end_to_end_seconds": round(time.perf_counter() - started, 3),
        },
        "trace": trace,
        "snapshot": snapshot,
    }


def run_variant(variant: Variant) -> dict[str, Any]:
    torch.set_num_threads(1)
    if variant.workload == REGRESSION:
        return run_regression(variant)
    return run_memory(variant)


def accepted(metrics: dict[str, Any]) -> bool:
    return (
        metrics["mismatches"] == 0
        and metrics["snapshot_resume"] != "differs"
        and (
            metrics["setting"] != "sgd"
            or metrics["ram_conformance"] in ("exact", "not applicable")
        )
    )


def finish(
    run_dir: Any, variant_count: int, results: list[dict[str, Any]], error: str | None
) -> int:
    metrics = [r["metrics"] for r in results]
    if error is None:
        with (run_dir / "trace.jsonl").open("w") as handle:
            for result in results:
                for line in result["trace"]:
                    handle.write(json.dumps(line) + "\n")
        (run_dir / "snapshots").mkdir(exist_ok=True)
        for result in results:
            if result["snapshot"] is not None:
                name = result["metrics"]["variant"].replace("/", "-")
                records.write_json(
                    run_dir / "snapshots" / f"{name}.json", result["snapshot"]
                )
        deviations = [
            m["ram_deviation"] for m in metrics if m.get("ram_deviation") is not None
        ]
        if deviations:
            records.write_json(run_dir / "ram_deviations.json", deviations)
    if error is not None:
        status = "error"
    elif len(metrics) == variant_count and all(accepted(m) for m in metrics):
        status = "passed"
    else:
        status = "failed"
    residues = [m["max_residue"] for m in metrics if m["max_residue"] is not None]
    records.write_json(
        run_dir / "metrics.json",
        {
            "schema_version": records.SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": status,
            "finished_at": records.now(),
            "acceptance": ACCEPTANCE,
            "totals": {
                "variants": variant_count,
                "variants_completed": len(metrics),
                "actions_checked": sum(m["actions_checked"] for m in metrics),
                "mismatches": sum(m["mismatches"] for m in metrics),
                "max_residue": max(residues, default=None),
                "max_reference_error": max(
                    (m["max_reference_error"] for m in metrics), default=None
                ),
                "ram_classification": {
                    label: sum(m["ram_conformance"] == label for m in metrics)
                    for label in (
                        "exact",
                        "conforms with residue",
                        "deviates",
                        "not applicable",
                    )
                },
            },
            "variants": metrics,
            "not_measured": NOT_MEASURED,
            "error": error,
        },
    )
    for m in metrics:
        print(
            f"{m['variant']:<40} {m['classification']:<17} "
            f"err {m['max_reference_error']:.2g}  "
            f"trajectory {m['trajectory']['classification']:<17} "
            f"RAM {m['ram_conformance']:<21} "
            f"first dev {m.get('first_ram_deviation')}"
        )
    if error:
        print(error, file=sys.stderr)
    print(f"status: {status}\nrecord: {run_dir}")
    return {"passed": 0, "failed": 1}.get(status, 2)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="E07: optimizer-defined memory updates versus RAM.",
    )
    parser.add_argument("--capacity", type=int, default=16, help="keys (slots)")
    parser.add_argument("--word-bits", type=int, default=8, help="values (bits)")
    parser.add_argument("--horizon", type=int, default=1000, help="actions per variant")
    parser.add_argument("--snapshot-at", type=int, help="default: horizon/2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settings", default=",".join(SETTINGS))
    parser.add_argument("--workloads", default=",".join(ALL_WORKLOADS))
    parser.add_argument("--dtypes", default=",".join(DEFAULT_DTYPES))
    parser.add_argument("--trace-limit", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--out", default="runs")
    args = parser.parse_args(argv)
    for option, allowed in (
        ("settings", SETTINGS),
        ("workloads", ALL_WORKLOADS),
        ("dtypes", DTYPES),
    ):
        for name in getattr(args, option).split(","):
            if name not in allowed:
                parser.error(
                    f"unknown {option[:-1]} {name!r}; choose from {list(allowed)}"
                )
    if args.horizon < 1 or args.jobs < 1:
        parser.error("--horizon and --jobs must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    run_dir, source, root = lockstep.start(args.out, "e07")
    snapshot_at = args.horizon // 2 if args.snapshot_at is None else args.snapshot_at
    variants = [
        Variant(
            setting=setting,
            workload=workload,
            dtype=dtype,
            capacity=args.capacity,
            word_bits=args.word_bits,
            horizon=args.horizon,
            snapshot_at=snapshot_at,
            seed=args.seed,
            trace_limit=args.trace_limit,
        )
        for setting in args.settings.split(",")
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
        device="cpu",
    )
    records.write_json(
        run_dir / "run.json",
        header
        | {
            "seeds": {
                "seed": args.seed,
                "derivation": (
                    "random.Random(f'{seed}:{workload}') per memory workload; "
                    "Regression(keys, values, seed) for regression; MLP "
                    "initialization from random.Random(f'{optimizer.seed}:mlp')"
                ),
            },
            "machine": {
                "version": VERSION,
                "reference": "machines/optimizer_memory.py, plain Python float64",
                "ram_oracle": MEMORY_VERSION,
                "keys": args.capacity,
                "values": args.word_bits,
            },
            "backend": {
                "module": "latent_machines.backends.optimizer_memory",
                "kernel": "eager PyTorch; linear in closed form, MLP by autograd",
                "decoding": "threshold each readout entry at 0.5",
                "tolerance": "|tensor - reference| <= tol * (1 + |reference|)",
                "tolerances": TOLERANCE,
            },
            "settings": {
                name: asdict(opt) | {"expected_ram_behavior": EXPECTED[name]}
                for name, opt in SETTINGS.items()
                if name in args.settings.split(",")
            },
            "variants": [asdict(v) | {"name": v.name} for v in variants],
        },
    )
    results, error = [], None
    try:
        results = lockstep.run_all(run_variant, variants, args.jobs)
    except Exception:
        error = traceback.format_exc()
    return finish(run_dir, len(variants), results, error)


if __name__ == "__main__":
    sys.exit(main())
