"""Lockstep conformance shared by the E00 and E01 runners.

``check`` steps an oracle and a backend over the same actions and compares
every transition: equal events and work, equal decoded state, and the
backend's residue before decoding. At ``snapshot_at`` it snapshots the
backend's decoded state, restores it through JSON, and continues from the
restored backend state, which must equal the live one exactly.

The oracle needs ``initial_state``, ``step``, ``snapshot``, and ``restore``;
the backend needs ``initial_state``, ``step``, ``decode``, ``encode``, and
``residue``. ``run_all`` and ``finish`` run variants and write the run record
described in "Run records and results" in EXPERIMENTS.md.
"""

import hashlib
import json
import multiprocessing
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

from latent_machines.experiments import records
from latent_machines.machines.events import Event


def action_json(action: Any) -> dict[str, Any]:
    return {"op": type(action).__name__.lower(), **asdict(action)}


def event_json(event: Event) -> dict[str, Any]:
    data: dict[str, Any] = {"kind": event.kind.value}
    if event.value is not None:
        data["value"] = event.value
    if event.reason is not None:
        data["reason"] = event.reason.value
    return data


def states_equal(a: Any, b: Any) -> bool:
    """Field-by-field equality of two backend states; tensors must match exactly."""
    for field in fields(a):
        x, y = getattr(a, field.name), getattr(b, field.name)
        if not (torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y):
            return False
    return True


def tensor_bytes(state: Any) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in (getattr(state, field.name) for field in fields(state))
        if isinstance(value, torch.Tensor)
    )


def check(
    machine: Any,
    backend: Any,
    actions: Iterable[Any],
    *,
    name: str,
    snapshot_at: int,
    trace_limit: int,
    reproduce: Callable[[int], list[str]],
) -> dict[str, Any]:
    """Run one variant in lockstep and return its metrics and evidence.

    Stops at the first mismatch. ``reproduce(horizon)`` returns the command
    that replays the run up to and including the failing action.
    """
    oracle, state = machine.initial_state(), backend.initial_state()
    digest = hashlib.sha256()
    trace: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    snapshot: dict[str, Any] | None = None
    snapshot_resume = "not reached"
    max_residue = 0.0
    steps = work = 0

    started = time.perf_counter()
    for index, action in enumerate(actions):
        digest.update(json.dumps(action_json(action)).encode() + b"\n")
        if index == snapshot_at:
            snapshot = json.loads(json.dumps(machine.snapshot(backend.decode(state))))
            restored = backend.encode(machine.restore(snapshot))
            snapshot_resume = "agrees" if states_equal(restored, state) else "differs"
            state = restored

        expected = machine.step(oracle, action)
        actual = backend.step(state, action)
        residue = backend.residue(actual.next_state)
        try:
            decoded, decode_error = backend.decode(actual.next_state), None
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

        if index < trace_limit or not ok:
            trace.append(
                {
                    "variant": name,
                    "index": index,
                    "action": action_json(action),
                    "event": event_json(actual.event),
                    "work": actual.work,
                    "residue": residue,
                }
            )
        if not ok:
            failure = {
                "variant": name,
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
                "reproduce": reproduce(index + 1),
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
    return {
        "metrics": {
            "variant": name,
            "actions_checked": steps,
            "mismatches": 0 if failure is None else 1,
            "first_mismatch": None if failure is None else failure["action_index"],
            "max_residue": max_residue,
            "classification": classification,
            "snapshot_resume": snapshot_resume,
            "ticks_per_action": work / steps if steps else None,
            "state_tensor_bytes": tensor_bytes(state),
            "end_to_end_seconds": round(seconds, 3),
            "action_sha256": digest.hexdigest(),
        },
        "trace": trace,
        "failure": failure,
        "snapshot": snapshot,
    }


def run_all(
    run_variant: Callable[[Any], dict[str, Any]], variants: Sequence[Any], jobs: int
) -> list[dict[str, Any]]:
    """Run variants serially or in ``jobs`` spawned worker processes."""
    if jobs > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(jobs, mp_context=context) as pool:
            return list(pool.map(run_variant, variants))
    return [run_variant(variant) for variant in variants]


def finish(
    run_dir: Path,
    *,
    experiment: str,
    acceptance: str,
    not_measured: list[str],
    variant_count: int,
    results: list[dict[str, Any]],
    error: str | None,
) -> int:
    """Write evidence and metrics.json, print a summary, return the exit status."""
    if error is None:
        try:
            _write_evidence(run_dir, results)
        except OSError:
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
            "experiment": experiment,
            "status": status,
            "finished_at": records.now(),
            "acceptance": acceptance,
            "totals": {
                "variants": variant_count,
                "variants_completed": len(metrics),
                "actions_checked": sum(m["actions_checked"] for m in metrics),
                "mismatches": sum(m["mismatches"] for m in metrics),
                "max_residue": max((m["max_residue"] for m in metrics), default=None),
            },
            "variants": metrics,
            "not_measured": not_measured,
            "error": error,
        },
    )
    for m in metrics:
        print(
            f"{m['variant']:<44} {m['actions_checked']:>9} actions  "
            f"{m['mismatches']} mismatches  residue {m['max_residue']:.3g}  "
            f"snapshot {m['snapshot_resume']:<11} {m['end_to_end_seconds']:>8.2f}s"
        )
    if error:
        print(error, file=sys.stderr)
    print(f"status: {status}\nrecord: {run_dir}")
    return {"passed": 0, "failed": 1}.get(status, 2)


def _write_evidence(run_dir: Path, results: list[dict[str, Any]]) -> None:
    with (run_dir / "trace.jsonl").open("w") as trace:
        for result in results:
            for line in result["trace"]:
                trace.write(json.dumps(line) + "\n")
    failures = [result["failure"] for result in results if result["failure"]]
    if failures:
        records.write_json(run_dir / "failure.json", failures)
    for result in results:
        if result["snapshot"] is not None:
            (run_dir / "snapshots").mkdir(exist_ok=True)
            name = result["metrics"]["variant"].replace("/", "-")
            records.write_json(
                run_dir / "snapshots" / f"{name}.json", result["snapshot"]
            )


def start(out: str, slug: str) -> tuple[Path, dict[str, Any], Path | None]:
    """Create the run directory; return it, the source info, and the repo root."""
    root = records.repo_root(Path(__file__).resolve().parent)
    source, patch = records.source(root)
    run_dir = records.new_run_dir(Path(out), slug, source.get("revision"))
    if patch:
        (run_dir / "source.patch").write_bytes(patch)
    return run_dir, source, root


def run_header(
    *,
    experiment: str,
    module: str,
    argv: list[str],
    run_dir: Path,
    source: dict[str, Any],
    root: Path | None,
    config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    """The run.json fields every lockstep experiment records."""
    lock = root / "uv.lock" if root else None
    return {
        "schema_version": records.SCHEMA_VERSION,
        "experiment": experiment,
        "run_id": run_dir.name,
        "command": ["python", "-m", module, *argv],
        "started_at": records.now(),
        "source": source,
        "config": config,
        "environment": {
            **records.environment(),
            "torch": torch.__version__,
            "torch_threads_per_variant": 1,
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "device": device,
            "uv_lock_sha256": records.file_sha256(lock) if lock else None,
        },
    }
