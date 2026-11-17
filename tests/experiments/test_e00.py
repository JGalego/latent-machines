"""The E00 runner: run records, failure evidence, and parallel execution."""

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends import delta  # noqa: E402
from latent_machines.backends.delta import DeltaMemory  # noqa: E402
from latent_machines.experiments import e00  # noqa: E402


def run(out: Path, *args: str) -> tuple[int, Path]:
    code = e00.main(["--out", str(out), "--horizon", "64", *args])
    (run_dir,) = out.iterdir()
    return code, run_dir


def load(run_dir: Path, name: str) -> Any:
    return json.loads((run_dir / name).read_text())


def test_small_run_passes_and_writes_a_complete_record(tmp_path: Path) -> None:
    code, run_dir = run(tmp_path, "--dtypes", "float32,bfloat16")
    assert code == 0

    metrics = load(run_dir, "metrics.json")
    assert metrics["status"] == "passed"
    variants = 2 * len(e00.WORKLOADS)
    assert metrics["totals"] == {
        "variants": variants,
        "variants_completed": variants,
        "actions_checked": variants * 64,
        "mismatches": 0,
        "max_residue": 0.0,
    }
    for m in metrics["variants"]:
        assert (m["classification"], m["snapshot_resume"]) == ("exact", "agrees")
        assert m["ticks_per_action"] == 1

    record = load(run_dir, "run.json")
    assert record["experiment"] == "E00"
    assert record["command"][:3] == ["python", "-m", "latent_machines.experiments.e00"]
    assert record["source"]["revision"]
    trace = (run_dir / "trace.jsonl").read_text().splitlines()
    assert len(trace) == variants * 64
    assert len(list((run_dir / "snapshots").iterdir())) == variants
    assert not (run_dir / "failure.json").exists()


def test_mismatch_writes_first_failure_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bits = DeltaMemory._bits
    monkeypatch.setattr(
        DeltaMemory, "_bits", lambda self, word: bits(self, word).flip(0)
    )
    code, run_dir = run(tmp_path, "--dtypes", "float32", "--workloads", "overwrite")
    assert code == 1

    (m,) = load(run_dir, "metrics.json")["variants"]
    assert (m["classification"], m["mismatches"]) == ("mismatch", 1)
    (failure,) = load(run_dir, "failure.json")
    assert failure["action_index"] == m["first_mismatch"]
    assert failure["actual"]["decoded_state"] != failure["expected"]["state"]
    assert "--horizon" in failure["reproduce"]


def test_residue_without_mismatch_is_reported_not_hidden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def kernel_style_norm(vector: torch.Tensor) -> torch.Tensor:
        return vector * torch.rsqrt((vector * vector).sum() + 1e-6)

    monkeypatch.setattr(delta, "_normalize", kernel_style_norm)
    code, run_dir = run(tmp_path, "--dtypes", "float32", "--workloads", "overwrite")

    (m,) = load(run_dir, "metrics.json")["variants"]
    assert m["mismatches"] == 0
    assert m["classification"] == "conforms with residue"
    assert m["max_residue"] > 0
    # A canonical snapshot drops the residue, so resuming is not the same run.
    assert m["snapshot_resume"] == "differs"
    assert code == 1


def test_parallel_workers_match_serial_run(tmp_path: Path) -> None:
    args = ("--dtypes", "float32", "--workloads", "overwrite,idle")
    _, serial = run(tmp_path / "serial", *args)
    _, parallel = run(tmp_path / "parallel", *args, "--jobs", "2")

    def comparable(run_dir: Path) -> list[dict[str, Any]]:
        variants = load(run_dir, "metrics.json")["variants"]
        return [
            {k: v for k, v in m.items() if k != "end_to_end_seconds"} for m in variants
        ]

    assert comparable(serial) == comparable(parallel)


def test_rejects_unknown_workload() -> None:
    with pytest.raises(SystemExit):
        e00.parse_args(["--workloads", "nope"])
