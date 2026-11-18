"""The E07 runner: own-reference checks, RAM findings, and the run record."""

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends import optimizer_memory  # noqa: E402
from latent_machines.experiments import e07  # noqa: E402


def run(out: Path, *args: str) -> tuple[int, Path]:
    code = e07.main(
        ["--out", str(out), "--horizon", "48", "--dtypes", "float32", *args]
    )
    (run_dir,) = out.iterdir()
    return code, run_dir


def load(run_dir: Path, name: str) -> Any:
    return json.loads((run_dir / name).read_text())


def by_name(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {v["variant"]: v for v in metrics["variants"]}


def test_small_run_passes_and_records_ram_findings(tmp_path: Path) -> None:
    code, run_dir = run(
        tmp_path,
        "--settings", "sgd,chunk-4,mlp",
        "--workloads", "overwrite,idle,regression",
    )  # fmt: skip
    assert code == 0
    metrics = load(run_dir, "metrics.json")
    assert metrics["status"] == "passed"
    assert metrics["totals"]["variants"] == 9
    variants = by_name(metrics)
    for v in variants.values():
        assert v["classification"] == "within tolerance"
        assert v["snapshot_resume"] == "agrees"
    assert variants["sgd/float32/overwrite"]["ram_conformance"] == "exact"
    assert variants["sgd/float32/overwrite"]["canonical_snapshot_resume"] == "agrees"
    chunk = variants["chunk-4/float32/overwrite"]
    assert chunk["ram_conformance"] == "deviates"
    assert chunk["first_ram_deviation"] == 2  # the second write in the first chunk
    assert variants["mlp/float32/idle"]["canonical_snapshot_resume"] == "not applicable"
    regression = variants["sgd/float32/regression"]
    assert regression["ram_conformance"] == "not applicable"
    assert regression["online_loss_last"] < regression["online_loss_first"]

    record = load(run_dir, "run.json")
    assert record["experiment"] == "E07"
    assert record["config"]["horizon"] == 48
    assert set(record["settings"]) == {"sgd", "chunk-4", "mlp"}
    assert len(list((run_dir / "snapshots").iterdir())) == 9
    assert (run_dir / "ram_deviations.json").exists()


def test_a_backend_bug_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = optimizer_memory.OptimizerMemory.direction

    def off_by_a_bit(self, params, key, target):  # type: ignore[no-untyped-def]
        return {n: 1.01 * g for n, g in real(self, params, key, target).items()}

    monkeypatch.setattr(optimizer_memory.OptimizerMemory, "direction", off_by_a_bit)
    code, run_dir = run(tmp_path, "--settings", "sgd", "--workloads", "overwrite")
    assert code == 1
    metrics = load(run_dir, "metrics.json")
    assert metrics["status"] == "failed"
    (variant,) = metrics["variants"]
    assert variant["classification"] == "exceeds tolerance"
    assert variant["first_mismatch"] == 0


def test_unknown_setting_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        e07.main(["--out", str(tmp_path), "--settings", "adam"])
