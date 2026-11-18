"""The E06 runner: exact cellular runs and numerical SSM runs."""

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends.cellular import ConvLookupCellular  # noqa: E402
from latent_machines.experiments import e06  # noqa: E402


def run(out: Path, *args: str) -> tuple[int, Path]:
    code = e06.main(["--out", str(out), *args])
    (run_dir,) = out.iterdir()
    return code, run_dir


def load(run_dir: Path, name: str) -> Any:
    return json.loads((run_dir / name).read_text())


def test_small_exact_run_passes(tmp_path: Path) -> None:
    code, run_dir = run(
        tmp_path, "--horizon", "40", "--size", "12", "--dtypes", "float32,bfloat16"
    )
    assert code == 0
    metrics = load(run_dir, "metrics.json")
    assert metrics["status"] == "passed"
    variants = metrics["variants"]
    assert len(variants) == 3 * 3 * 4  # backends (int + 2 dtypes) x rules x workloads
    for m in variants:
        assert (m["classification"], m["snapshot_resume"]) == ("exact", "agrees")
        assert m["cells_per_second"] > 0
    names = {m["variant"] for m in variants}
    assert "conv-lookup/bfloat16/B36/S23/gliders-torus" in names
    assert "torch-int/uint8/B2/S/edge-zero" in names
    record = load(run_dir, "run.json")
    assert record["experiment"] == "E06" and record["config"]["horizon"] == 40
    assert run_dir.name.startswith("e06-exact-")


def test_miswired_rule_fails_with_reproducible_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = ConvLookupCellular.__init__

    def miswired(self, machine, dtype=torch.float32, device="cpu") -> None:
        original(self, machine, dtype, device)
        self.birth = self.birth.roll(1)  # birth on the wrong neighbor counts

    monkeypatch.setattr(ConvLookupCellular, "__init__", miswired)
    code, run_dir = run(
        tmp_path,
        *("--horizon", "40", "--size", "12", "--backends", "conv-lookup"),
        *("--dtypes", "float32", "--rules", "B3/S23", "--workloads", "soup50-zero"),
    )
    assert code == 1
    (failure,) = load(run_dir, "failure.json")
    assert "--rules" in failure["reproduce"]
    assert failure["actual"]["decoded_state"] != failure["expected"]["state"]


def test_numerical_run_declares_tolerances_and_classifies(tmp_path: Path) -> None:
    code, run_dir = run(
        tmp_path,
        *("--branch", "numerical", "--horizon", "300", "--chunk", "64"),
        *("--systems", "oscillator", "--signals", "noise"),
    )
    metrics = load(run_dir, "metrics.json")
    variants = {m["variant"]: m for m in metrics["variants"]}
    assert len(variants) == 3 * 3  # backends x dtypes
    for m in variants.values():
        assert m["contract"] == "approximate" and m["snapshot_resume"] == "agrees"
        assert m["tolerance"] == e06.TOLERANCES[m["dtype"]]
    assert variants["ssm-chunked/float64/oscillator/noise"]["classification"] == (
        "within tolerance"
    )
    assert variants["direct/bfloat16/oscillator/noise"]["classification"] == (
        "exceeds tolerance"
    )
    assert code == 1  # bfloat16 exceeds its declared tolerance
    record = load(run_dir, "run.json")
    assert record["config"]["tolerances"] == e06.TOLERANCES
    assert run_dir.name.startswith("e06-numerical-")


def test_rejects_unknown_names() -> None:
    with pytest.raises(SystemExit):
        e06.parse_args(["--rules", "B9/S9"])
    with pytest.raises(SystemExit):
        e06.parse_args(["--branch", "numerical", "--dtypes", "float16"])
