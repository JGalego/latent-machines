"""The E05 runner: run records for graph, semiring, and logic machines."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from latent_machines.experiments import e05  # noqa: E402


def run(out: Path, *args: str) -> tuple[int, dict, dict]:
    code = e05.main(["--out", str(out), *args])
    (run_dir,) = out.iterdir()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    record = json.loads((run_dir / "run.json").read_text())
    return code, metrics, record


def test_small_run_records_conformance_and_benchmark(tmp_path: Path) -> None:
    code, metrics, record = run(
        tmp_path, "--horizon", "2", "--dtypes", "float32,python-int",
        "--workloads", "sparse,weighted,bound,forest", "--bench-sizes", "64",
    )  # fmt: skip
    assert code == 0 and metrics["status"] == "passed"
    variants = {v["variant"]: v for v in metrics["variants"]}
    assert "conventional/python-int/datalog/forest" in variants
    bound = variants["dense-tensor/float32/minplus/bound"]
    assert bound["terminal_events"] == {"fault:distance_bound_exceeded": 2}
    assert bound["snapshot_resume"] == "agrees"
    assert metrics["benchmark"]["rows"][0]["results_match"]
    assert record["experiment"] == "E05" and record["config"]["horizon"] == 2


def test_bfloat16_long_distances_fail_and_are_reported(tmp_path: Path) -> None:
    code, metrics, _ = run(
        tmp_path, "--horizon", "1", "--dtypes", "bfloat16", "--machines", "minplus",
        "--workloads", "long", "--no-benchmark",
    )  # fmt: skip
    assert code == 1 and metrics["status"] == "failed"
    (variant,) = metrics["variants"]
    assert variant["classification"] == "mismatch"
