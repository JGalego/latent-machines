"""The E08 runner: verifier conformance and search records."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from latent_machines.experiments import e08  # noqa: E402


def test_small_run_records_conformance_and_search(tmp_path: Path) -> None:
    code = e08.main(
        ["--out", str(tmp_path), "--horizon", "40", "--instances", "1",
         "--budget", "300", "--dtypes", "float32,bfloat16"]
    )  # fmt: skip
    (run_dir,) = tmp_path.iterdir()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert code == 0 and metrics["status"] == "passed"
    variants = metrics["variants"]
    assert len(variants) == 2 * 2 * len(e08.sat_instances.WORKLOADS)
    for m in variants:
        if m["within_declared_limits"]:
            assert m["classification"] == "exact"
        assert m["invalid_accepted"] == 0
    beyond = [m for m in variants if not m["within_declared_limits"]]
    assert {(m["workload"], m["dtype"]) for m in beyond} == {
        ("wide-clauses", "bfloat16")
    }
    checks = metrics["search_checks"]
    assert checks["invalid_solutions"] == checks["energy_decision_mismatches"] == 0
    assert {s["arm"] for s in metrics["search"]} == set(e08.ARMS)
    assert len(metrics["search_instances"]) == len(e08.ARMS) * len(e08.CLASSES)
    record = json.loads((run_dir / "run.json").read_text())
    assert record["experiment"] == "E08" and record["config"]["horizon"] == 40
