"""The E04 runner: conformance, probes, and timing records."""

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from latent_machines.experiments import e04  # noqa: E402

SMALL = [
    "--programs", "2", "--inputs", "6", "--timing-programs", "1",
    "--timing-batch", "8", "--timing-repeats", "2",
]  # fmt: skip


def run(out: Path, *args: str) -> tuple[int, Path]:
    code = e04.main(["--out", str(out), *SMALL, *args])
    (run_dir,) = out.iterdir()
    return code, run_dir


def load(run_dir: Path, name: str) -> Any:
    return json.loads((run_dir / name).read_text())


def test_wide_formats_pass_on_every_workload(tmp_path: Path) -> None:
    code, run_dir = run(
        tmp_path,
        "--backends", "python-int,torch-int64,compiled-graph,expert-dispatch",
        "--dtypes", "int,int64,float32",
    )  # fmt: skip
    assert code == 0
    metrics = load(run_dir, "metrics.json")
    assert metrics["status"] == "passed"
    variants = metrics["variants"]
    assert len(variants) == 4 * 3
    for m in variants:
        assert m["classification"] == "exact" and m["mismatches"] == 0
        assert m["actions_checked"] == 12
        assert m["snapshot_resume"] in ("agrees", "not applicable", "not reached")
        assert m["throughput_evaluations_per_second"] > 0
    graph = next(m for m in variants if m["backend"] == "compiled-graph")
    assert "amortize_evaluations_vs_python_int_batch1" in graph
    record = load(run_dir, "run.json")
    assert record["experiment"] == "E04"
    assert record["config"]["horizon"] == 12


def test_probe_mismatches_are_reported_but_not_accepted(tmp_path: Path) -> None:
    code, run_dir = run(
        tmp_path,
        "--backends", "compiled-graph", "--dtypes", "bfloat16",
        "--workloads", "dot-product",
    )  # fmt: skip
    metrics = load(run_dir, "metrics.json")
    (m,) = metrics["variants"]
    assert m["precision_probe"] and m["classification"] == "mismatch"
    assert (code, metrics["status"]) == (0, "passed")
    assert metrics["totals"]["mismatches_outside_probes"] == 0
    failure = load(run_dir, "failure.json")[0]
    assert failure["variant"] == "compiled-graph/bfloat16/dot-product"
