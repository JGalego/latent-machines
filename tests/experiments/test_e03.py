"""The E03 runner end to end on a tiny random model."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from latent_machines.experiments import e03  # noqa: E402


def test_tiny_run_writes_records_and_snapshots_resume(tmp_path: Path) -> None:
    code = e03.main(
        ["--model", "tiny-random", "--train", "12", "--test", "6", "--long-test", "1",
         "--epochs", "1", "--seeds", "0", "--snapshot-programs", "3",
         "--out", str(tmp_path)]
    )  # fmt: skip
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["status"] == "completed"
    assert metrics["executor_mismatches"] == 0
    runs = {(r["arm"], r["split"]): r for r in metrics["runs"]}
    assert len(runs) == 4 * 3  # arms x test splits
    split = "test_in_distribution"
    assert (
        runs["T", split]["feedback_positions"] > runs["L", split]["feedback_positions"]
    )
    assert runs["L", split]["feedback_positions"] > 0
    assert runs["H", split]["feedback_positions"] == 0
    assert runs["N", split]["feedback_positions"] == 0
    for snapshot in metrics["snapshots"]:
        assert snapshot["differ"] == 0
        assert snapshot["programs"] > 0
    record = json.loads((run_dir / "run.json").read_text())
    assert record["trained"]["L"]["adaptor_parameters"] > 0
    assert record["trained"]["T"]["adaptor_parameters"] == 0
