"""The E02 runner end to end on a tiny random model."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from latent_machines.experiments import e02  # noqa: E402


def test_tiny_run_writes_records_and_d_matches_c(tmp_path: Path) -> None:
    code = e02.main(
        ["--model", "tiny-random", "--train", "20", "--test", "4", "--long-test", "2",
         "--epochs", "2", "--seeds", "0", "--out", str(tmp_path)]
    )  # fmt: skip
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["status"] == "completed"
    assert metrics["executor_mismatches"] == 0
    runs = {(r["arm"], r["split"]): r for r in metrics["runs"]}
    assert len(runs) == 4 * 4  # arms x test splits
    for split in ("test_in_distribution", "test_length_64"):
        c, d = runs["C", split], runs["D", split]
        assert c["task_accuracy"] == d["task_accuracy"]
        assert c["action_accuracy"] == d["action_accuracy"]
        assert runs["A", split]["action_accuracy"] is None
        assert runs["B", split]["transcript_tokens"] > 0
        assert runs["C", split]["transcript_tokens"] == 0
    record = json.loads((run_dir / "run.json").read_text())
    assert record["model"]["frozen"] is True
    assert set(record["trained"]) == {"A", "B", "C", "D"}
    assert record["trained"]["A"]["action_examples"] == 0
