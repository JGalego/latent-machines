"""The E09 runner end to end on a tiny random model, and its baseline."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from latent_machines.experiments import e09, e09_folds  # noqa: E402
from latent_machines.tasks import records as task  # noqa: E402


def test_baseline_is_exact_without_errors_and_falls_with_them() -> None:
    split = task.Split("t", 20, 32, 32, task.HELD_OUT)
    programs = list(task.programs(split, 0))
    assert e09.replay_accuracy(programs, 0.0, seed=0) == 1.0
    noisy = e09.replay_accuracy(programs, 0.2, seed=0)
    assert 0.0 < noisy < 1.0
    assert e09.replay_accuracy(programs, 0.2, seed=0) == noisy  # reproducible


def test_bound_matches_the_e02_report() -> None:
    assert round(e09.bound(1592), 3) == 0.035
    assert round(e09.bound(3216), 3) == 0.025


def test_tiny_run_writes_records_and_predictions(tmp_path: Path) -> None:
    code = e09.main(
        ["--model", "tiny-random", "--train", "20", "--test", "4", "--long-test", "2",
         "--epochs", "2", "--seeds", "0", "--out", str(tmp_path)]
    )  # fmt: skip
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["status"] == "completed"
    assert metrics["executor_mismatches"] == 0
    runs = {(r["arm"], r["split"]): r for r in metrics["runs"]}
    assert len(runs) == len(e09.ARMS) * len(e09.TEST_SPLITS)
    for split in e09.TEST_SPLITS:
        s, d = runs["S", split], runs["S+delta", split]
        assert s["task_accuracy"] == d["task_accuracy"]
        assert s["action_accuracy"] == d["action_accuracy"]
        assert runs["W", split]["action_accuracy"] is None
    # S reads the same features wherever a statement sits, so its action
    # accuracy is the same on every split up to the programs drawn.
    positions = metrics["context_positions"]
    assert (
        positions["S"]["test_length_256"]["max_positions"]
        == positions["S"]["test_in_distribution"]["max_positions"]
    )
    assert (
        positions["C"]["test_length_256"]["max_positions"]
        > positions["C"]["test_length_32"]["max_positions"]
    )
    assert set(metrics["predictions"]) == {
        "1_s_beats_c_beyond_bound",
        "2_s_tracks_compounding_baseline",
        "3_w_stays_at_floor",
        "sanity_s_action_accuracy_constant",
    }
    record = json.loads((run_dir / "run.json").read_text())
    assert set(record["arms"]) == set(e09.ARMS)


def test_held_out_run_measures_unseen_statements(tmp_path: Path) -> None:
    code = e09.main(
        ["--model", "tiny-random", "--train", "20", "--test", "4", "--epochs", "2",
         "--seeds", "0", "--held-out", "--out", str(tmp_path)]
    )  # fmt: skip
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    assert "e09-heldout" in run_dir.name
    metrics = json.loads((run_dir / "metrics.json").read_text())
    splits = {r["split"] for r in metrics["runs"]}
    assert splits == set(e09.HELD_OUT_SPLITS)
    (statements,) = metrics["statement_accuracy"]
    assert 0 <= statements["unseen"] <= 1 and 0 <= statements["seen"] <= 1
    # The 8 held-out texts and the 128 others cover every statement once.
    predictions = metrics["predictions"]
    assert set(predictions) == {
        "1_s_reads_unseen_statements",
        "2_s_beats_c_on_held_out_pairs",
        "3_pretraining_adds_over_random",
        "sanity_s_seen_statements",
    }
    runs = {(r["arm"], r["split"]): r for r in metrics["runs"]}
    assert runs["S", "test_held_out_pairs"]["questions"] > 0


def test_folds_are_disjoint_and_cover_every_set_statement() -> None:
    pairs = [pair for fold in e09.FOLDS.values() for pair in fold]
    assert len(pairs) == len(set(pairs)) == 32
    assert {r for r, _ in pairs} == set(range(len(task.NAMES)))
    assert e09.FOLDS[0] == task.HELD_OUT


def test_fold_run_holds_out_its_own_statements(tmp_path: Path) -> None:
    code = e09.main(
        ["--model", "tiny-random", "--train", "20", "--test", "4", "--epochs", "2",
         "--seeds", "0", "--held-out", "--fold", "1", "--arms", "S,S+delta,W",
         "--out", str(tmp_path)]
    )  # fmt: skip
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    assert "fold1" in run_dir.name
    run = json.loads((run_dir / "run.json").read_text())
    assert run["task"]["fold"] == 1
    assert run["task"]["held_out_pairs"] == [list(p) for p in sorted(e09.FOLDS[1])]
    train = run["task"]["splits"]["train"]
    assert train["held_out"] == [list(p) for p in sorted(e09.FOLDS[1])]
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert {r["arm"] for r in metrics["runs"]} == {"S", "S+delta", "W"}
    (statements,) = metrics["statement_accuracy"]
    assert len(statements["failed_unseen"]) <= 8
    assert all(f["statement"].startswith("Set ") for f in statements["failed_unseen"])
    # The unseen statements are the ones with records a, b and values 0-3.
    assert {
        e09_folds.per_statement([statements], 1)[s]
        for s in ("Set a to 0.", "Set b to 3.")
    } <= {0, 1}


def test_fold_aggregation_evaluates_the_three_predictions() -> None:
    def rows(unseen: float, seen: float = 1.0) -> list[dict]:
        return [{"unseen": unseen, "seen": seen, "failed_unseen": []}] * 3

    smollm = {fold: rows(0.9) for fold in e09.FOLDS}
    random = {fold: rows(0.2) for fold in e09.FOLDS}
    result = e09_folds.evaluate(smollm, random)
    assert result["1_s_reads_unseen_statements"]["met"]
    assert result["2_no_fold_below_half"]["met"]
    assert result["3_pretraining_adds_over_random"]["met"]
    assert result["sanity_seen_statements"]["ok"]
    smollm[2] = rows(0.4)
    result = e09_folds.evaluate(smollm, random)
    assert not result["2_no_fold_below_half"]["met"]
    assert not result["1_s_reads_unseen_statements"]["met"]
    assert result["1_s_reads_unseen_statements"]["inconclusive"]
