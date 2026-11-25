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
    assert len(runs) == 5 * 3  # arms x test splits
    split = "test_in_distribution"
    assert (
        runs["T", split]["feedback_positions"] > runs["L", split]["feedback_positions"]
    )
    assert runs["L", split]["feedback_positions"] > 0
    assert runs["S", split]["feedback_positions"] > 0
    assert runs["H", split]["feedback_positions"] == 0
    assert runs["N", split]["feedback_positions"] == 0
    for snapshot in metrics["snapshots"]:
        assert snapshot["differ"] == 0
        assert snapshot["programs"] > 0
    record = json.loads((run_dir / "run.json").read_text())
    assert record["trained"]["L"]["adaptor_parameters"] > 0
    assert record["trained"]["S"]["adaptor_parameters"] > 0
    assert record["trained"]["T"]["adaptor_parameters"] == 0


def test_default_arms_are_the_recorded_five() -> None:
    assert e03.ARMS == ("T", "L", "S", "H", "N")
    assert e03.parse_args([]).arms == "T,L,S,H,N"
    assert "V" in e03.LATENT_ARMS and "V" not in e03.ARMS


def test_slot_codes_agree_with_slot_code_and_arm_v_has_no_action() -> None:
    import torch

    from latent_machines.interfaces import ticks
    from latent_machines.machines.events import Event, EventKind
    from latent_machines.machines.memory import Read, Write

    codes = [
        ticks.feedback(Read(3), Event(EventKind.OUTPUT, value=9)),
        ticks.feedback(Write(1, 5), Event(EventKind.CONTINUE)),
    ]
    tensor = torch.tensor([c.indices() for c in codes])
    for arm in ("L", "S", "V"):
        batch = e03.slot_codes(arm, tensor)
        single = torch.tensor([e03.slot_code(arm, c).indices() for c in codes])
        assert torch.equal(batch, single)
    assert torch.equal(e03.slot_codes("L", tensor), tensor)  # L is unchanged
    start = torch.tensor(ticks.START.indices())
    assert (e03.slot_codes("S", tensor) == start).all()
    reply = e03.slot_codes("V", tensor)
    assert (reply[:, :3] == start[:3]).all()  # no operation, record, or value
    assert torch.equal(reply[:, 3], tensor[:, 3])  # the reply stays


def test_arm_v_runs_with_s_on_a_tiny_model(tmp_path: Path) -> None:
    code = e03.main(
        ["--model", "tiny-random", "--train", "12", "--test", "6", "--long-test", "1",
         "--epochs", "1", "--seeds", "0", "--snapshot-programs", "3",
         "--arms", "V,S", "--out", str(tmp_path)]
    )  # fmt: skip
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert {r["arm"] for r in metrics["runs"]} == {"V", "S"}
    assert metrics["executor_mismatches"] == 0
    assert all(s["differ"] == 0 for s in metrics["snapshots"])
    record = json.loads((run_dir / "run.json").read_text())
    assert record["trained"]["V"]["adaptor_parameters"] > 0
    assert "V" in record["feedback_reentry"]
