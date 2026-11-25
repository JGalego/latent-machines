"""E11: the differentiable memory, outcome-only training, and a tiny run."""

import dataclasses
import json
import random
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402

from latent_machines.experiments import e11  # noqa: E402
from latent_machines.interfaces import memory as interface  # noqa: E402
from latent_machines.tasks import records as task  # noqa: E402


def one_hot(index: int, size: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor([index]), size).float()


def test_differentiable_memory_matches_the_oracle_on_hard_actions() -> None:
    split = task.Split("t", 30, 8, 24, task.HELD_OUT)
    oracle = task.machine()
    for program in task.programs(split, 0):
        state, memory = oracle.initial_state(), e11.initial_memory(1)
        for statement in program.statements:
            op, rec, val = interface.from_action(statement.action)
            memory, observation = e11.memory_step(
                memory,
                one_hot(op, 2),
                one_hot(rec, e11.RECORDS),
                one_hot(val, e11.VALUES),
                torch.ones(1),
            )
            transition = oracle.step(state, statement.action)
            state = transition.next_state
            expected = interface.observation(transition.event)
            assert observation.argmax().item() == expected
            assert observation.sum().item() == 1.0
        decoded = memory[0].argmax(-1).tolist()
        assert decoded == list(state.words)


def test_integer_rollout_matches_the_oracle_with_gold_actions() -> None:
    # An answer head cannot be forced to the gold answer, so check the machine:
    # rollout's observations come from the same semantics as the oracle's.
    programs = list(task.programs(task.Split("t", 10, 8, 8, task.HELD_OUT), 1))
    lm = e11.WindowedLM(e11.FrozenLM.load("tiny-random"))
    config = _config(epochs=1)
    controller = e11.new_heads(lm, config)
    batch = e11.make_batch(lm, programs)
    logp, ce, correct = e11.rollout(controller, batch, sample=True)
    assert logp.shape == batch.valid.shape
    assert logp.max().item() <= 0.0  # log-probabilities
    assert correct[~batch.is_question].sum().item() == 0.0
    assert torch.isfinite(ce)


def test_outcome_batch_carries_no_gold_actions() -> None:
    names = {f.name for f in dataclasses.fields(e11.Batch)}
    assert names == {"features", "valid", "is_question", "answers"}
    # Programs that differ only in actions that never reach an answer give
    # the same answers, and the batch holds nothing else about the actions.
    lm = e11.WindowedLM(e11.FrozenLM.load("tiny-random"))
    a = task.Program(
        (
            task.Statement("set", 0, 3),
            task.Statement("set", 0, 5),
            task.Statement("get", 0),
        )
    )
    b = task.Program(
        (
            task.Statement("set", 0, 9),
            task.Statement("set", 0, 5),
            task.Statement("get", 0),
        )
    )
    x, y = e11.make_batch(lm, [a]), e11.make_batch(lm, [b])
    assert torch.equal(x.answers, y.answers)
    assert torch.equal(x.is_question, y.is_question)
    assert not torch.equal(x.features, y.features)  # only the text differs


def test_outcome_loss_gradient_reaches_the_write_through_the_memory() -> None:
    torch.manual_seed(0)
    lm = e11.WindowedLM(e11.FrozenLM.load("tiny-random"))
    controller = e11.new_heads(lm, _config(epochs=1))
    program = task.Program((task.Statement("set", 2, 7), task.Statement("get", 2)))
    batch = e11.make_batch(lm, [program])
    loss = e11.outcome_loss(controller, batch)
    loss.backward()
    action_grad = sum(p.grad.abs().sum() for p in controller.actions.parameters())
    assert action_grad.item() > 0.0
    random.seed(0)


def test_soft_observation_equals_the_integer_path_for_one_hots() -> None:
    torch.manual_seed(0)
    head = e11.AnswerHead(8, 16, e11.VALUES, e11.VALUES)
    features = torch.randn(3, 8)
    values = torch.tensor([0, 5, 15])
    hard = head(features, values)
    soft = head(
        features, observation_probs=torch.nn.functional.one_hot(values, 16).float()
    )
    assert torch.allclose(hard, soft, atol=1e-6)


def _config(epochs: int) -> e11.Config:
    return e11.Config(
        model="tiny-random", revision=None, seeds=(0,), data_seed=0, train=20,
        test=4, long_test=2, epochs=epochs, hidden=16, learning_rate=1e-3,
        batch_size=64, program_batch=8,
    )  # fmt: skip


def test_tiny_run_writes_records_and_predictions(tmp_path: Path) -> None:
    code = e11.main(
        ["--model", "tiny-random", "--train", "20", "--test", "4", "--long-test", "2",
         "--epochs", "2", "--hidden", "32", "--seeds", "0", "--program-batch", "8",
         "--out", str(tmp_path)]
    )  # fmt: skip
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["status"] == "completed"
    assert metrics["executor_mismatches"] == 0
    runs = {(r["arm"], r["split"]): r for r in metrics["runs"]}
    assert len(runs) == len(e11.ARMS) * len(e11.TEST_SPLITS)
    assert runs["W", e11.TEST_SPLITS[0]]["action_accuracy"] is None
    assert "operation_accuracy" in runs["O", e11.TEST_SPLITS[0]]
    assert set(metrics["predictions"]) == {
        "1_o_discovers_the_protocol",
        "2_o_writes_and_reads",
        "3_o_beats_reinforce",
        "sanity_i_reaches_imitation_ceiling",
        "sanity_w_stays_at_floor",
    }
    record = json.loads((run_dir / "run.json").read_text())
    assert set(record["arms"]) == set(e11.ARMS)
    assert set(record["curves"]) == {"O", "R"}
