"""E10: the in-layer memory, the layer hook, and the runner on a tiny random model."""

import json
import random
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch  # noqa: E402

from latent_machines.controllers import layer_machine as lmach  # noqa: E402
from latent_machines.controllers.frozen_lm import FrozenLM  # noqa: E402
from latent_machines.experiments import e10  # noqa: E402
from latent_machines.machines.memory import Read  # noqa: E402
from latent_machines.tasks import copies as task  # noqa: E402
from latent_machines.tasks import records as record_task  # noqa: E402


def test_in_layer_state_matches_the_oracle_on_hard_actions() -> None:
    rng = random.Random(0)
    oracle = task.machine()
    expected = oracle.initial_state()
    state = lmach.initial_state()
    for _ in range(400):
        macro = rng.choice([lmach.SET, lmach.GET, lmach.COPY])
        record, value = rng.randrange(8), rng.randrange(16)
        source = rng.choice([r for r in range(8) if r != record])
        stored = lmach.decode(state)[0]
        for action in e10.macro_actions(macro, record, source, value, stored):
            transition = oracle.step(expected, action)
            expected = transition.next_state
            if isinstance(action, Read) and macro == lmach.GET:
                read_value = transition.event.value
        t = torch.tensor
        state, seen = lmach.apply(
            state, t([macro]), t([record]), t([source]), t([value])
        )
        assert lmach.decode(state)[0] == list(expected.words)
        if macro == lmach.GET:
            assert seen.argmax().item() == read_value
            assert seen.sum().item() == 1
        else:
            assert seen.abs().sum().item() == 0


def test_copy_reads_before_it_writes_and_leaves_other_records_alone() -> None:
    t = torch.tensor
    state = lmach.initial_state()
    state, _ = lmach.apply(state, t([lmach.SET]), t([2]), t([0]), t([9]))
    state, _ = lmach.apply(state, t([lmach.SET]), t([5]), t([0]), t([4]))
    state, _ = lmach.apply(state, t([lmach.COPY]), t([3]), t([2]), t([0]))
    assert lmach.decode(state)[0] == [0, 0, 9, 9, 0, 4, 0, 0]


def test_straight_through_is_one_hot_forward_and_has_gradient() -> None:
    logits = torch.tensor([[0.1, 2.0, -1.0]], requires_grad=True)
    y = lmach.st_onehot(logits)
    assert y.tolist() == [[0.0, 1.0, 0.0]]
    (y * torch.tensor([[1.0, 2.0, 3.0]])).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


@pytest.fixture(scope="module")
def lm() -> FrozenLM:
    return FrozenLM.load("tiny-random")


def test_hook_is_removed_after_use_and_after_an_error(lm: FrozenLM) -> None:
    layer = lm.model.layers[0]
    ids = torch.tensor([lmach.window_ids(lm, record_task.PREAMBLE, " What is a?")])
    with lmach.LayerHook(lm, 1) as hook:
        assert len(layer._forward_hooks) == 1
        lm.model(input_ids=ids)
    assert len(layer._forward_hooks) == 0
    assert hook.calls == 1
    with pytest.raises(RuntimeError), lmach.LayerHook(lm, 1):
        raise RuntimeError("boom")
    assert len(layer._forward_hooks) == 0


def test_injection_changes_the_top_feature_and_not_the_layer_k_feature(
    lm: FrozenLM,
) -> None:
    ids = torch.tensor([lmach.window_ids(lm, record_task.PREAMBLE, " What is a?")])
    base = lmach.inject_forward(lm, 1, ids, torch.zeros(1, lm.hidden_size))
    pushed = lmach.inject_forward(lm, 1, ids, torch.ones(1, lm.hidden_size))
    assert not torch.allclose(base, pushed)
    cache = lmach.WindowCache(lm, 1, record_task.PREAMBLE)
    cache.add([" What is a?"])
    assert torch.allclose(cache.top[" What is a?"], base[0], atol=1e-5)


@pytest.mark.parametrize("residual", [True, False])
def test_live_pass_matches_the_cached_path_and_a_copy_takes_one_pass(
    lm: FrozenLM, residual: bool
) -> None:
    torch.manual_seed(0)
    machine = lmach.LayerMachine(lm.hidden_size, 16, reject=False, residual=residual)
    program = [" Set a to 7.", " Copy a to c.", " What is c?"]
    cache = lmach.WindowCache(lm, 1, record_task.PREAMBLE)
    state, cached_state = lmach.initial_state(), lmach.initial_state()
    for text in program:
        live = lmach.live_statement(lm, 1, machine, record_task.PREAMBLE, text, state)
        assert live.passes == 1  # one model pass, whatever the macro
        state = live.state
        mid = cache.mid_features([text])
        choice = [int(x) for x in machine.choose(mid)]
        cached_state, seen = lmach.apply(
            cached_state, *(torch.tensor([x]) for x in choice)
        )
        assert (live.macro, live.record, live.source, live.value) == tuple(choice)
        assert torch.equal(state, cached_state)
        assert torch.equal(live.read, seen)
    # The same machine run live on a copy statement alone: still one pass.
    start = lmach.initial_state()
    lone = lmach.live_statement(
        lm, 1, machine, record_task.PREAMBLE, " Copy a to b.", start
    )
    assert lone.passes == 1


def test_invalid_statements_are_disjoint_from_valid_ones_and_from_each_other() -> None:
    groups = e10.invalid_sets(0)
    valid = set(e10.valid_texts())
    seen: set[str] = set()
    for texts in groups.values():
        assert not valid & set(texts)
        assert not seen & set(texts)
        assert len(texts) == len(set(texts))
        seen |= set(texts)
    assert len(groups["trained_form_train"]) + len(groups["trained_form_test"]) == 227
    assert all(" l" in t or "l." in t or "l?" in t for t in groups["unseen_record"])
    for text in groups["unseen_values"]:
        assert 32 <= int(text.split()[-1].rstrip(".")) <= 47


def test_programs_respect_the_held_out_pairs() -> None:
    train = e10.draw("train", 50, 2, 8, 0, "exclude")
    held = e10.draw("held", 20, 2, 8, 0, "include")
    long = e10.draw("long", 5, 256, 256, 0, "exclude")
    assert not any(e10.uses_held_out(p) for p in train + long)
    assert all(e10.uses_held_out(p) for p in held)
    assert len(long[0].statements) == 256


def test_tiny_run_writes_records_and_predictions(tmp_path: Path) -> None:
    code = e10.main(
        ["--model", "tiny-random", "--train", "40", "--test", "6", "--long-test", "2",
         "--epochs", "2", "--seeds", "0", "--hidden", "16", "--out", str(tmp_path)]
    )  # fmt: skip
    assert code == 0
    (run_dir,) = tmp_path.iterdir()
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["status"] == "completed"
    assert metrics["executor_mismatches"] == 0
    assert metrics["layer_mismatches"] == 0
    runs = {(r["arm"], r["split"]): r for r in metrics["runs"]}
    assert len(runs) == len(e10.ARMS) * len(e10.TEST_SPLITS)
    assert runs["O", "test_in_distribution"]["layer_mismatches"] is None
    assert (
        runs["M", "test_length_256"]["decisions"]
        == (runs["M", "test_length_256"]["statements"])
    )
    assert set(metrics["predictions"]) == {
        "1_m_not_worse_than_o",
        "2_m_and_mh_within_bound",
        "3_in_layer_state_matches_oracle",
        "4_reject_learned_for_trained_forms",
        "5_reject_does_not_generalize",
        "sanity",
    }
    run = json.loads((run_dir / "run.json").read_text())
    assert run["model"]["machine_layer"] == 1
    assert run["trained"]["M"]["trainable_parameters"] > 0
