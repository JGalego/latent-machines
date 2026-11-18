"""The tensor backend for ``optimizer-memory/v1`` against its reference."""

import json
import random

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends.optimizer_memory import OptimizerMemory  # noqa: E402
from latent_machines.experiments.e07 import SETTINGS  # noqa: E402
from latent_machines.machines.memory import Memory, Write  # noqa: E402
from latent_machines.machines.optimizer_memory import (  # noqa: E402
    OptimizerMemoryReference,
)
from latent_machines.tasks.memory_workloads import actions  # noqa: E402


@pytest.mark.parametrize("setting", sorted(SETTINGS))
def test_float64_backend_tracks_the_reference(setting: str) -> None:
    optimizer = SETTINGS[setting]
    ref = OptimizerMemoryReference(6, 3, optimizer)
    backend = OptimizerMemory(6, 3, optimizer, torch.float64)
    r, t = ref.initial_state(), backend.initial_state()
    machine = Memory(6, 3)
    for action in actions("random_mix", machine, 150, random.Random(setting)):
        expected, actual = ref.step(r, action), backend.step(t, action)
        assert actual.event == expected.event
        r, t = expected.next_state, actual.next_state
        for name, values in r.params.items():
            got = t.params[name].reshape(-1).tolist()
            assert got == pytest.approx(values, abs=1e-12)


def test_planted_learning_rate_error_is_visible() -> None:
    optimizer = SETTINGS["sgd"]
    ref = OptimizerMemoryReference(4, 2, optimizer)
    wrong = OptimizerMemory(4, 2, SETTINGS["lr-0.9"], torch.float64)
    r = ref.step(ref.initial_state(), Write(1, 3)).next_state
    t = wrong.step(wrong.initial_state(), Write(1, 3)).next_state
    assert t.params["W"].reshape(-1).tolist() != pytest.approx(r.params["W"])


@pytest.mark.parametrize("setting", ["momentum-0.5", "chunk-4", "mlp"])
def test_full_snapshot_restores_exactly(setting: str) -> None:
    backend = OptimizerMemory(4, 3, SETTINGS[setting], torch.float32)
    state = backend.initial_state()
    for action in (Write(0, 5), Write(1, 2), Write(0, 7)):
        state = backend.step(state, action).next_state
    restored = backend.restore(json.loads(json.dumps(backend.snapshot(state))))
    assert restored.steps == state.steps
    for name in state.params:
        assert torch.equal(restored.params[name], state.params[name])
        assert torch.equal(restored.momentum[name], state.momentum[name])
    if state.chunk_start is not None:
        assert restored.chunk_start is not None
        for name in state.chunk_start:
            assert torch.equal(restored.chunk_start[name], state.chunk_start[name])


def test_canonical_words_drop_optimizer_buffers() -> None:
    backend = OptimizerMemory(2, 2, SETTINGS["momentum-0.5"], torch.float32)
    state = backend.from_words((3, 1), steps=5)
    assert state.params["W"].tolist() == [[1.0, 1.0], [1.0, 0.0]]
    assert not state.momentum["W"].any()
    with pytest.raises(ValueError):
        OptimizerMemory(2, 2, SETTINGS["mlp"]).from_words((0, 0), 0)


def test_bfloat16_cannot_resolve_the_decay_factor() -> None:
    backend = OptimizerMemory(1, 1, SETTINGS["decay-0.001"], torch.bfloat16)
    state = backend.step(backend.initial_state(), Write(0, 1)).next_state
    for _ in range(1000):
        state = backend.update(state, None, None)
    assert state.params["W"].item() == 1.0
