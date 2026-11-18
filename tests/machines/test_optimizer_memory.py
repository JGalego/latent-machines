"""The float64 reference for ``optimizer-memory/v1``."""

import random

import pytest

from latent_machines.machines.events import EventKind
from latent_machines.machines.memory import Idle, Memory, Read, Write
from latent_machines.machines.optimizer_memory import (
    Optimizer,
    OptimizerMemoryReference,
    decode,
)
from latent_machines.tasks.memory_workloads import actions


def column(ref: OptimizerMemoryReference, state, address: int) -> list[float]:
    return ref.predict(state.params, ref.one_hot(address))


def test_plain_sgd_is_delta_replacement_on_every_workload() -> None:
    ref = OptimizerMemoryReference(8, 4, Optimizer())
    machine = Memory(8, 4)
    for workload in ("overwrite", "untouched", "idle", "invalid", "random_mix"):
        state, oracle = ref.initial_state(), machine.initial_state()
        for action in actions(workload, machine, 300, random.Random(workload)):
            expected = machine.step(oracle, action)
            actual = ref.step(state, action)
            assert actual.event == expected.event
            oracle, state = expected.next_state, actual.next_state
            words = tuple(decode(column(ref, state, a)) for a in range(8))
            assert words == oracle.words
            assert all(x in (0.0, 1.0) for x in state.params["W"])


def test_stale_chunk_gradients_sum_repeated_writes() -> None:
    ref = OptimizerMemoryReference(2, 2, Optimizer(chunk=4))
    state = ref.initial_state()
    for action in (Write(0, 1), Write(0, 1)):
        state = ref.step(state, action).next_state
    # Both corrections are computed from the zero chunk-start state.
    assert column(ref, state, 0) == [2.0, 0.0]


def test_decay_forgets_during_idle() -> None:
    ref = OptimizerMemoryReference(1, 1, Optimizer(decay=1e-3))
    state = ref.step(ref.initial_state(), Write(0, 1)).next_state
    for _ in range(691):  # the read is the 692nd decay step
        state = ref.step(state, Idle()).next_state
    assert ref.step(state, Read(0)).event.value == 1
    for _ in range(2):
        state = ref.step(state, Idle()).next_state
    assert ref.step(state, Read(0)).event.value == 0


def test_momentum_keeps_moving_without_a_sample() -> None:
    ref = OptimizerMemoryReference(1, 1, Optimizer(momentum=0.5))
    state = ref.step(ref.initial_state(), Write(0, 1)).next_state
    assert state.params["W"] == [1.0]
    state = ref.step(state, Idle()).next_state
    assert state.params["W"] == [1.5]


def test_normalized_gradient_has_unit_norm() -> None:
    ref = OptimizerMemoryReference(1, 2, Optimizer(normalize=True))
    state = ref.step(ref.initial_state(), Write(0, 3)).next_state
    assert state.params["W"] == pytest.approx([2**-0.5, 2**-0.5])
    # A second unit step overshoots the target.
    again = ref.step(state, Write(0, 3)).next_state
    assert again.params["W"] == pytest.approx([2**0.5, 2**0.5])
    # A write that changes nothing has zero gradient and stays put.
    one = ref.step(ref.initial_state(), Write(0, 1)).next_state
    assert one.params["W"] == [1.0, 0.0]
    assert ref.step(one, Write(0, 1)).next_state.params["W"] == [1.0, 0.0]


def test_mlp_direction_matches_finite_differences() -> None:
    ref = OptimizerMemoryReference(3, 2, Optimizer(model="mlp", hidden=5, seed=1))
    params = ref.initial_state().params
    key, target = [0.3, -0.7, 0.5], [1.0, 0.0]

    def loss(p) -> float:
        out = ref.predict(p, key)
        return 0.5 * sum((o - y) ** 2 for o, y in zip(out, target, strict=True))

    direction = ref.direction(params, key, target)
    h = 1e-6
    for name, values in params.items():
        for i in range(len(values)):
            up = {**params, name: [*values[:i], values[i] + h, *values[i + 1 :]]}
            down = {**params, name: [*values[:i], values[i] - h, *values[i + 1 :]]}
            numeric = -(loss(up) - loss(down)) / (2 * h)
            assert direction[name][i] == pytest.approx(numeric, abs=1e-6)


def test_faults_change_nothing() -> None:
    ref = OptimizerMemoryReference(2, 2, Optimizer(momentum=0.5, chunk=3))
    state = ref.step(ref.initial_state(), Write(0, 1)).next_state
    for bad in (Read(2), Write(-1, 0), Write(0, 4)):
        transition = ref.step(state, bad)
        assert transition.event.kind is EventKind.FAULT
        assert transition.next_state == state
