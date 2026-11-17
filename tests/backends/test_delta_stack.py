"""Stepwise conformance of the delta-rule stack backend to stack/v1."""

import json
import random

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends.delta_stack import (  # noqa: E402
    DeltaStack,
    DeltaStackState,
)
from latent_machines.machines.stack import (  # noqa: E402
    Pop,
    Push,
    Stack,
    StackAction,
    StackState,
)

CAPACITY = 6
WORD_BITS = 8
MAX_WORD = 2**WORD_BITS - 1
DTYPES = [torch.float32, torch.float64, torch.bfloat16, torch.float16]


@pytest.fixture
def stack() -> Stack:
    return Stack(capacity=CAPACITY, word_bits=WORD_BITS)


@pytest.fixture(params=DTYPES, ids=str)
def backend(request: pytest.FixtureRequest, stack: Stack) -> DeltaStack:
    return DeltaStack(stack, request.param)


def run_lockstep(
    stack: Stack, backend: DeltaStack, actions: list[StackAction]
) -> DeltaStackState:
    """Step both together; slots above the top must hold zeros."""
    expected_state, state = stack.initial_state(), backend.initial_state()
    for index, action in enumerate(actions):
        expected = stack.step(expected_state, action)
        actual = backend.step(state, action)
        where = f"first mismatch at action {index}: {action!r}"
        assert actual.event == expected.event, where
        assert actual.work == expected.work, where
        assert backend.decode(actual.next_state) == expected.next_state, where
        assert backend.residue(actual.next_state) == 0.0, where
        depth = len(expected.next_state.items)
        freed = actual.next_state.memory[:, depth:]
        assert not freed.count_nonzero(), where
        expected_state, state = expected.next_state, actual.next_state
    return state


def random_action(rng: random.Random) -> StackAction:
    if rng.random() < 0.55:
        return Push(rng.randrange(-1, MAX_WORD + 2))
    return Pop()


@pytest.mark.parametrize("seed", range(3))
def test_matches_oracle_on_random_actions(
    stack: Stack, backend: DeltaStack, seed: int
) -> None:
    rng = random.Random(seed)
    run_lockstep(stack, backend, [random_action(rng) for _ in range(400)])


def test_fill_overflow_drain_underflow(stack: Stack, backend: DeltaStack) -> None:
    fill: list[StackAction] = [Push(MAX_WORD - i) for i in range(CAPACITY)]
    drain: list[StackAction] = [Pop()] * CAPACITY
    run_lockstep(stack, backend, [*fill, Push(1), *drain, Pop(), *fill, *drain])


def test_every_value_at_every_depth(stack: Stack, backend: DeltaStack) -> None:
    actions: list[StackAction] = []
    for value in range(MAX_WORD + 1):
        depth = value % CAPACITY
        actions += [Push(0)] * depth + [Push(value), Pop()] + [Pop()] * depth
    run_lockstep(stack, backend, actions)


def test_faults_leave_tensors_untouched(stack: Stack, backend: DeltaStack) -> None:
    full = run_lockstep(stack, backend, [Push(v) for v in range(CAPACITY)])
    for action in (Push(1), Push(MAX_WORD + 1)):
        assert backend.step(full, action).next_state is full
    empty = backend.initial_state()
    assert backend.step(empty, Pop()).next_state is empty


def test_snapshot_resume_matches_uninterrupted_run(
    stack: Stack, backend: DeltaStack
) -> None:
    rng = random.Random(7)
    actions = [random_action(rng) for _ in range(200)]
    full = run_lockstep(stack, backend, actions)
    middle = run_lockstep(stack, backend, actions[:100])
    data = json.loads(json.dumps(stack.snapshot(backend.decode(middle))))
    state = backend.encode(stack.restore(data))
    assert torch.equal(state.memory, middle.memory)
    for action in actions[100:]:
        state = backend.step(state, action).next_state
    assert torch.equal(state.memory, full.memory)
    assert torch.equal(state.pointer, full.pointer)


def test_decode_rejects_corrupt_pointers(stack: Stack) -> None:
    backend = DeltaStack(stack)
    state = backend.initial_state()
    for pointer in (torch.zeros(CAPACITY + 1), torch.ones(CAPACITY + 1)):
        with pytest.raises(ValueError):
            backend.decode(DeltaStackState(state.memory, pointer))


def test_encode_rejects_invalid_states(stack: Stack) -> None:
    backend = DeltaStack(stack)
    with pytest.raises(ValueError):
        backend.encode(StackState((0,) * (CAPACITY + 1)))
    with pytest.raises(ValueError):
        backend.encode(StackState((MAX_WORD + 1,)))
