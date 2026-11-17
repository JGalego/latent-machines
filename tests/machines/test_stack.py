"""Conformance tests for the stack/v1 bounded stack oracle."""

import json
import random

import pytest

from latent_machines.machines.events import Event, EventKind
from latent_machines.machines.stack import (
    Pop,
    Push,
    Stack,
    StackAction,
    StackFault,
    StackState,
)

CAPACITY = 4
WORD_BITS = 8
MAX_WORD = 2**WORD_BITS - 1


@pytest.fixture
def stack() -> Stack:
    return Stack(capacity=CAPACITY, word_bits=WORD_BITS)


def fault(reason: StackFault) -> Event:
    return Event(EventKind.FAULT, reason=reason)


def run(stack: Stack, actions: list[StackAction]) -> tuple[StackState, list[Event]]:
    state, events = stack.initial_state(), []
    for action in actions:
        transition = stack.step(state, action)
        assert transition.work == 1
        state = transition.next_state
        events.append(transition.event)
    return state, events


def test_last_in_first_out(stack: Stack) -> None:
    state, events = run(
        stack, [Push(1), Push(2), Push(3), Pop(), Pop(), Push(4), Pop()]
    )
    assert [e.value for e in events if e.kind is EventKind.OUTPUT] == [3, 2, 4]
    assert state == StackState((1,))


def test_overflow_and_underflow_fault_without_mutation(stack: Stack) -> None:
    full, _ = run(stack, [Push(v) for v in range(CAPACITY)])
    transition = stack.step(full, Push(9))
    assert (transition.event, transition.next_state) == (
        fault(StackFault.OVERFLOW),
        full,
    )
    empty = stack.initial_state()
    transition = stack.step(empty, Pop())
    assert (transition.event, transition.next_state) == (
        fault(StackFault.UNDERFLOW),
        empty,
    )


@pytest.mark.parametrize("value", [-1, MAX_WORD + 1])
def test_value_fault_takes_precedence_over_overflow(stack: Stack, value: int) -> None:
    full, _ = run(stack, [Push(0)] * CAPACITY)
    assert stack.step(full, Push(value)).event == fault(StackFault.VALUE_OUT_OF_RANGE)


def test_rejects_bad_actions_sizes_and_states(stack: Stack) -> None:
    with pytest.raises(TypeError):
        Push(True)
    with pytest.raises(TypeError):
        stack.step(stack.initial_state(), "pop")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Stack(0, 8)
    with pytest.raises(ValueError):
        stack.step(StackState((0,) * (CAPACITY + 1)), Pop())


def test_snapshot_round_trip_and_rejections(stack: Stack) -> None:
    state, _ = run(stack, [Push(7), Push(200)])
    data = json.loads(json.dumps(stack.snapshot(state)))
    assert stack.restore(data) == state
    for key, bad in [
        ("items", [0] * (CAPACITY + 1)),
        ("items", [MAX_WORD + 1]),
        ("items", [True]),
        ("capacity", CAPACITY + 1),
        ("version", "stack/v0"),
    ]:
        with pytest.raises(ValueError):
            stack.restore(data | {key: bad})
    with pytest.raises(ValueError):
        stack.restore(data | {"pointer": 2})


def reference(actions: list[StackAction]) -> list[tuple[str, object]]:
    items: list[int] = []
    events: list[tuple[str, object]] = []
    for action in actions:
        if isinstance(action, Push):
            if not 0 <= action.value <= MAX_WORD:
                events.append(("fault", "value_out_of_range"))
            elif len(items) == CAPACITY:
                events.append(("fault", "overflow"))
            else:
                items.append(action.value)
                events.append(("continue", None))
        elif items:
            events.append(("output", items.pop()))
        else:
            events.append(("fault", "underflow"))
    return events


@pytest.mark.parametrize("seed", range(5))
def test_matches_list_model_on_random_actions(stack: Stack, seed: int) -> None:
    rng = random.Random(seed)
    actions: list[StackAction] = [
        Push(rng.randrange(-1, MAX_WORD + 2)) if rng.random() < 0.55 else Pop()
        for _ in range(1000)
    ]
    _, events = run(stack, actions)
    payload = [(e.kind.value, e.reason.value if e.reason else e.value) for e in events]
    assert payload == reference(actions)
