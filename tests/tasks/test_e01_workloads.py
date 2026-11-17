"""E01 workloads: the bracket parser task and the stack and symbol streams."""

import random

import pytest

from latent_machines.machines.stack import Pop, Push, Stack
from latent_machines.machines.transducer import parity
from latent_machines.tasks import brackets, stack_workloads, transducer_workloads

STACK = Stack(capacity=4, word_bits=8)


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("([]{})", "balanced"),
        ("(()", "unclosed"),
        ("(]", "mismatch"),
        ("())", "underflow"),
        ("(((((", "overflow"),
        ("", "balanced"),
    ],
)
def test_parse_ground_truth(text: str, verdict: str) -> None:
    assert brackets.parse(text, capacity=4)[0] == verdict


def run_events(text: str) -> tuple:
    state, events = STACK.initial_state(), []
    for action in brackets.actions_for(text, STACK.capacity):
        transition = STACK.step(state, action)
        state = transition.next_state
        events.append(transition.event)
    return state, events


def test_stack_parser_agrees_with_ground_truth_and_leaves_stack_empty() -> None:
    rng = random.Random(0)
    verdicts = set()
    for _ in range(2000):
        text = brackets.random_text(rng, rng.randrange(0, 12))
        state, events = run_events(text)
        expected = brackets.parse(text, STACK.capacity)[0]
        assert brackets.verdict_from_events(text, events) == expected, text
        assert state.items == (), text
        verdicts.add(expected)
    assert verdicts == {"balanced", "unclosed", "mismatch", "underflow", "overflow"}


@pytest.mark.parametrize("name", stack_workloads.WORKLOADS)
def test_stack_workloads_are_deterministic(name: str) -> None:
    def take() -> list:
        return list(stack_workloads.actions(name, STACK, 300, random.Random(1)))

    first = take()
    assert len(first) == 300 and first == take()
    assert any(isinstance(a, Push) for a in first) and any(
        isinstance(a, Pop) for a in first
    )


def test_symbol_stream_mixes_valid_and_invalid_symbols() -> None:
    feeds = list(transducer_workloads.actions(parity(), 2000, random.Random(0)))
    invalid = sum(not 0 <= f.symbol < 2 for f in feeds)
    assert 0 < invalid < 0.1 * len(feeds)
