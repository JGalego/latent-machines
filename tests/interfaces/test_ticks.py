"""Tick-level control and the native loop for E03."""

import json

import pytest

from latent_machines.interfaces import ticks
from latent_machines.machines.events import Event, EventKind
from latent_machines.machines.memory import Read, Write
from latent_machines.tasks import copies


def test_feedback_codes_carry_action_and_reply() -> None:
    read = ticks.feedback(Read(3), Event(EventKind.OUTPUT, value=9))
    assert read.indices() == (ticks.OPERATIONS.index("read"), 3, ticks.NONE_VALUE, 9)
    write = ticks.feedback(Write(1, 5), Event(EventKind.CONTINUE))
    assert write.indices() == (ticks.OPERATIONS.index("write"), 1, 5, ticks.NONE_VALUE)
    assert all(
        i < n for i, n in zip(ticks.START.indices(), ticks.CODE_SIZES, strict=True)
    )


def test_targets_end_every_statement() -> None:
    assert ticks.targets([Read(2), Write(4, 7)]) == [(1, 2, 0), (0, 4, 7), (2, 0, 0)]
    assert ticks.to_action(*ticks.targets([Write(4, 7)])[0]) == Write(4, 7)
    assert ticks.to_action(ticks.OPERATIONS.index("end"), 0, 0) is None


def run(loop: ticks.NativeLoop, machine, state):
    actions = []
    while (action := loop.next_action()) is not None:
        transition = machine.step(state, action)
        state = transition.next_state
        actions.append(action)
        loop.observe(transition.event)
    return actions, state


def test_native_copy_resumes_from_a_snapshot_between_ticks() -> None:
    machine = copies.machine()
    state = machine.step(machine.initial_state(), Write(2, 7)).next_state
    macro = ticks.macro_target([Read(2), Write(4, 7)])
    whole, expected = run(ticks.NativeLoop(macro), machine, state)
    assert whole == [Read(2), Write(4, 7)]

    loop = ticks.NativeLoop(macro)
    first = loop.next_action()
    transition = machine.step(state, first)
    loop.observe(transition.event)
    data = json.loads(json.dumps(loop.snapshot()))
    assert data["pending"] == {"record": 4, "value": 7}
    rest, resumed = run(ticks.NativeLoop.restore(data), machine, transition.next_state)
    assert [first, *rest] == whole
    assert resumed == expected


def test_macro_targets() -> None:
    assert ticks.macro_target([Write(1, 3)]) == ticks.Macro(0, 1, value=3)
    assert ticks.macro_target([Read(1)]) == ticks.Macro(1, 1)
    with pytest.raises(ValueError):
        ticks.macro_target([Write(1, 3), Read(1)])
