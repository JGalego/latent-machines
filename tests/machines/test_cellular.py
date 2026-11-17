"""The ``cellular/v1`` oracle against hand-checked patterns."""

import json

import pytest

from latent_machines.machines.cellular import (
    Advance,
    Cellular,
    CellularFault,
    Rule,
    Set,
)
from latent_machines.machines.events import EventKind

LIFE = Rule.parse("B3/S23")


def grid(height: int, width: int, live: set[tuple[int, int]]) -> list[list[int]]:
    return [[int((r, c) in live) for c in range(width)] for r in range(height)]


def live(cells) -> set[tuple[int, int]]:
    return {(r, c) for r, row in enumerate(cells) for c, v in enumerate(row) if v}


def run(machine: Cellular, steps: int):
    state = machine.initial_state()
    for _ in range(steps):
        state = machine.step(state, Advance()).next_state
    return state


def test_rule_parsing_round_trips() -> None:
    rule = Rule.parse("B36/S23")
    assert (rule.birth, rule.survival) == ({3, 6}, {2, 3})
    assert rule.text == "B36/S23"
    assert Rule.parse("B2/S").survival == frozenset()
    with pytest.raises(ValueError):
        Rule.parse("B9/S23")


def test_blinker_has_period_two() -> None:
    start = {(2, 1), (2, 2), (2, 3)}
    machine = Cellular(LIFE, 5, 5, "zero", grid(5, 5, start))
    once = machine.step(machine.initial_state(), Advance())
    assert live(once.next_state.cells) == {(1, 2), (2, 2), (3, 2)}
    assert once.event.kind is EventKind.OUTPUT and once.event.value == 3
    assert once.work == 25
    assert live(run(machine, 2).cells) == start
    assert run(machine, 2).generation == 2


def test_glider_wraps_around_the_torus() -> None:
    glider = {(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)}
    machine = Cellular(LIFE, 6, 6, "torus", grid(6, 6, glider))
    # A glider moves one cell down and right every 4 generations; after 24 it
    # has crossed both seams and returned to its starting cells.
    assert live(run(machine, 4).cells) == {
        ((r + 1) % 6, (c + 1) % 6) for r, c in glider
    }
    assert live(run(machine, 24).cells) == glider


def test_zero_boundary_stops_a_glider() -> None:
    glider = {(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)}
    machine = Cellular(LIFE, 6, 6, "zero", grid(6, 6, glider))
    # At the corner the glider becomes a 2x2 block, which is stable.
    assert live(run(machine, 40).cells) == {(4, 4), (4, 5), (5, 4), (5, 5)}


def test_set_and_faults() -> None:
    machine = Cellular(LIFE, 3, 4, "zero", grid(3, 4, set()))
    state = machine.initial_state()
    done = machine.step(state, Set(1, 3, 1))
    assert done.event.kind is EventKind.CONTINUE and done.work == 1
    assert live(done.next_state.cells) == {(1, 3)}
    for action, reason in (
        (Set(3, 0, 1), CellularFault.CELL_OUT_OF_RANGE),
        (Set(0, -1, 1), CellularFault.CELL_OUT_OF_RANGE),
        (Set(0, 0, 2), CellularFault.VALUE_OUT_OF_RANGE),
    ):
        fault = machine.step(done.next_state, action)
        assert fault.event.kind is EventKind.FAULT and fault.event.reason is reason
        assert fault.next_state == done.next_state and fault.work == 1
    with pytest.raises(TypeError):
        Set(True, 0, 1)
    with pytest.raises(TypeError):
        machine.step(state, "advance")


def test_snapshot_round_trip_and_rejection() -> None:
    machine = Cellular(LIFE, 4, 4, "torus", grid(4, 4, {(1, 1), (1, 2), (2, 1)}))
    state = run(machine, 3)
    snapshot = json.loads(json.dumps(machine.snapshot(state)))
    assert machine.restore(snapshot) == state
    other = Cellular(LIFE, 4, 4, "zero", machine.initial)
    with pytest.raises(ValueError):
        other.restore(snapshot)
    with pytest.raises(ValueError):
        machine.restore(snapshot | {"cells": ["0102"] * 4})


def test_rejects_malformed_machines() -> None:
    with pytest.raises(ValueError):
        Cellular(LIFE, 2, 2, "sphere", grid(2, 2, set()))
    with pytest.raises(ValueError):
        Cellular(LIFE, 2, 2, "zero", grid(2, 3, set()))
