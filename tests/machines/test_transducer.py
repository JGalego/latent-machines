"""Conformance tests for the transducer/v1 oracle and its E01 targets."""

import json
import random
from itertools import permutations

import pytest

from latent_machines.machines.events import Event, EventKind
from latent_machines.machines.transducer import (
    Feed,
    Transducer,
    TransducerFault,
    TransducerState,
    modular_counter,
    parity,
    permutation_tracker,
)

FAULT = Event(EventKind.FAULT, reason=TransducerFault.SYMBOL_OUT_OF_RANGE)


def outputs(machine: Transducer, symbols: list[int]) -> list[int | None]:
    state = machine.initial_state()
    values = []
    for symbol in symbols:
        transition = machine.step(state, Feed(symbol))
        assert transition.work == 1
        state = transition.next_state
        values.append(transition.event.value)
    return values


def test_parity_matches_running_xor() -> None:
    bits = [random.Random(0).randrange(2) for _ in range(200)]
    expected, acc = [], 0
    for bit in bits:
        acc ^= bit
        expected.append(acc)
    assert outputs(parity(), bits) == expected


@pytest.mark.parametrize(("modulus", "alphabet"), [(5, 3), (7, 2), (1, 4)])
def test_modular_counter_matches_running_sum(modulus: int, alphabet: int) -> None:
    rng = random.Random(modulus)
    steps = [rng.randrange(alphabet) for _ in range(200)]
    expected = [sum(steps[: i + 1]) % modulus for i in range(len(steps))]
    assert outputs(modular_counter(modulus, alphabet), steps) == expected


@pytest.mark.parametrize("n", [2, 3, 4])
def test_permutation_tracker_matches_explicit_swaps(n: int) -> None:
    rng = random.Random(n)
    swaps = [rng.randrange(n - 1) for _ in range(200)]
    arrangements = list(permutations(range(n)))
    arrangement, expected = list(range(n)), []
    for i in swaps:
        arrangement[i], arrangement[i + 1] = arrangement[i + 1], arrangement[i]
        expected.append(arrangements.index(tuple(arrangement)))
    assert outputs(permutation_tracker(n), swaps) == expected


def test_s3_swaps_do_not_commute() -> None:
    tracker = permutation_tracker(3)
    assert outputs(tracker, [0, 1])[-1] != outputs(tracker, [1, 0])[-1]


@pytest.mark.parametrize("symbol", [-1, 2, 100])
def test_out_of_range_symbol_faults_without_mutation(symbol: int) -> None:
    machine = parity()
    state = machine.step(machine.initial_state(), Feed(1)).next_state
    transition = machine.step(state, Feed(symbol))
    assert (transition.event, transition.next_state, transition.work) == (
        FAULT,
        state,
        1,
    )


@pytest.mark.parametrize("make", [lambda: Feed(True), lambda: Feed(1.0)])
def test_feed_rejects_non_int_symbols(make) -> None:
    with pytest.raises(TypeError):
        make()


def test_step_rejects_unknown_action_and_bad_state() -> None:
    machine = parity()
    with pytest.raises(TypeError):
        machine.step(machine.initial_state(), 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        machine.step(TransducerState(2), Feed(0))


@pytest.mark.parametrize(
    ("next_state", "output", "output_size", "start"),
    [
        ([[0, 1]], [[0, 0]], 1, 0),  # next state out of range
        ([[0], [0]], [[0], [1]], 1, 0),  # output out of range
        ([[0, 0], [0]], [[0, 0], [0]], 1, 0),  # ragged rows
        ([[0]], [[0]], 1, 1),  # start out of range
        ([[True]], [[0]], 1, 0),  # bool entry
    ],
)
def test_rejects_invalid_tables(next_state, output, output_size, start) -> None:
    with pytest.raises((ValueError, TypeError)):
        Transducer("bad", next_state, output, output_size, start)


def test_snapshot_round_trips_and_rejects_other_machines() -> None:
    counter = modular_counter(5, 3)
    state = TransducerState(3)
    data = json.loads(json.dumps(counter.snapshot(state)))
    assert counter.restore(data) == state
    with pytest.raises(ValueError):
        modular_counter(5, 2).restore(data)  # same name, different table
    for key, bad in [("state", 5), ("state", True), ("version", "transducer/v0")]:
        with pytest.raises(ValueError):
            counter.restore(data | {key: bad})
    with pytest.raises(ValueError):
        counter.restore(data | {"extra": 0})
