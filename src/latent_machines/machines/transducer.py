"""Deterministic finite-state transducers: the reference oracle for E01.

Specification ``transducer/v1``:

- A transducer has states ``0..num_states-1``, input symbols
  ``0..alphabet_size-1``, output symbols ``0..output_size-1``, a transition
  table ``next_state[state][symbol]``, an output table
  ``output[state][symbol]``, and a start state.
- Action: ``Feed(symbol)``. The symbol must be ``int``, not ``bool``; anything
  else is outside the action schema and raises ``TypeError``.
- An in-range symbol moves to ``next_state[state][symbol]`` and yields
  ``output`` carrying ``output[state][symbol]``. An out-of-range symbol yields
  ``fault`` with reason ``symbol_out_of_range`` and leaves the state unchanged.
- Work: one unit per action, faulted actions included.
- Snapshots: a JSON-compatible dict with the version, the transducer's name
  and table digest, and the current state, so a snapshot cannot be restored
  into a different transducer.

``parity``, ``modular_counter``, and ``permutation_tracker`` build the E01
targets.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import permutations
from typing import Any

from latent_machines.machines.events import Event, EventKind

VERSION = "transducer/v1"
WORK_PER_ACTION = 1


def _require_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be int, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class Feed:
    symbol: int

    def __post_init__(self) -> None:
        _require_int("symbol", self.symbol)


class TransducerFault(Enum):
    SYMBOL_OUT_OF_RANGE = "symbol_out_of_range"


@dataclass(frozen=True, slots=True)
class TransducerState:
    state: int


@dataclass(frozen=True, slots=True)
class TransducerTransition:
    next_state: TransducerState
    event: Event
    work: int


def _table(rows: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(row) for row in rows)


@dataclass(frozen=True)
class Transducer:
    """A ``transducer/v1`` machine defined by its two tables."""

    name: str
    next_state: tuple[tuple[int, ...], ...]
    output: tuple[tuple[int, ...], ...]
    output_size: int
    start: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "next_state", _table(self.next_state))
        object.__setattr__(self, "output", _table(self.output))
        states, symbols = self.num_states, self.alphabet_size
        if states < 1 or symbols < 1 or self.output_size < 1:
            raise ValueError("states, symbols, and outputs must each be at least 1")
        for table, bound, label in (
            (self.next_state, states, "next_state"),
            (self.output, self.output_size, "output"),
        ):
            if len(table) != states or any(len(row) != symbols for row in table):
                raise ValueError(f"{label} must be a {states} x {symbols} table")
            for row in table:
                for entry in row:
                    _require_int(label, entry)
                    if not 0 <= entry < bound:
                        raise ValueError(f"{label} entry {entry} out of range")
        _require_int("start", self.start)
        if not 0 <= self.start < states:
            raise ValueError(f"start state {self.start} out of range")

    @property
    def num_states(self) -> int:
        return len(self.next_state)

    @property
    def alphabet_size(self) -> int:
        return len(self.next_state[0])

    @property
    def digest(self) -> str:
        tables = [self.next_state, self.output, self.output_size, self.start]
        return hashlib.sha256(json.dumps(tables).encode()).hexdigest()

    def initial_state(self) -> TransducerState:
        return TransducerState(self.start)

    def step(self, state: TransducerState, action: Feed) -> TransducerTransition:
        if not isinstance(action, Feed):
            raise TypeError(f"not a transducer/v1 action: {action!r}")
        if not 0 <= state.state < self.num_states:
            raise ValueError(f"state {state.state} out of range")
        symbol = action.symbol
        if not 0 <= symbol < self.alphabet_size:
            fault = Event(EventKind.FAULT, reason=TransducerFault.SYMBOL_OUT_OF_RANGE)
            return TransducerTransition(state, fault, WORK_PER_ACTION)
        next_state = TransducerState(self.next_state[state.state][symbol])
        event = Event(EventKind.OUTPUT, value=self.output[state.state][symbol])
        return TransducerTransition(next_state, event, WORK_PER_ACTION)

    def snapshot(self, state: TransducerState) -> dict[str, Any]:
        return {
            "version": VERSION,
            "name": self.name,
            "digest": self.digest,
            "state": state.state,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> TransducerState:
        expected = {"version": VERSION, "name": self.name, "digest": self.digest}
        if set(snapshot) != set(expected) | {"state"}:
            raise ValueError(f"snapshot keys are {sorted(snapshot)}")
        for key, value in expected.items():
            if snapshot[key] != value:
                raise ValueError(
                    f"snapshot {key} is {snapshot[key]!r}, expected {value!r}"
                )
        state = snapshot["state"]
        if type(state) is not int or not 0 <= state < self.num_states:
            raise ValueError(f"snapshot state {state!r} out of range")
        return TransducerState(state)


def parity() -> Transducer:
    """Running parity of a bit stream; outputs the parity after each bit."""
    table = [[state ^ bit for bit in (0, 1)] for state in (0, 1)]
    return Transducer("parity", table, table, output_size=2)


def modular_counter(modulus: int, alphabet_size: int = 2) -> Transducer:
    """Running sum of increments ``0..alphabet_size-1`` modulo ``modulus``."""
    table = [
        [(count + step) % modulus for step in range(alphabet_size)]
        for count in range(modulus)
    ]
    return Transducer(f"mod{modulus}-counter", table, table, output_size=modulus)


def permutation_tracker(n: int) -> Transducer:
    """Track an arrangement of ``n`` items under adjacent swaps.

    States are the ``n!`` arrangements in lexicographic order, starting from
    the identity; symbol ``i`` swaps positions ``i`` and ``i + 1``. For
    ``n >= 3`` the swaps do not commute, so order matters.
    """
    arrangements = list(permutations(range(n)))
    index = {arrangement: i for i, arrangement in enumerate(arrangements)}
    table = []
    for arrangement in arrangements:
        row = []
        for i in range(n - 1):
            swapped = list(arrangement)
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            row.append(index[tuple(swapped)])
        table.append(row)
    return Transducer(f"s{n}-tracker", table, table, output_size=len(arrangements))
