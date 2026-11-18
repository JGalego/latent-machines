"""Binary outer-totalistic cellular automata: the exact-branch oracle for E06.

Specification ``cellular/v1``:

- A machine has a rule, a grid of ``height`` x ``width`` cells, a boundary,
  and an initial pattern. Each cell is 0 (dead) or 1 (alive).
- Rule: birth and survival sets of Moore-neighborhood counts (0-8), written
  ``B3/S23``. A dead cell with ``n`` live neighbors becomes alive iff
  ``n`` is in the birth set; a live cell stays alive iff ``n`` is in the
  survival set.
- Boundary ``zero``: cells outside the grid are permanently dead.
  Boundary ``torus``: rows and columns wrap around.
- Actions:

  - ``Advance()``: one generation. Every cell's next value is computed
    from the current grid only (simultaneous update, as with double
    buffering), then the whole grid is replaced. Yields ``output`` carrying
    the new population (number of live cells). Work: ``height * width``
    cell updates.
  - ``Set(row, col, value)``: overwrite one cell. Validation precedes any
    effect: ``0 <= row < height`` and ``0 <= col < width`` (else fault
    ``cell_out_of_range``), then ``value in {0, 1}`` (else fault
    ``value_out_of_range``). Yields ``continue``. Work: 1 unit, faulted
    actions included. A fault leaves the state unchanged.

  Arguments must be ``int``, not ``bool``; anything else is outside the
  action schema and raises ``TypeError``.
- State: the grid plus the generation count (number of ``Advance`` steps).
- Snapshots: a JSON-compatible dict tagged with the version and a digest of
  the rule, size, boundary, and initial pattern, so a snapshot cannot be
  restored into a different machine. Every action completes in one step,
  so there is never pending work between steps.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from latent_machines.machines.events import Event, EventKind

VERSION = "cellular/v1"
BOUNDARIES = ("zero", "torus")
NEIGHBOR_COUNTS = range(9)  # Moore neighborhood: 0..8 live neighbors
_RULE = re.compile(r"B([0-8]*)/S([0-8]*)")

type Cells = tuple[tuple[int, ...], ...]


def _require_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be int, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class Advance:
    pass


@dataclass(frozen=True, slots=True)
class Set:
    row: int
    col: int
    value: int

    def __post_init__(self) -> None:
        for name in ("row", "col", "value"):
            _require_int(name, getattr(self, name))


type Action = Advance | Set


class CellularFault(Enum):
    CELL_OUT_OF_RANGE = "cell_out_of_range"
    VALUE_OUT_OF_RANGE = "value_out_of_range"


@dataclass(frozen=True)
class Rule:
    """Birth and survival neighbor counts, parsed from ``B.../S...``."""

    birth: frozenset[int]
    survival: frozenset[int]

    @classmethod
    def parse(cls, text: str) -> "Rule":
        match = _RULE.fullmatch(text)
        if match is None:
            raise ValueError(f"rule {text!r} is not of the form B<digits>/S<digits>")
        birth, survival = (frozenset(int(d) for d in group) for group in match.groups())
        return cls(birth, survival)

    @property
    def text(self) -> str:
        return f"B{''.join(map(str, sorted(self.birth)))}/S" + "".join(
            map(str, sorted(self.survival))
        )

    def next_value(self, value: int, neighbors: int) -> int:
        return int(neighbors in (self.survival if value else self.birth))


@dataclass(frozen=True, slots=True)
class Grid:
    cells: Cells
    generation: int = 0


@dataclass(frozen=True, slots=True)
class Transition:
    next_state: Grid
    event: Event
    work: int


_CONTINUE = Event(EventKind.CONTINUE)


@dataclass(frozen=True)
class Cellular:
    """The ``cellular/v1`` machine for one rule, size, boundary, and start."""

    rule: Rule
    height: int
    width: int
    boundary: str
    initial: Cells

    def __post_init__(self) -> None:
        for name in ("height", "width"):
            size = getattr(self, name)
            _require_int(name, size)
            if size < 1:
                raise ValueError(f"{name} must be at least 1, got {size}")
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"boundary {self.boundary!r} not in {BOUNDARIES}")
        object.__setattr__(self, "initial", self._check_cells(self.initial))

    @property
    def digest(self) -> str:
        spec = [self.rule.text, self.height, self.width, self.boundary, self.initial]
        return hashlib.sha256(json.dumps(spec).encode()).hexdigest()

    @property
    def cells(self) -> int:
        return self.height * self.width

    def initial_state(self) -> Grid:
        return Grid(self.initial)

    def step(self, state: Grid, action: Action) -> Transition:
        if not isinstance(action, (Advance, Set)):
            raise TypeError(f"not a cellular/v1 action: {action!r}")
        match action:
            case Advance():
                cells = tuple(
                    tuple(
                        self.rule.next_value(
                            state.cells[r][c], self._neighbors(state.cells, r, c)
                        )
                        for c in range(self.width)
                    )
                    for r in range(self.height)
                )
                population = sum(map(sum, cells))
                event = Event(EventKind.OUTPUT, value=population)
                return Transition(Grid(cells, state.generation + 1), event, self.cells)
            case Set(row, col, value):
                if not (0 <= row < self.height and 0 <= col < self.width):
                    return self._fault(state, CellularFault.CELL_OUT_OF_RANGE)
                if value not in (0, 1):
                    return self._fault(state, CellularFault.VALUE_OUT_OF_RANGE)
                cells = list(state.cells)
                updated = list(cells[row])
                updated[col] = value
                cells[row] = tuple(updated)
                return Transition(Grid(tuple(cells), state.generation), _CONTINUE, 1)

    def _neighbors(self, cells: Cells, row: int, col: int) -> int:
        total = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                r, c = row + dr, col + dc
                if self.boundary == "torus":
                    total += cells[r % self.height][c % self.width]
                elif 0 <= r < self.height and 0 <= c < self.width:
                    total += cells[r][c]
        return total

    def snapshot(self, state: Grid) -> dict[str, Any]:
        return {
            "version": VERSION,
            "digest": self.digest,
            "generation": state.generation,
            "cells": ["".join(map(str, row)) for row in state.cells],
        }

    def restore(self, snapshot: Mapping[str, Any]) -> Grid:
        if set(snapshot) != {"version", "digest", "generation", "cells"}:
            raise ValueError(f"snapshot keys are {sorted(snapshot)}")
        for key, value in (("version", VERSION), ("digest", self.digest)):
            if snapshot[key] != value:
                raise ValueError(f"snapshot {key} is {snapshot[key]!r}")
        generation = snapshot["generation"]
        if type(generation) is not int or generation < 0:
            raise ValueError(f"snapshot generation {generation!r} is invalid")
        rows = snapshot["cells"]
        if not isinstance(rows, list) or not all(
            isinstance(row, str) and set(row) <= {"0", "1"} for row in rows
        ):
            raise ValueError("snapshot cells must be a list of 0/1 strings")
        return Grid(
            self._check_cells([[int(ch) for ch in row] for row in rows]), generation
        )

    def _check_cells(self, rows: Sequence[Sequence[int]]) -> Cells:
        if len(rows) != self.height or any(len(row) != self.width for row in rows):
            raise ValueError(f"cells must be a {self.height} x {self.width} grid")
        for row in rows:
            for value in row:
                _require_int("cell", value)
                if value not in (0, 1):
                    raise ValueError(f"cell value {value} is not 0 or 1")
        return tuple(tuple(row) for row in rows)

    def _fault(self, state: Grid, reason: CellularFault) -> Transition:
        return Transition(state, Event(EventKind.FAULT, reason=reason), 1)
