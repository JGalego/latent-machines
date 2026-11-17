"""Bounded stack of unsigned words: the reference oracle for E01.

Specification ``stack/v1``:

- State: a sequence of at most ``capacity`` unsigned ``word_bits``-bit
  words, bottom first, empty initially.
- Actions: ``Push(value)`` and ``Pop()``. Values must be ``int``, not
  ``bool``; anything else is outside the action schema and raises
  ``TypeError``.
- ``Push`` checks, in order, ``0 <= value < 2**word_bits`` and that the stack
  is not full. ``Pop`` checks that the stack is not empty. The first failed
  check yields a ``fault`` naming it and leaves the state unchanged.
- Events: ``Push`` yields ``continue``; ``Pop`` yields ``output`` carrying the
  removed top word.
- Work: one unit per action, faulted actions included.
- Snapshots: a JSON-compatible dict of the items, tagged with the
  specification version and parameters.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

from latent_machines.machines.events import Event, EventKind

VERSION = "stack/v1"
WORK_PER_ACTION = 1


def _require_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be int, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class Push:
    value: int

    def __post_init__(self) -> None:
        _require_int("value", self.value)


@dataclass(frozen=True, slots=True)
class Pop:
    pass


type StackAction = Push | Pop


class StackFault(Enum):
    VALUE_OUT_OF_RANGE = "value_out_of_range"
    OVERFLOW = "overflow"
    UNDERFLOW = "underflow"


@dataclass(frozen=True, slots=True)
class StackState:
    items: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class StackTransition:
    next_state: StackState
    event: Event
    work: int


@dataclass(frozen=True, slots=True)
class Stack:
    """The ``stack/v1`` machine with fixed capacity and word width."""

    capacity: int
    word_bits: int

    def __post_init__(self) -> None:
        for name in ("capacity", "word_bits"):
            size = getattr(self, name)
            _require_int(name, size)
            if size < 1:
                raise ValueError(f"{name} must be at least 1, got {size}")

    @property
    def max_word(self) -> int:
        return (1 << self.word_bits) - 1

    def initial_state(self) -> StackState:
        return StackState()

    def step(self, state: StackState, action: StackAction) -> StackTransition:
        if not isinstance(action, Push | Pop):
            raise TypeError(f"not a stack/v1 action: {action!r}")
        if len(state.items) > self.capacity:
            raise ValueError(
                f"state holds {len(state.items)} items, capacity {self.capacity}"
            )
        match action:
            case Push(value):
                if not 0 <= value <= self.max_word:
                    return self._fault(state, StackFault.VALUE_OUT_OF_RANGE)
                if len(state.items) == self.capacity:
                    return self._fault(state, StackFault.OVERFLOW)
                pushed = StackState((*state.items, value))
                return StackTransition(
                    pushed, Event(EventKind.CONTINUE), WORK_PER_ACTION
                )
            case Pop():
                if not state.items:
                    return self._fault(state, StackFault.UNDERFLOW)
                popped = StackState(state.items[:-1])
                event = Event(EventKind.OUTPUT, value=state.items[-1])
                return StackTransition(popped, event, WORK_PER_ACTION)

    def snapshot(self, state: StackState) -> dict[str, Any]:
        return {
            "version": VERSION,
            "capacity": self.capacity,
            "word_bits": self.word_bits,
            "items": list(state.items),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> StackState:
        expected = {
            "version": VERSION,
            "capacity": self.capacity,
            "word_bits": self.word_bits,
        }
        if set(snapshot) != set(expected) | {"items"}:
            raise ValueError(f"snapshot keys are {sorted(snapshot)}")
        for key, value in expected.items():
            actual = snapshot[key]
            if type(actual) is not type(value) or actual != value:
                raise ValueError(f"snapshot {key} is {actual!r}, expected {value!r}")
        items = snapshot["items"]
        if not isinstance(items, list) or len(items) > self.capacity:
            raise ValueError(
                f"snapshot items must be a list of at most {self.capacity}"
            )
        for item in items:
            if type(item) is not int or not 0 <= item <= self.max_word:
                raise ValueError(f"snapshot item {item!r} out of range")
        return StackState(tuple(items))

    def _fault(self, state: StackState, reason: StackFault) -> StackTransition:
        return StackTransition(
            state, Event(EventKind.FAULT, reason=reason), WORK_PER_ACTION
        )
