"""Integer random-access memory: the reference oracle for E00.

Specification ``memory/v1``:

- State: ``capacity`` slots, each holding an unsigned ``word_bits``-bit
  integer, all zero initially, plus a halted flag.
- Actions: ``Read(address)``, ``Write(address, value)``, ``Idle()``, and
  ``Halt()``. Arguments must be ``int``, not ``bool``; anything else is
  outside the action schema and raises ``TypeError``.
- Validation precedes any effect and checks, in order: the machine is not
  halted, ``0 <= address < capacity``, and ``0 <= value < 2**word_bits``.
  The first failed check yields a ``fault`` event naming it and leaves the
  state unchanged. Values are never clamped or wrapped, and negative
  addresses are not Python-style indices from the end.
- Events: ``Read`` yields ``output`` carrying the stored word, ``Write`` and
  ``Idle`` yield ``continue``, and ``Halt`` yields ``halt``. Every action on
  a halted machine faults, including ``Halt``.
- Work: one unit per action, faulted actions included.
- Snapshots: a JSON-compatible dict of the canonical state, tagged with the
  specification version and parameters. Each action completes in one step,
  so there is never pending work between steps.

States are immutable and must come from ``initial_state``, ``step``, or
``restore`` on a machine with the same parameters.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

VERSION = "memory/v1"
WORK_PER_ACTION = 1


def _require_int(name: str, value: object) -> None:
    # bool subclasses int, but True is not a valid address, value, or size.
    if type(value) is not int:
        raise TypeError(f"{name} must be int, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class Read:
    address: int

    def __post_init__(self) -> None:
        _require_int("address", self.address)


@dataclass(frozen=True, slots=True)
class Write:
    address: int
    value: int

    def __post_init__(self) -> None:
        _require_int("address", self.address)
        _require_int("value", self.value)


@dataclass(frozen=True, slots=True)
class Idle:
    pass


@dataclass(frozen=True, slots=True)
class Halt:
    pass


type Action = Read | Write | Idle | Halt
_ACTION_TYPES = (Read, Write, Idle, Halt)


class EventKind(Enum):
    CONTINUE = "continue"
    OUTPUT = "output"
    HALT = "halt"
    FAULT = "fault"


class FaultReason(Enum):
    HALTED = "halted"
    ADDRESS_OUT_OF_RANGE = "address_out_of_range"
    VALUE_OUT_OF_RANGE = "value_out_of_range"


@dataclass(frozen=True, slots=True)
class Event:
    kind: EventKind
    value: int | None = None  # the word read, for OUTPUT only
    reason: FaultReason | None = None  # for FAULT only


@dataclass(frozen=True, slots=True)
class MemoryState:
    words: tuple[int, ...]
    halted: bool = False


@dataclass(frozen=True, slots=True)
class Transition:
    next_state: MemoryState
    event: Event
    work: int


_CONTINUE = Event(EventKind.CONTINUE)


@dataclass(frozen=True, slots=True)
class Memory:
    """The ``memory/v1`` machine with fixed capacity and word width."""

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

    def initial_state(self) -> MemoryState:
        return MemoryState(words=(0,) * self.capacity)

    def step(self, state: MemoryState, action: Action) -> Transition:
        if not isinstance(action, _ACTION_TYPES):
            raise TypeError(f"not a memory/v1 action: {action!r}")
        if len(state.words) != self.capacity:
            raise ValueError(
                f"state has {len(state.words)} words, machine has {self.capacity}"
            )
        if state.halted:
            return self._fault(state, FaultReason.HALTED)

        match action:
            case Read(address):
                if not 0 <= address < self.capacity:
                    return self._fault(state, FaultReason.ADDRESS_OUT_OF_RANGE)
                output = Event(EventKind.OUTPUT, value=state.words[address])
                return Transition(state, output, WORK_PER_ACTION)
            case Write(address, value):
                if not 0 <= address < self.capacity:
                    return self._fault(state, FaultReason.ADDRESS_OUT_OF_RANGE)
                if not 0 <= value <= self.max_word:
                    return self._fault(state, FaultReason.VALUE_OUT_OF_RANGE)
                words = state.words[:address] + (value,) + state.words[address + 1 :]
                return Transition(MemoryState(words), _CONTINUE, WORK_PER_ACTION)
            case Idle():
                return Transition(state, _CONTINUE, WORK_PER_ACTION)
            case Halt():
                halted = MemoryState(state.words, halted=True)
                return Transition(halted, Event(EventKind.HALT), WORK_PER_ACTION)

    def snapshot(self, state: MemoryState) -> dict[str, Any]:
        return {
            "version": VERSION,
            "capacity": self.capacity,
            "word_bits": self.word_bits,
            "words": list(state.words),
            "halted": state.halted,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> MemoryState:
        """Rebuild a state from ``snapshot``, rejecting anything malformed."""
        expected_keys = {"version", "capacity", "word_bits", "words", "halted"}
        if set(snapshot) != expected_keys:
            raise ValueError(f"snapshot keys are {sorted(snapshot)}")
        for key, expected in (
            ("version", VERSION),
            ("capacity", self.capacity),
            ("word_bits", self.word_bits),
        ):
            actual = snapshot[key]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(f"snapshot {key} is {actual!r}, expected {expected!r}")

        words = snapshot["words"]
        if not isinstance(words, list) or len(words) != self.capacity:
            raise ValueError(f"snapshot words must be a list of {self.capacity} ints")
        for address, word in enumerate(words):
            if type(word) is not int or not 0 <= word <= self.max_word:
                raise ValueError(f"snapshot word {address} is {word!r}, out of range")
        halted = snapshot["halted"]
        if type(halted) is not bool:
            raise ValueError(f"snapshot halted is {halted!r}, expected bool")
        return MemoryState(tuple(words), halted)

    def _fault(self, state: MemoryState, reason: FaultReason) -> Transition:
        return Transition(state, Event(EventKind.FAULT, reason=reason), WORK_PER_ACTION)
