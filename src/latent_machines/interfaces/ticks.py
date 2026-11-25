"""Multi-tick control of a ``memory/v1`` machine for E03.

A statement may take several machine ticks. Every arm uses the same action
vocabulary per tick: ``write``, ``read``, or ``end``, which returns control
to the task without a machine action. A statement is bounded: after
``TICK_BUDGET`` machine actions without ``end``, the loop stops with a
``budget_exhausted`` outcome.

After each tick, the action and the machine's event form a ``FeedbackCode``
of four small integers. The arms differ only in how that code reaches the
next decision:

- token feedback: rendered as request and reply text (``memory.text_request``
  and ``memory.text_reply``) and read by the language model;
- latent feedback: embedded as one input vector that enters the language
  model's next position;
- head feedback: embedded and given to the trained heads, not the model.

The native loop is the conventional baseline: a controller issues one macro
per statement (``write``, ``read``, or ``copy``) and ``NativeLoop`` executes
it tick by tick, holding the read value of a copy as pending work that a
snapshot must carry.
"""

from dataclasses import asdict, dataclass
from typing import Any

from latent_machines.machines.events import Event, EventKind
from latent_machines.machines.memory import Action, Read, Write
from latent_machines.tasks.copies import MAX_VALUE, NAMES

OPERATIONS = ("write", "read", "end")
MACROS = ("write", "read", "copy")
TICK_BUDGET = 3  # machine actions per statement; gold control needs at most 2

NONE_OP = len(OPERATIONS) - 1  # "end" never reaches feedback; reused as "none"
NONE_RECORD = len(NAMES)
NONE_VALUE = MAX_VALUE + 1
CODE_SIZES = (len(OPERATIONS), len(NAMES) + 1, MAX_VALUE + 2, MAX_VALUE + 2)


@dataclass(frozen=True)
class FeedbackCode:
    """One tick's action and reply as indices; ``START`` before any tick."""

    operation: int
    record: int
    value: int
    observation: int

    def indices(self) -> tuple[int, int, int, int]:
        return (self.operation, self.record, self.value, self.observation)


START = FeedbackCode(NONE_OP, NONE_RECORD, NONE_VALUE, NONE_VALUE)


def feedback(action: Action, event: Event) -> FeedbackCode:
    observation = (
        event.value
        if event.kind is EventKind.OUTPUT and event.value is not None
        else NONE_VALUE
    )
    match action:
        case Write(address, value):
            return FeedbackCode(OPERATIONS.index("write"), address, value, observation)
        case Read(address):
            return FeedbackCode(
                OPERATIONS.index("read"), address, NONE_VALUE, observation
            )
    raise ValueError(f"no feedback code for {action!r}")


def reply_only(code: FeedbackCode) -> FeedbackCode:
    """``code`` without its action: constants for the operation, record, and
    value written, and the observation unchanged. The observation is the value
    a read returned, or ``NONE_VALUE`` when the event carried none."""
    return FeedbackCode(NONE_OP, NONE_RECORD, NONE_VALUE, code.observation)


def to_action(operation: int, record: int, value: int) -> Action | None:
    """The machine action for head outputs, or ``None`` for ``end``."""
    name = OPERATIONS[operation]
    if name == "end":
        return None
    return Write(record, value) if name == "write" else Read(record)


def targets(actions: list[Action]) -> list[tuple[int, int, int]]:
    """Per-decision head targets for a statement: its actions, then ``end``."""
    rows = []
    for action in actions:
        match action:
            case Write(address, value):
                rows.append((OPERATIONS.index("write"), address, value))
            case Read(address):
                rows.append((OPERATIONS.index("read"), address, 0))
            case _:
                raise ValueError(f"no head target for {action!r}")
    return rows + [(OPERATIONS.index("end"), 0, 0)]


# ---------- native loop ----------


@dataclass(frozen=True)
class Macro:
    operation: int  # index into MACROS
    record: int  # written or read record
    source: int = 0  # for copy
    value: int = 0  # for write


def macro_target(actions: list[Action]) -> Macro:
    """The macro whose expansion is ``actions``."""
    match actions:
        case [Write(address, value)]:
            return Macro(MACROS.index("write"), address, value=value)
        case [Read(address)]:
            return Macro(MACROS.index("read"), address)
        case [Read(source), Write(address, _)]:
            return Macro(MACROS.index("copy"), address, source=source)
    raise ValueError(f"no macro for {actions!r}")


class NativeLoop:
    """Executes one macro tick by tick.

    Call ``next_action``, execute it on the machine, and pass the event to
    ``observe``, until ``next_action`` returns ``None``. Between ticks the
    loop's state is plain data: the macro, the phase, and, for a copy, the
    pending write with the value read. A snapshot taken between the read and
    the write therefore resumes in a new loop without repeating the read.
    """

    def __init__(
        self, macro: Macro, phase: int = 0, pending: dict[str, int] | None = None
    ) -> None:
        self.macro, self.phase, self.pending = macro, phase, pending

    def next_action(self) -> Action | None:
        name = MACROS[self.macro.operation]
        if self.phase == 0:
            if name == "write":
                return Write(self.macro.record, self.macro.value)
            return Read(self.macro.source if name == "copy" else self.macro.record)
        if self.phase == 1:
            assert self.pending is not None
            return Write(self.pending["record"], self.pending["value"])
        return None

    def observe(self, event: Event) -> None:
        copying = MACROS[self.macro.operation] == "copy"
        if self.phase == 0 and copying and event.kind is EventKind.OUTPUT:
            assert event.value is not None
            self.phase, self.pending = (
                1,
                {"record": self.macro.record, "value": event.value},
            )
        else:
            self.phase, self.pending = 2, None

    def snapshot(self) -> dict[str, Any]:
        return {
            "macro": asdict(self.macro),
            "phase": self.phase,
            "pending": self.pending,
        }

    @classmethod
    def restore(cls, data: dict[str, Any]) -> "NativeLoop":
        return cls(Macro(**data["macro"]), data["phase"], data["pending"])
