"""Event vocabulary shared by machine specifications.

See "Machine transitions" in EXPERIMENTS.md. Each machine supports an
explicit subset of kinds and defines its own fault reasons as an ``Enum``.
"""

from dataclasses import dataclass
from enum import Enum


class EventKind(Enum):
    CONTINUE = "continue"
    OUTPUT = "output"
    HALT = "halt"
    FAULT = "fault"


@dataclass(frozen=True, slots=True)
class Event:
    kind: EventKind
    value: int | None = None  # for OUTPUT only
    reason: Enum | None = None  # the machine's fault reason, for FAULT only
