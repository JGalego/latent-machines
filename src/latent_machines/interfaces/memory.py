"""Controller interfaces to a ``memory/v1`` machine for E02.

A controller chooses a typed action from three head outputs: an operation,
a record (address), and a value. Two transports carry the action and the
machine's reply:

- **Text** (arm B): the request and the reply are rendered as text chunks
  appended to the language model's context, such as ``" [read c] [7]"``.
  The model must read the reply back through its own computation.
- **Typed** (arms C and D): the action goes to the machine as an ``Action``
  and the reply returns as an integer observation, never as text.

Both transports carry the same information. The interface owns neither the
task's answers nor the machine's arithmetic.
"""

from latent_machines.machines.events import Event, EventKind
from latent_machines.machines.memory import Action, Read, Write
from latent_machines.tasks.records import NAMES

OPERATIONS = ("write", "read")
NO_OBSERVATION = 0  # observation for events that carry no value


def to_action(operation: int, record: int, value: int) -> Action:
    """The typed action for head outputs; ``value`` is ignored for reads."""
    if OPERATIONS[operation] == "write":
        return Write(record, value)
    return Read(record)


def from_action(action: Action) -> tuple[int, int, int]:
    """Head targets (operation, record, value) for a gold action."""
    match action:
        case Write(address, value):
            return OPERATIONS.index("write"), address, value
        case Read(address):
            return OPERATIONS.index("read"), address, 0
    raise ValueError(f"no head target for {action!r}")


def observation(event: Event) -> int:
    """The typed observation: the value read, or ``NO_OBSERVATION``."""
    return (
        event.value
        if event.kind is EventKind.OUTPUT and event.value is not None
        else NO_OBSERVATION
    )


def text_request(action: Action) -> str:
    match action:
        case Write(address, value):
            return f" [write {NAMES[address]} {value}]"
        case Read(address):
            return f" [read {NAMES[address]}]"
    raise ValueError(f"no text form for {action!r}")


def text_reply(event: Event) -> str:
    if event.kind is EventKind.OUTPUT:
        return f" [{event.value}]"
    if event.kind is EventKind.FAULT and event.reason is not None:
        return f" [fault {event.reason.value}]"
    return " [ok]"
