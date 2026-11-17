"""Text and typed transports for memory actions and replies."""

from latent_machines.interfaces import memory as interface
from latent_machines.machines.events import Event, EventKind
from latent_machines.machines.memory import FaultReason, Read, Write


def test_head_targets_round_trip() -> None:
    for action in (Write(3, 9), Read(5)):
        assert interface.to_action(*interface.from_action(action)) == action


def test_text_transport() -> None:
    assert interface.text_request(Write(2, 7)) == " [write c 7]"
    assert interface.text_request(Read(0)) == " [read a]"
    assert interface.text_reply(Event(EventKind.OUTPUT, value=7)) == " [7]"
    assert interface.text_reply(Event(EventKind.CONTINUE)) == " [ok]"
    fault = Event(EventKind.FAULT, reason=FaultReason.ADDRESS_OUT_OF_RANGE)
    assert interface.text_reply(fault) == " [fault address_out_of_range]"


def test_typed_observation() -> None:
    assert interface.observation(Event(EventKind.OUTPUT, value=11)) == 11
    assert interface.observation(Event(EventKind.CONTINUE)) == interface.NO_OBSERVATION
