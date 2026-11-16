"""Conformance tests for the memory/v1 integer oracle."""

import json
import random

import pytest

from latent_machines.machines.memory import (
    VERSION,
    WORK_PER_ACTION,
    Action,
    Event,
    EventKind,
    FaultReason,
    Halt,
    Idle,
    Memory,
    MemoryState,
    Read,
    Write,
)

CAPACITY = 16
WORD_BITS = 8
MAX_WORD = 2**WORD_BITS - 1

CONTINUE = Event(EventKind.CONTINUE)


def fault(reason: FaultReason) -> Event:
    return Event(EventKind.FAULT, reason=reason)


@pytest.fixture
def memory() -> Memory:
    return Memory(capacity=CAPACITY, word_bits=WORD_BITS)


def run(
    memory: Memory, actions: list[Action], state: MemoryState | None = None
) -> tuple[MemoryState, list[Event]]:
    """Apply actions in order; return the final state and every event."""
    state = memory.initial_state() if state is None else state
    events = []
    for action in actions:
        transition = memory.step(state, action)
        assert transition.work == WORK_PER_ACTION
        state = transition.next_state
        events.append(transition.event)
    return state, events


def random_action(rng: random.Random) -> Action:
    """A non-halting action, mostly valid, with some out-of-range arguments."""
    address = rng.randrange(-2, CAPACITY + 2)
    value = rng.randrange(-2, MAX_WORD + 3)
    roll = rng.random()
    if roll < 0.45:
        return Read(address)
    if roll < 0.9:
        return Write(address, value)
    return Idle()


# Construction and action schema


@pytest.mark.parametrize(("capacity", "word_bits"), [(0, 8), (-1, 8), (16, 0)])
def test_rejects_non_positive_sizes(capacity: int, word_bits: int) -> None:
    with pytest.raises(ValueError):
        Memory(capacity, word_bits)


@pytest.mark.parametrize(("capacity", "word_bits"), [(True, 8), (16, 8.0), ("16", 8)])
def test_rejects_non_int_sizes(capacity: object, word_bits: object) -> None:
    with pytest.raises(TypeError):
        Memory(capacity, word_bits)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "make",
    [
        lambda: Read(1.0),
        lambda: Read(True),
        lambda: Write(0, "1"),
        lambda: Write(False, 0),
    ],
    ids=["float-address", "bool-address", "str-value", "bool-address-write"],
)
def test_actions_reject_non_int_arguments(make) -> None:
    with pytest.raises(TypeError):
        make()


@pytest.mark.parametrize("halted", [False, True])
def test_step_rejects_unknown_action_even_when_halted(
    memory: Memory, halted: bool
) -> None:
    state = MemoryState((0,) * CAPACITY, halted=halted)
    with pytest.raises(TypeError):
        memory.step(state, "read")  # type: ignore[arg-type]


def test_step_rejects_state_of_another_capacity(memory: Memory) -> None:
    with pytest.raises(ValueError):
        memory.step(MemoryState((0,) * (CAPACITY + 1)), Idle())


# Transitions


def test_initial_state_is_zeroed_and_running(memory: Memory) -> None:
    assert memory.initial_state() == MemoryState((0,) * CAPACITY, halted=False)


def test_write_then_read_returns_value(memory: Memory) -> None:
    state, events = run(memory, [Write(3, 42), Read(3)])
    assert events == [CONTINUE, Event(EventKind.OUTPUT, value=42)]
    assert state.words[3] == 42


def test_read_leaves_state_unchanged(memory: Memory) -> None:
    state, _ = run(memory, [Write(5, 7)])
    assert memory.step(state, Read(5)).next_state == state


def test_last_write_wins(memory: Memory) -> None:
    _, events = run(memory, [Write(4, 1), Write(4, 2), Write(4, 3), Read(4)])
    assert events[-1] == Event(EventKind.OUTPUT, value=3)


def test_every_single_write_changes_only_its_slot(memory: Memory) -> None:
    reads: list[Action] = [Read(address) for address in range(CAPACITY)]
    for address in range(CAPACITY):
        for value in range(MAX_WORD + 1):
            _, events = run(memory, [Write(address, value), *reads])
            outputs = [event.value for event in events[1:]]
            expected = [0] * CAPACITY
            expected[address] = value
            assert outputs == expected, (address, value)


def test_boundary_addresses_and_values_are_valid(memory: Memory) -> None:
    actions: list[Action] = [
        Write(0, MAX_WORD),
        Write(CAPACITY - 1, 0),
        Write(CAPACITY - 1, MAX_WORD),
        Read(0),
        Read(CAPACITY - 1),
    ]
    _, events = run(memory, actions)
    assert events == [CONTINUE] * 3 + [Event(EventKind.OUTPUT, value=MAX_WORD)] * 2


def test_idle_retains_state(memory: Memory) -> None:
    writes: list[Action] = [
        Write(a, (a * 37) % (MAX_WORD + 1)) for a in range(CAPACITY)
    ]
    state, _ = run(memory, writes)
    after, events = run(memory, [Idle()] * 1000, state)
    assert after == state
    assert events == [CONTINUE] * 1000


# Faults: a nonzero state at both ends exposes mutation and negative indexing.


@pytest.fixture
def filled(memory: Memory) -> MemoryState:
    state, _ = run(memory, [Write(0, 9), Write(CAPACITY - 1, 9)])
    return state


@pytest.mark.parametrize("address", [-1, -CAPACITY, CAPACITY, CAPACITY + 100])
@pytest.mark.parametrize(
    "make", [Read, lambda address: Write(address, 1)], ids=["read", "write"]
)
def test_out_of_range_address_faults_without_mutation(
    memory: Memory, filled: MemoryState, address: int, make
) -> None:
    transition = memory.step(filled, make(address))
    assert transition.event == fault(FaultReason.ADDRESS_OUT_OF_RANGE)
    assert transition.next_state == filled
    assert transition.work == WORK_PER_ACTION


@pytest.mark.parametrize("value", [-1, MAX_WORD + 1, 2**64])
def test_out_of_range_value_faults_without_mutation(
    memory: Memory, filled: MemoryState, value: int
) -> None:
    transition = memory.step(filled, Write(0, value))
    assert transition.event == fault(FaultReason.VALUE_OUT_OF_RANGE)
    assert transition.next_state == filled
    assert transition.work == WORK_PER_ACTION


def test_address_fault_takes_precedence_over_value_fault(memory: Memory) -> None:
    transition = memory.step(memory.initial_state(), Write(CAPACITY, MAX_WORD + 1))
    assert transition.event == fault(FaultReason.ADDRESS_OUT_OF_RANGE)


def test_halt_then_every_action_faults(memory: Memory) -> None:
    state, events = run(memory, [Write(2, 5), Halt()])
    assert events[-1] == Event(EventKind.HALT)
    assert state == MemoryState((0, 0, 5) + (0,) * (CAPACITY - 3), halted=True)
    for action in [Read(2), Write(2, 6), Idle(), Halt(), Read(CAPACITY)]:
        transition = memory.step(state, action)
        assert transition.event == fault(FaultReason.HALTED), action
        assert transition.next_state == state
        assert transition.work == WORK_PER_ACTION


# Snapshots


def json_round_trip(data: dict) -> dict:
    return json.loads(json.dumps(data))


def test_snapshot_round_trips_through_json(memory: Memory) -> None:
    state, _ = run(memory, [Write(1, 200), Write(15, 3), Halt()])
    data = json_round_trip(memory.snapshot(state))
    assert data["version"] == VERSION
    assert memory.restore(data) == state


def test_resume_from_snapshot_matches_uninterrupted_run(memory: Memory) -> None:
    rng = random.Random(0)
    actions = [random_action(rng) for _ in range(400)]
    full_state, full_events = run(memory, actions)
    middle, first = run(memory, actions[:200])
    restored = memory.restore(json_round_trip(memory.snapshot(middle)))
    end, second = run(memory, actions[200:], restored)
    assert end == full_state
    assert first + second == full_events


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("version", "memory/v0"),
        ("capacity", CAPACITY + 1),
        ("capacity", float(CAPACITY)),
        ("word_bits", WORD_BITS * 2),
        ("words", [0] * (CAPACITY - 1)),
        ("words", [0] * (CAPACITY - 1) + [MAX_WORD + 1]),
        ("words", [0] * (CAPACITY - 1) + [-1]),
        ("words", [0] * (CAPACITY - 1) + [True]),
        ("words", "0" * CAPACITY),
        ("halted", 0),
    ],
)
def test_restore_rejects_invalid_field(memory: Memory, key: str, bad: object) -> None:
    data = memory.snapshot(memory.initial_state())
    data[key] = bad
    with pytest.raises(ValueError):
        memory.restore(data)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_restore_rejects_missing_or_extra_keys(memory: Memory, change: str) -> None:
    data = memory.snapshot(memory.initial_state())
    if change == "missing":
        del data["halted"]
    else:
        data["pending"] = []
    with pytest.raises(ValueError):
        memory.restore(data)


# Differential check against a separately written mutable-list model


def reference_events(actions: list[Action]) -> list[tuple[str, object]]:
    words = [0] * CAPACITY
    halted = False
    events: list[tuple[str, object]] = []
    for action in actions:
        if halted:
            events.append(("fault", "halted"))
        elif isinstance(action, Read | Write) and not 0 <= action.address < CAPACITY:
            events.append(("fault", "address_out_of_range"))
        elif isinstance(action, Write) and not 0 <= action.value <= MAX_WORD:
            events.append(("fault", "value_out_of_range"))
        elif isinstance(action, Read):
            events.append(("output", words[action.address]))
        elif isinstance(action, Write):
            words[action.address] = action.value
            events.append(("continue", None))
        elif isinstance(action, Idle):
            events.append(("continue", None))
        else:
            halted = True
            events.append(("halt", None))
    return events


def as_tuple(event: Event) -> tuple[str, object]:
    payload = event.reason.value if event.reason else event.value
    return (event.kind.value, payload)


@pytest.mark.parametrize("seed", range(5))
def test_matches_reference_model_on_random_actions(memory: Memory, seed: int) -> None:
    rng = random.Random(seed)
    actions = [random_action(rng) for _ in range(2000)]
    # Halt late so most of the sequence exercises live transitions.
    actions.insert(rng.randrange(1500, 2000), Halt())
    _, events = run(memory, actions)
    assert [as_tuple(event) for event in events] == reference_events(actions)
