"""Stepwise conformance of the eager delta-rule backend to memory/v1."""

import json
import random

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends.delta import DeltaMemory, DeltaState  # noqa: E402
from latent_machines.machines.memory import (  # noqa: E402
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

DTYPES = [torch.float32, torch.float64, torch.bfloat16, torch.float16]


@pytest.fixture
def memory() -> Memory:
    return Memory(capacity=CAPACITY, word_bits=WORD_BITS)


@pytest.fixture(params=DTYPES, ids=str)
def backend(request: pytest.FixtureRequest, memory: Memory) -> DeltaMemory:
    return DeltaMemory(memory, dtype=request.param)


def run_lockstep(
    memory: Memory,
    backend: DeltaMemory,
    actions: list[Action],
    start: MemoryState | None = None,
) -> tuple[MemoryState, DeltaState, list[Event]]:
    """Step oracle and backend together, checking every transition exactly.

    Each step must match in event and work, decode to the oracle's state,
    and leave zero residue before thresholding.
    """
    oracle_state = memory.initial_state() if start is None else start
    backend_state = backend.encode(oracle_state)
    events = []
    for index, action in enumerate(actions):
        expected = memory.step(oracle_state, action)
        actual = backend.step(backend_state, action)
        where = f"first mismatch at action {index}: {action!r}"
        assert actual.event == expected.event, where
        assert actual.work == expected.work, where
        assert backend.decode(actual.next_state) == expected.next_state, where
        assert backend.residue(actual.next_state) == 0.0, where
        oracle_state, backend_state = expected.next_state, actual.next_state
        events.append(actual.event)
    return oracle_state, backend_state, events


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


# E00 workloads, each checked at every step


@pytest.mark.parametrize("seed", range(3))
def test_matches_oracle_on_random_actions(
    memory: Memory, backend: DeltaMemory, seed: int
) -> None:
    rng = random.Random(seed)
    actions = [random_action(rng) for _ in range(300)]
    actions.insert(rng.randrange(200, 300), Halt())
    run_lockstep(memory, backend, actions)


def test_every_value_and_address(memory: Memory, backend: DeltaMemory) -> None:
    actions: list[Action] = []
    for value in range(MAX_WORD + 1):
        actions += [Write(value % CAPACITY, value), Read(value % CAPACITY)]
    run_lockstep(memory, backend, actions)


def test_repeated_overwrite_of_one_slot(memory: Memory, backend: DeltaMemory) -> None:
    actions: list[Action] = []
    for value in range(MAX_WORD + 1):
        actions += [Write(3, value), Read(3)]
    run_lockstep(memory, backend, actions)


def test_alternating_addresses(memory: Memory, backend: DeltaMemory) -> None:
    actions: list[Action] = []
    for step in range(200):
        address = 0 if step % 2 == 0 else CAPACITY - 1
        actions += [Write(address, step % (MAX_WORD + 1)), Read(CAPACITY - 1 - address)]
    run_lockstep(memory, backend, actions)


def test_fill_to_capacity_then_clear_a_region(
    memory: Memory, backend: DeltaMemory
) -> None:
    fill: list[Action] = [Write(a, MAX_WORD - a) for a in range(CAPACITY)]
    clear: list[Action] = [Write(a, 0) for a in range(4, 8)]
    reads: list[Action] = [Read(a) for a in range(CAPACITY)]
    run_lockstep(memory, backend, fill + reads + clear + reads)


def test_long_idle_retention(memory: Memory, backend: DeltaMemory) -> None:
    fill: list[Action] = [Write(a, (a * 37) % (MAX_WORD + 1)) for a in range(CAPACITY)]
    reads: list[Action] = [Read(a) for a in range(CAPACITY)]
    run_lockstep(memory, backend, fill + [Idle()] * 1000 + reads)


@pytest.mark.parametrize(
    "action",
    [Read(-1), Read(CAPACITY), Write(-CAPACITY, 1), Write(0, MAX_WORD + 1)],
    ids=repr,
)
def test_faults_skip_the_update(
    memory: Memory, backend: DeltaMemory, action: Action
) -> None:
    _, state, _ = run_lockstep(memory, backend, [Write(0, 9), Write(CAPACITY - 1, 9)])
    transition = backend.step(state, action)
    assert transition.event.kind is EventKind.FAULT
    assert transition.next_state.matrix is state.matrix


def test_halt_then_every_action_faults(memory: Memory, backend: DeltaMemory) -> None:
    after_halt: list[Action] = [Read(2), Write(2, 6), Idle(), Halt(), Read(CAPACITY)]
    _, _, events = run_lockstep(memory, backend, [Write(2, 5), Halt(), *after_halt])
    halted = Event(EventKind.FAULT, reason=FaultReason.HALTED)
    assert events[1:] == [Event(EventKind.HALT)] + [halted] * len(after_halt)


# Lowering, snapshots, and decoding


def test_lowering_keeps_gates_at_endpoints(backend: DeltaMemory) -> None:
    for action in [Write(5, 200), Read(5), Idle(), Halt()]:
        inputs = backend.lower(action)
        assert inputs.alpha == 1.0
        assert inputs.beta == (1.0 if isinstance(action, Write) else 0.0)
        assert inputs.key.sum().item() == inputs.beta  # one-hot key only on writes


def test_encode_decode_round_trip(memory: Memory, backend: DeltaMemory) -> None:
    rng = random.Random(0)
    for halted in (False, True):
        words = tuple(rng.randrange(MAX_WORD + 1) for _ in range(CAPACITY))
        state = backend.encode(MemoryState(words, halted))
        assert backend.decode(state) == MemoryState(words, halted)
        assert backend.residue(state) == 0.0


def test_resume_from_snapshot_matches_uninterrupted_run(
    memory: Memory, backend: DeltaMemory
) -> None:
    rng = random.Random(1)
    actions = [random_action(rng) for _ in range(200)]
    full_state, _, full_events = run_lockstep(memory, backend, actions)

    _, middle, first = run_lockstep(memory, backend, actions[:100])
    snapshot = json.loads(json.dumps(memory.snapshot(backend.decode(middle))))
    restored = memory.restore(snapshot)
    end, _, second = run_lockstep(memory, backend, actions[100:], start=restored)
    assert end == full_state
    assert first + second == full_events


def test_residue_exposes_margin_that_thresholding_hides(backend: DeltaMemory) -> None:
    state = backend.initial_state()
    matrix = state.matrix.clone()
    matrix[0, 0] = 0.75
    perturbed = DeltaState(matrix)
    assert backend.decode(perturbed).words[0] == 1
    assert backend.residue(perturbed) == pytest.approx(0.25)


def test_decode_rejects_non_finite_state(backend: DeltaMemory) -> None:
    matrix = backend.initial_state().matrix.clone()
    matrix[1, 2] = float("nan")
    with pytest.raises(ValueError):
        backend.decode(DeltaState(matrix))


def test_rejects_state_of_wrong_shape(memory: Memory) -> None:
    backend = DeltaMemory(memory)
    wrong = DeltaState(torch.zeros(WORD_BITS, CAPACITY + 1))
    with pytest.raises(ValueError):
        backend.step(wrong, Idle())


def test_rejects_unknown_action(memory: Memory) -> None:
    backend = DeltaMemory(memory)
    with pytest.raises(TypeError):
        backend.step(backend.initial_state(), "read")  # type: ignore[arg-type]
