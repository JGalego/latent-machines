"""The chunked delta-rule backend agrees with the recurrent update and the oracle."""

import random

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends import delta  # noqa: E402
from latent_machines.backends import delta_chunked as chunked  # noqa: E402
from latent_machines.machines.memory import (  # noqa: E402
    Event,
    EventKind,
    FaultReason,
    Halt,
    Idle,
    Memory,
    Read,
    Write,
)
from latent_machines.tasks.memory_workloads import WORKLOADS, actions  # noqa: E402


def recurrent_reference(state, keys, values, beta, queries):
    """Per-step delta rule from ``delta.py``: outputs and the final state."""
    outputs = []
    for key, value, b, query in zip(keys, values, beta, queries, strict=True):
        state = delta.delta_step(
            state, delta.DeltaInputs(key, value, 1.0, b.item(), None)
        )
        outputs.append(state @ query)
    return torch.stack(outputs), state


@pytest.mark.parametrize("chunk", [1, 3, 8, 20])
def test_chunk_matches_recurrent_update_for_general_keys(chunk: int) -> None:
    torch.manual_seed(chunk)
    steps, capacity, bits = 20, 6, 4
    keys = torch.nn.functional.normalize(
        torch.randn(steps, capacity, dtype=torch.float64), dim=-1
    )
    queries = torch.nn.functional.normalize(
        torch.randn(steps, capacity, dtype=torch.float64), dim=-1
    )
    values = torch.randn(steps, bits, dtype=torch.float64)
    beta = torch.rand(steps, dtype=torch.float64)
    start = torch.randn(bits, capacity, dtype=torch.float64)

    want_out, want_state = recurrent_reference(start, keys, values, beta, queries)
    state, outputs = start, []
    for i in range(0, steps, chunk):
        s = slice(i, i + chunk)
        out, state = chunked.chunk_update(
            state, keys[s], values[s], beta[s], queries[s]
        )
        outputs.append(out)
    assert torch.allclose(torch.cat(outputs), want_out, atol=1e-10)
    assert torch.allclose(state, want_state, atol=1e-10)


@pytest.mark.parametrize("workload", sorted(WORKLOADS))
@pytest.mark.parametrize("chunk", [4, 16, 64])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_one_hot_workloads_match_the_oracle(workload, chunk, dtype) -> None:
    machine = Memory(capacity=12, word_bits=8)
    sequence = list(actions(workload, machine, 400, random.Random(f"0:{workload}")))
    lowered = chunked.lower_sequence(machine, sequence)
    backend = chunked.ChunkedDeltaMemory(machine, chunk, dtype)
    oracle = machine.initial_state()
    for piece in backend.run(lowered):
        for event, i in zip(
            backend.events(lowered, piece), range(piece.start, piece.stop), strict=True
        ):
            expected = machine.step(oracle, sequence[i])
            oracle = expected.next_state
            assert event == expected.event
        assert backend.decode(piece.state) == oracle.words


def test_chunked_state_matches_recurrent_backend_at_chunk_ends() -> None:
    machine = Memory(capacity=8, word_bits=8)
    sequence = list(actions("random_mix", machine, 200, random.Random("1:random_mix")))
    recurrent = delta.DeltaMemory(machine, torch.float32)
    state = recurrent.initial_state()
    states = []
    for action in sequence:
        state = recurrent.step(state, action).next_state
        states.append(state.matrix)
    backend = chunked.ChunkedDeltaMemory(machine, 16, torch.float32)
    for piece in backend.run(chunked.lower_sequence(machine, sequence)):
        assert torch.equal(piece.state, states[piece.stop - 1])


def test_lowering_validates_before_any_effect() -> None:
    machine = Memory(capacity=4, word_bits=4)
    lowered = chunked.lower_sequence(
        machine,
        [
            Write(9, 1),
            Write(1, 99),
            Read(-1),
            Write(1, 5),
            Idle(),
            Read(1),
            Halt(),
            Write(2, 3),
        ],
    )
    assert lowered.key == [-1, -1, -1, 1, -1, -1, -1, -1]
    assert lowered.query == [-1, -1, -1, -1, -1, 1, -1, -1]
    assert lowered.host_events[0] == Event(
        EventKind.FAULT, reason=FaultReason.ADDRESS_OUT_OF_RANGE
    )
    assert lowered.host_events[1] == Event(
        EventKind.FAULT, reason=FaultReason.VALUE_OUT_OF_RANGE
    )
    assert lowered.host_events[5] is None
    assert lowered.host_events[6] == Event(EventKind.HALT)
    assert lowered.host_events[7] == Event(EventKind.FAULT, reason=FaultReason.HALTED)


def test_residue_measures_distance_from_zero_and_one() -> None:
    assert chunked.residue(torch.tensor([0.0, 1.0])) == 0.0
    assert chunked.residue(torch.tensor([0.0, 0.75])) == pytest.approx(0.25)
