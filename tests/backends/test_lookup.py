"""Stepwise conformance of the ReLU lookup-circuit transducer."""

import json
import random

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends.lookup import LookupState, LookupTransducer  # noqa: E402
from latent_machines.machines.transducer import (  # noqa: E402
    Feed,
    Transducer,
    TransducerState,
    modular_counter,
    parity,
    permutation_tracker,
)

DTYPES = [torch.float32, torch.float64, torch.bfloat16, torch.float16]
MACHINES = {
    "parity": parity(),
    "mod5": modular_counter(5, 3),
    "s3": permutation_tracker(3),
}


@pytest.fixture(params=DTYPES, ids=str)
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


@pytest.fixture(params=MACHINES.values(), ids=MACHINES.keys())
def machine(request: pytest.FixtureRequest) -> Transducer:
    return request.param


def run_lockstep(
    machine: Transducer, backend: LookupTransducer, symbols: list[int]
) -> LookupState:
    expected_state = machine.initial_state()
    state = backend.initial_state()
    for index, symbol in enumerate(symbols):
        expected = machine.step(expected_state, Feed(symbol))
        actual = backend.step(state, Feed(symbol))
        where = f"first mismatch at action {index}: symbol {symbol}"
        assert actual.event == expected.event, where
        assert actual.work == expected.work, where
        assert backend.decode(actual.next_state) == expected.next_state, where
        assert backend.residue(actual.next_state) == 0.0, where
        expected_state, state = expected.next_state, actual.next_state
    return state


def test_matches_oracle_on_random_symbols(
    machine: Transducer, dtype: torch.dtype
) -> None:
    rng = random.Random(0)
    symbols = [rng.randrange(-1, machine.alphabet_size + 1) for _ in range(500)]
    run_lockstep(machine, LookupTransducer(machine, dtype), symbols)


def test_every_state_and_symbol_pair(machine: Transducer, dtype: torch.dtype) -> None:
    backend = LookupTransducer(machine, dtype)
    for state in range(machine.num_states):
        for symbol in range(machine.alphabet_size):
            expected = machine.step(TransducerState(state), Feed(symbol))
            actual = backend.step(backend.encode(TransducerState(state)), Feed(symbol))
            assert actual.event == expected.event
            assert backend.decode(actual.next_state) == expected.next_state


def test_fault_skips_the_circuit(dtype: torch.dtype) -> None:
    backend = LookupTransducer(parity(), dtype)
    state = backend.initial_state()
    assert backend.step(state, Feed(2)).next_state is state


def test_snapshot_resume_matches_uninterrupted_run(dtype: torch.dtype) -> None:
    machine = permutation_tracker(3)
    backend = LookupTransducer(machine, dtype)
    rng = random.Random(1)
    symbols = [rng.randrange(2) for _ in range(200)]
    full = run_lockstep(machine, backend, symbols)
    middle = run_lockstep(machine, backend, symbols[:100])
    data = json.loads(json.dumps(machine.snapshot(backend.decode(middle))))
    restored = backend.encode(machine.restore(data))
    assert torch.equal(restored.vector, middle.vector)
    state = restored
    for symbol in symbols[100:]:
        state = backend.step(state, Feed(symbol)).next_state
    assert torch.equal(state.vector, full.vector)


def test_circuit_stats_count_one_unit_per_pair() -> None:
    stats = LookupTransducer(permutation_tracker(3)).circuit_stats()
    pairs = 6 * 2
    assert stats["hidden_width"] == pairs
    assert stats["nonzero_weights"] == 4 * pairs  # 2 inputs, 1 next state, 1 output
    assert stats["depth"] == 2


def test_decode_rejects_vectors_that_are_not_one_hot() -> None:
    backend = LookupTransducer(parity())
    for vector in ([0.0, 0.0], [1.0, 1.0], [float("nan"), 1.0]):
        with pytest.raises(ValueError):
            backend.decode(LookupState(torch.tensor(vector)))


def test_residue_exposes_drift_that_decoding_hides() -> None:
    backend = LookupTransducer(parity())
    drifted = LookupState(torch.tensor([0.9, 0.0]))
    assert backend.decode(drifted) == TransducerState(0)
    assert backend.residue(drifted) == pytest.approx(0.1)
