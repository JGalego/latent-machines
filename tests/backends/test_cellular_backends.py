"""Tensor ``cellular/v1`` backends against the oracle, step by step."""

import random

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends.cellular import (  # noqa: E402
    ConvLookupCellular,
    TorchIntCellular,
)
from latent_machines.machines.cellular import Advance  # noqa: E402
from latent_machines.tasks import cellular_workloads  # noqa: E402

DTYPES = ("float32", "float64", "bfloat16", "float16")
torch.set_num_threads(1)  # tiny grids; avoid thread overhead


def backends(machine):
    yield TorchIntCellular(machine)
    for dtype in DTYPES:
        yield ConvLookupCellular(machine, getattr(torch, dtype))


@pytest.mark.parametrize("rule", ["B3/S23", "B36/S23", "B2/S"])
@pytest.mark.parametrize("workload", list(cellular_workloads.WORKLOADS))
def test_backends_match_oracle_every_step(rule: str, workload: str) -> None:
    rng = random.Random(f"test:{rule}:{workload}")
    machine = cellular_workloads.machine(rule, workload, 12, 12, rng)
    actions = list(cellular_workloads.actions(machine, 25, rng))
    for backend in backends(machine):
        oracle, state = machine.initial_state(), backend.initial_state()
        for action in actions:
            expected = machine.step(oracle, action)
            actual = backend.step(state, action)
            assert actual.event == expected.event
            assert actual.work == expected.work
            assert backend.decode(actual.next_state) == expected.next_state
            assert backend.residue(actual.next_state) == 0.0
            oracle, state = expected.next_state, actual.next_state


def test_population_above_256_is_counted_exactly_in_bfloat16() -> None:
    rng = random.Random(0)
    machine = cellular_workloads.machine("B2/S", "soup15-torus", 40, 40, rng)
    backend = ConvLookupCellular(machine, torch.bfloat16)
    expected = machine.step(machine.initial_state(), Advance())
    actual = backend.step(backend.initial_state(), Advance())
    assert expected.event.value > 256
    assert actual.event == expected.event


def test_circuit_is_small_and_fixed() -> None:
    rng = random.Random(0)
    machine = cellular_workloads.machine("B3/S23", "soup50-zero", 8, 8, rng)
    stats = ConvLookupCellular(machine).circuit_stats()
    assert stats["stored_weights"] == 9 + 9 + 9 + 9
    assert stats["nonzero_weights"] == 8 + 8 + 2 + 1  # kernel, counts 1-8, S, B


def test_rejects_wrong_shape() -> None:
    rng = random.Random(0)
    machine = cellular_workloads.machine("B3/S23", "soup50-zero", 8, 8, rng)
    backend = ConvLookupCellular(machine)
    state = backend.initial_state()
    bad = type(state)(state.cells[:4], 0)
    with pytest.raises(ValueError):
        backend.step(bad, Advance())
