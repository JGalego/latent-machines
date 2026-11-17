"""E00 workload streams: length, determinism, and intent."""

import random

import pytest

from latent_machines.machines.memory import Action, Idle, Memory, Read, Write
from latent_machines.tasks.memory_workloads import WORKLOADS, actions

MACHINE = Memory(capacity=16, word_bits=8)


def generate(name: str, horizon: int = 2000, seed: int = 0) -> list[Action]:
    return list(actions(name, MACHINE, horizon, random.Random(seed)))


def in_range(action: Action) -> bool:
    if isinstance(action, Read | Write) and not 0 <= action.address < 16:
        return False
    return not (isinstance(action, Write) and not 0 <= action.value <= 255)


@pytest.mark.parametrize("name", WORKLOADS)
def test_yields_exactly_horizon_actions_deterministically(name: str) -> None:
    first = generate(name, horizon=500, seed=1)
    assert len(first) == 500
    assert first == generate(name, horizon=500, seed=1)


@pytest.mark.parametrize("name", WORKLOADS)
def test_only_fault_workloads_leave_the_valid_range(name: str) -> None:
    valid = all(in_range(action) for action in generate(name))
    assert valid == (name not in {"invalid", "random_mix"})


def test_untouched_writes_only_the_lower_half_after_filling() -> None:
    writes = [a for a in generate("untouched")[16:] if isinstance(a, Write)]
    assert writes and all(a.address < 8 for a in writes)


def test_idle_is_mostly_idle() -> None:
    stream = generate("idle")
    assert sum(isinstance(a, Idle) for a in stream) > 0.9 * len(stream)


def test_capacity_writes_then_reads_every_slot() -> None:
    cycle = generate("capacity", horizon=32)
    assert [a.address for a in cycle if isinstance(a, Write)] == list(range(16))
    assert [a.address for a in cycle if isinstance(a, Read)] == list(range(16))


def test_clear_region_writes_zeros_after_filling() -> None:
    assert any(
        isinstance(a, Write) and a.value == 0 for a in generate("clear_region")[16:]
    )
