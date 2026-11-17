"""Deterministic ``memory/v1`` action streams for the E00 workloads.

Each workload is an endless generator; ``actions`` takes its first
``horizon`` items. A (workload, machine, horizon, seed) tuple therefore fixes
the input exactly.
"""

import random
from collections.abc import Callable, Iterator
from itertools import islice

from latent_machines.machines.memory import Action, Idle, Memory, Read, Write

type Workload = Callable[[Memory, random.Random], Iterator[Action]]

IDLE_STRETCH = 1000  # idle steps between full reads in the idle workload


def _word(machine: Memory, rng: random.Random) -> int:
    return rng.randrange(machine.max_word + 1)


def _fill(machine: Memory, rng: random.Random) -> Iterator[Action]:
    for address in range(machine.capacity):
        yield Write(address, _word(machine, rng))


def _read_all(machine: Memory) -> Iterator[Action]:
    for address in range(machine.capacity):
        yield Read(address)


def overwrite(machine: Memory, rng: random.Random) -> Iterator[Action]:
    """Write one slot over and over, reading it back after each write."""
    slot = rng.randrange(machine.capacity)
    while True:
        yield Write(slot, _word(machine, rng))
        yield Read(slot)


def alternate(machine: Memory, rng: random.Random) -> Iterator[Action]:
    """Alternate writes between the first and last slots, reading the other."""
    first, last = 0, machine.capacity - 1
    while True:
        for target, other in ((first, last), (last, first)):
            yield Write(target, _word(machine, rng))
            yield Read(other)


def untouched(machine: Memory, rng: random.Random) -> Iterator[Action]:
    """Fill memory, then write only the lower half while reading anywhere."""
    yield from _fill(machine, rng)
    hammered = max(1, machine.capacity // 2)
    while True:
        yield Write(rng.randrange(hammered), _word(machine, rng))
        yield Read(rng.randrange(machine.capacity))


def idle(machine: Memory, rng: random.Random) -> Iterator[Action]:
    """Fill memory, then idle in long stretches separated by full reads."""
    yield from _fill(machine, rng)
    while True:
        for _ in range(IDLE_STRETCH):
            yield Idle()
        yield from _read_all(machine)


def clear_region(machine: Memory, rng: random.Random) -> Iterator[Action]:
    """Fill memory, then repeatedly zero a random contiguous region and refill it."""
    yield from _fill(machine, rng)
    while True:
        start = rng.randrange(machine.capacity)
        stop = rng.randrange(start, machine.capacity) + 1
        for address in range(start, stop):
            yield Write(address, 0)
        yield from _read_all(machine)
        for address in range(start, stop):
            yield Write(address, _word(machine, rng))


def capacity(machine: Memory, rng: random.Random) -> Iterator[Action]:
    """Write every slot, then read every slot, in cycles."""
    while True:
        yield from _fill(machine, rng)
        yield from _read_all(machine)


def invalid(machine: Memory, rng: random.Random) -> Iterator[Action]:
    """Interleave valid writes and reads with out-of-range addresses and values."""
    bad_addresses = (-1, -machine.capacity, machine.capacity, machine.capacity + 7)
    bad_values = (-1, machine.max_word + 1)
    while True:
        address = rng.randrange(machine.capacity)
        yield Write(address, _word(machine, rng))
        yield rng.choice(
            [
                Read(rng.choice(bad_addresses)),
                Write(rng.choice(bad_addresses), _word(machine, rng)),
                Write(address, rng.choice(bad_values)),
            ]
        )
        yield Read(address)


def random_mix(machine: Memory, rng: random.Random) -> Iterator[Action]:
    """Random reads, writes, and idles, with arguments sometimes just out of range."""
    while True:
        address = rng.randrange(-2, machine.capacity + 2)
        roll = rng.random()
        if roll < 0.45:
            yield Read(address)
        elif roll < 0.9:
            yield Write(address, rng.randrange(-2, machine.max_word + 3))
        else:
            yield Idle()


WORKLOADS: dict[str, Workload] = {
    "overwrite": overwrite,
    "alternate": alternate,
    "untouched": untouched,
    "idle": idle,
    "clear_region": clear_region,
    "capacity": capacity,
    "invalid": invalid,
    "random_mix": random_mix,
}


def actions(
    name: str, machine: Memory, horizon: int, rng: random.Random
) -> Iterator[Action]:
    """The first ``horizon`` actions of workload ``name``."""
    return islice(WORKLOADS[name](machine, rng), horizon)
