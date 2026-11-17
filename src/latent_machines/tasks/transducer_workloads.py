"""Deterministic ``transducer/v1`` symbol streams for E01."""

import random
from collections.abc import Iterator
from itertools import islice

from latent_machines.machines.transducer import Feed, Transducer

INVALID_RATE = 0.05


def symbols(machine: Transducer, rng: random.Random) -> Iterator[Feed]:
    """Random in-range symbols; about 5% are just outside the alphabet."""
    while True:
        if rng.random() < INVALID_RATE:
            yield Feed(rng.choice((-1, machine.alphabet_size)))
        else:
            yield Feed(rng.randrange(machine.alphabet_size))


def actions(machine: Transducer, horizon: int, rng: random.Random) -> Iterator[Feed]:
    return islice(symbols(machine, rng), horizon)
