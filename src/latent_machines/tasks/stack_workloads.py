"""Deterministic ``stack/v1`` action streams for E01.

Each workload is an endless generator of actions; ``actions`` takes the
first ``horizon``.
"""

import random
from collections.abc import Callable, Iterator
from itertools import islice

from latent_machines.machines.stack import Pop, Push, Stack, StackAction
from latent_machines.tasks import brackets

type StackWorkload = Callable[[Stack, random.Random], Iterator[StackAction]]


def random_ops(machine: Stack, rng: random.Random) -> Iterator[StackAction]:
    """Random pushes and pops; a few pushed values are out of range."""
    while True:
        if rng.random() < 0.55:
            yield Push(rng.randrange(-1, machine.max_word + 2))
        else:
            yield Pop()


def boundaries(machine: Stack, rng: random.Random) -> Iterator[StackAction]:
    """Fill past capacity, then drain past empty, in cycles."""
    while True:
        for _ in range(machine.capacity + 1):
            yield Push(rng.randrange(machine.max_word + 1))
        for _ in range(machine.capacity + 1):
            yield Pop()


WORKLOADS: dict[str, StackWorkload] = {
    "brackets": brackets.stream,
    "random_ops": random_ops,
    "boundaries": boundaries,
}


def actions(
    name: str, machine: Stack, horizon: int, rng: random.Random
) -> Iterator[StackAction]:
    return islice(WORKLOADS[name](machine, rng), horizon)
