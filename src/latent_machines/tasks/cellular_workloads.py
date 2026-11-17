"""Seeded ``cellular/v1`` starting grids and action streams for E06.

A workload fixes the boundary and the starting pattern:

- ``soup15-torus``: random soup at 15% density on a torus.
- ``soup50-zero``: random soup at 50% density with dead cells outside.
- ``gliders-torus``: four B3/S23 gliders heading across the wrap seams, plus
  a blinker straddling the corner, on a torus.
- ``edge-zero``: a 40% soup confined to the two outermost rings, so most
  activity touches the dead boundary.

The action stream is mostly ``Advance``; about 5% of actions are ``Set``
perturbations, and about 5% of those are invalid (a coordinate just outside
the grid or the value 2), so every fault kind occurs.
"""

import random
from collections.abc import Iterator
from itertools import islice

from latent_machines.machines.cellular import Advance, Cells, Cellular, Rule, Set

SET_RATE = 0.05
INVALID_RATE = 0.05
GLIDER = ((0, 1), (1, 2), (2, 0), (2, 1), (2, 2))  # moves down-right


def _soup(rng: random.Random, height: int, width: int, density: float) -> Cells:
    return tuple(
        tuple(int(rng.random() < density) for _ in range(width)) for _ in range(height)
    )


def _edge(rng: random.Random, height: int, width: int, density: float) -> Cells:
    def near_edge(r: int, c: int) -> bool:
        return min(r, c, height - 1 - r, width - 1 - c) < 2

    return tuple(
        tuple(int(near_edge(r, c) and rng.random() < density) for c in range(width))
        for r in range(height)
    )


def _gliders(height: int, width: int) -> Cells:
    if height < 12 or width < 12:
        raise ValueError("gliders-torus needs at least a 12 x 12 grid")
    grid = [[0] * width for _ in range(height)]
    for top, left, flip_r, flip_c in (
        (height - 4, width // 2, False, False),  # crosses the bottom seam
        (height // 2, width - 4, False, False),  # crosses the right seam
        (1, width // 4, True, False),  # heads up across the top seam
        (height // 4, 1, False, True),  # heads left across the left seam
    ):
        for dr, dc in GLIDER:
            r = top + (2 - dr if flip_r else dr)
            c = left + (2 - dc if flip_c else dc)
            grid[r % height][c % width] = 1
    for r, c in ((height - 1, 0), (0, 0), (1, 0)):  # blinker across the corner
        grid[r][c] = 1
    return tuple(map(tuple, grid))


WORKLOADS = {
    "soup15-torus": "torus",
    "soup50-zero": "zero",
    "gliders-torus": "torus",
    "edge-zero": "zero",
}


def machine(
    rule: str, workload: str, height: int, width: int, rng: random.Random
) -> Cellular:
    """The machine for ``workload``: its boundary and seeded starting grid."""
    if workload == "soup15-torus":
        initial = _soup(rng, height, width, 0.15)
    elif workload == "soup50-zero":
        initial = _soup(rng, height, width, 0.5)
    elif workload == "gliders-torus":
        initial = _gliders(height, width)
    elif workload == "edge-zero":
        initial = _edge(rng, height, width, 0.4)
    else:
        raise ValueError(f"unknown workload {workload!r}")
    return Cellular(Rule.parse(rule), height, width, WORKLOADS[workload], initial)


def stream(machine: Cellular, rng: random.Random) -> Iterator[Advance | Set]:
    while True:
        if rng.random() >= SET_RATE:
            yield Advance()
        elif rng.random() < INVALID_RATE:
            yield rng.choice(
                (
                    Set(machine.height, rng.randrange(machine.width), 1),
                    Set(rng.randrange(machine.height), -1, 1),
                    Set(0, 0, 2),
                )
            )
        else:
            yield Set(
                rng.randrange(machine.height),
                rng.randrange(machine.width),
                rng.randrange(2),
            )


def actions(
    machine: Cellular, horizon: int, rng: random.Random
) -> Iterator[Advance | Set]:
    return islice(stream(machine, rng), horizon)
