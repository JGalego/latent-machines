"""Online linear regression: E07's non-one-hot workload.

A fixed random map ``A`` (values x keys, entries N(0, 1)) generates samples
``y = A x + noise`` with ``x ~ N(0, I / keys)``, so ``|x|^2`` averages 1, and
noise N(0, ``noise``^2) per entry. Keys are dense, so writes interfere; the
task measures adaptation, not storage. ``(keys, values, seed)`` fixes the
stream exactly.
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Regression:
    keys: int
    values: int
    seed: int
    noise: float = 0.01

    def target_map(self) -> list[list[float]]:
        rng = random.Random(f"{self.seed}:regression-map")
        return [[rng.gauss(0, 1) for _ in range(self.keys)] for _ in range(self.values)]

    def samples(self) -> Iterator[tuple[list[float], list[float]]]:
        a = self.target_map()
        rng = random.Random(f"{self.seed}:regression-samples")
        scale = self.keys**-0.5
        while True:
            x = [rng.gauss(0, scale) for _ in range(self.keys)]
            y = [
                sum(w * xi for w, xi in zip(row, x, strict=True))
                + rng.gauss(0, self.noise)
                for row in a
            ]
            yield x, y
