"""Linear dynamical systems: the numerical-branch reference for E06.

Specification ``lds/v1``: a single-input, single-output discrete-time
system with state ``x`` (dimension ``n``, zero initially) and matrices
``A`` (n x n), ``B`` (n), ``C`` (n), and scalar ``D``. For each input
sample ``u_t``:

    y_t     = C x_t + D u_t
    x_{t+1} = A x_t + B u_t

``reference`` evaluates this recurrence in IEEE double precision with plain
Python floats, in this order, as the conventional reference. It is not
exact arithmetic: it is the declared reference against which tensor
implementations are measured under an error tolerance, never classified as
exact.

``damped_oscillator``, ``butterworth_lowpass``, and ``random_stable`` build
the E06 systems; ``noise`` and ``step`` build input signals. This module has
no tensor dependency.
"""

import math
import random
from dataclasses import dataclass

VERSION = "lds/v1"

type Matrix = tuple[tuple[float, ...], ...]
type Vector = tuple[float, ...]


@dataclass(frozen=True)
class LinearSystem:
    name: str
    a: Matrix
    b: Vector
    c: Vector
    d: float = 0.0

    def __post_init__(self) -> None:
        n = len(self.a)
        if n < 1 or any(len(row) != n for row in self.a):
            raise ValueError("A must be square and non-empty")
        if len(self.b) != n or len(self.c) != n:
            raise ValueError(f"B and C must have length {n}")

    @property
    def dimension(self) -> int:
        return len(self.a)


def reference(system: LinearSystem, inputs: list[float]) -> list[float]:
    """Outputs of the recurrence in double precision, from a zero state."""
    a, b, c, d, n = system.a, system.b, system.c, system.d, system.dimension
    x = [0.0] * n
    outputs = []
    for u in inputs:
        outputs.append(sum(c[i] * x[i] for i in range(n)) + d * u)
        x = [sum(a[i][j] * x[j] for j in range(n)) + b[i] * u for i in range(n)]
    return outputs


def damped_oscillator(radius: float = 0.999, period: float = 100.0) -> LinearSystem:
    """A rotation by 2 pi / period, scaled by ``radius`` each sample."""
    theta = 2 * math.pi / period
    cos, sin = radius * math.cos(theta), radius * math.sin(theta)
    return LinearSystem(
        f"oscillator-r{radius}",
        ((cos, -sin), (sin, cos)),
        (1.0, 0.0),
        (1.0, 0.0),
    )


def butterworth_lowpass(cutoff: float = 0.05) -> LinearSystem:
    """Second-order Butterworth low-pass by the bilinear transform.

    ``cutoff`` is the -3 dB frequency in cycles per sample. The transfer
    function ``(b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)`` is
    realized in controllable canonical form.
    """
    k = math.tan(math.pi * cutoff)
    norm = 1 / (1 + math.sqrt(2) * k + k * k)
    b0 = k * k * norm
    b1, b2 = 2 * b0, b0
    a1 = 2 * (k * k - 1) * norm
    a2 = (1 - math.sqrt(2) * k + k * k) * norm
    return LinearSystem(
        f"butterworth-{cutoff}",
        ((-a1, -a2), (1.0, 0.0)),
        (1.0, 0.0),
        (b1 - b0 * a1, b2 - b0 * a2),
        b0,
    )


def random_stable(dimension: int = 8, seed: int = 0) -> LinearSystem:
    """A dense stable system: rotation blocks with radii 0.9-0.98, rotated.

    ``A = Q R Q^T`` for a random orthogonal ``Q`` (Gram-Schmidt) and a block
    diagonal ``R`` of scaled 2 x 2 rotations, so ``A`` is normal with
    spectral radius at most 0.98.
    """
    if dimension % 2:
        raise ValueError("dimension must be even")
    rng = random.Random(f"{seed}:random-stable:{dimension}")
    block = [[0.0] * dimension for _ in range(dimension)]
    for k in range(0, dimension, 2):
        radius, theta = rng.uniform(0.9, 0.98), rng.uniform(0.05, 1.5)
        cos, sin = radius * math.cos(theta), radius * math.sin(theta)
        block[k][k], block[k][k + 1] = cos, -sin
        block[k + 1][k], block[k + 1][k + 1] = sin, cos
    q = _orthogonal(dimension, rng)
    a = [
        [
            sum(
                q[i][k] * block[k][m] * q[j][m]
                for k in range(dimension)
                for m in range(dimension)
            )
            for j in range(dimension)
        ]
        for i in range(dimension)
    ]
    b = tuple(rng.gauss(0, 1) for _ in range(dimension))
    c = tuple(rng.gauss(0, 1) / math.sqrt(dimension) for _ in range(dimension))
    return LinearSystem(f"random-stable-{dimension}", tuple(map(tuple, a)), b, c)


def _orthogonal(n: int, rng: random.Random) -> list[list[float]]:
    """Columns of a random orthogonal matrix, by Gram-Schmidt."""
    columns: list[list[float]] = []
    while len(columns) < n:
        v = [rng.gauss(0, 1) for _ in range(n)]
        for u in columns:
            dot = sum(x * y for x, y in zip(u, v, strict=True))
            v = [x - dot * y for x, y in zip(v, u, strict=True)]
        norm = math.sqrt(sum(x * x for x in v))
        if norm > 1e-6:
            columns.append([x / norm for x in v])
    return [[columns[j][i] for j in range(n)] for i in range(n)]


def noise(length: int, seed: int) -> list[float]:
    rng = random.Random(f"{seed}:noise")
    return [rng.gauss(0, 1) for _ in range(length)]


def step(length: int) -> list[float]:
    return [1.0] * length


SYSTEMS = {
    "oscillator": damped_oscillator,
    "butterworth": butterworth_lowpass,
    "random-stable-8": random_stable,
}
INPUTS = ("noise", "step")
