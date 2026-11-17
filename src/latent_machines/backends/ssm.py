"""Tensor implementations of ``lds/v1`` for E06's numerical branch.

``DirectLDS`` is the conventional recurrence ``y = C x + D u``,
``x <- A x + B u`` with dense matrices, evaluated in a chosen dtype.

``DiagonalSSM`` is the same system as a diagonal state-space model. ``A`` is
diagonalized once in float64, ``A = V diag(lambda) V^-1``, giving complex
modes ``z = V^-1 x`` with

    z <- lambda * z + b u,    y = Re(c . z) + D u,    b = V^-1 B,  c = C V.

The modes, ``b``, and ``c`` are cast to the run dtype, and complex
arithmetic is done on separate real and imaginary tensors, so bfloat16
works. Two schedules:

- ``sequential``: one sample at a time.
- ``chunked``: chunks of ``chunk`` samples. Within a chunk the states are
  ``z_j = lambda^j z_0 + sum_{i<j} lambda^{j-1-i} b u_i``, computed with a
  per-mode lower-triangular kernel; only the last state passes to the next
  chunk. The powers are computed in float64 and then cast.

Neither is exact. Each is measured against ``machines.lds.reference`` under
a declared tolerance. ``snapshot`` and ``restore`` cover the state between
samples (between chunks for ``chunked``).
"""

from dataclasses import dataclass
from typing import Any

import torch

from latent_machines.machines.lds import LinearSystem


@dataclass
class Modes:
    re: torch.Tensor
    im: torch.Tensor


def _complex_mul(ar, ai, br, bi):
    return ar * br - ai * bi, ar * bi + ai * br


class DirectLDS:
    """Dense state-space recurrence in ``dtype``: the conventional baseline."""

    schedule = "sequential"

    def __init__(self, system: LinearSystem, dtype: torch.dtype) -> None:
        self.system, self.dtype = system, dtype
        self.a = torch.tensor(system.a, dtype=torch.float64).to(dtype)
        self.b = torch.tensor(system.b, dtype=torch.float64).to(dtype)
        self.c = torch.tensor(system.c, dtype=torch.float64).to(dtype)
        self.d = torch.tensor(system.d, dtype=torch.float64).to(dtype)
        self.state = torch.zeros(system.dimension, dtype=dtype)

    @property
    def state_bytes(self) -> int:
        return self.state.numel() * self.state.element_size()

    @property
    def cast_spectral_radius(self) -> float:
        """Spectral radius of ``A`` after rounding to the run dtype."""
        return torch.linalg.eigvals(self.a.to(torch.float64)).abs().max().item()

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.empty(len(inputs), dtype=self.dtype)
        x = self.state
        for t, u in enumerate(inputs.to(self.dtype)):
            outputs[t] = self.c @ x + self.d * u
            x = self.a @ x + self.b * u
        self.state = x
        return outputs

    def snapshot(self) -> dict[str, Any]:
        return {"x": self.state.to(torch.float64).tolist()}

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.state = torch.tensor(snapshot["x"], dtype=torch.float64).to(self.dtype)


class DiagonalSSM:
    """``lds/v1`` as a diagonal complex SSM, sequential or chunked."""

    def __init__(
        self,
        system: LinearSystem,
        dtype: torch.dtype,
        schedule: str = "sequential",
        chunk: int = 64,
    ) -> None:
        if schedule not in ("sequential", "chunked"):
            raise ValueError(f"unknown schedule {schedule!r}")
        self.system, self.dtype, self.schedule, self.chunk = (
            system,
            dtype,
            schedule,
            chunk,
        )
        a = torch.tensor(system.a, dtype=torch.float64)
        eigenvalues, vectors = torch.linalg.eig(a)
        b = torch.linalg.solve(vectors, torch.tensor(system.b, dtype=torch.complex128))
        c = torch.tensor(system.c, dtype=torch.complex128) @ vectors
        self.condition = torch.linalg.cond(vectors).item()
        self.lam = self._split(eigenvalues)
        self.b, self.c = self._split(b), self._split(c)
        self.d = torch.tensor(system.d, dtype=torch.float64).to(dtype)
        if schedule == "chunked":
            j = torch.arange(chunk + 1, dtype=torch.float64)
            powers = eigenvalues[:, None] ** j.to(torch.complex128)  # (modes, L+1)
            lag = j[:, None] - 1 - j[None, :chunk]  # (L+1, L): j - 1 - i
            kernel = eigenvalues[:, None, None] ** lag.clamp(min=0).to(torch.complex128)
            kernel = kernel * (lag >= 0)
            self.powers, self.kernel = self._split(powers), self._split(kernel)
        modes = len(eigenvalues)
        self.state = Modes(
            torch.zeros(modes, dtype=dtype), torch.zeros(modes, dtype=dtype)
        )

    def _split(self, z: torch.Tensor) -> Modes:
        return Modes(z.real.to(self.dtype), z.imag.to(self.dtype))

    @property
    def state_bytes(self) -> int:
        return 2 * self.state.re.numel() * self.state.re.element_size()

    @property
    def cast_spectral_radius(self) -> float:
        """Largest mode magnitude after rounding the modes to the run dtype."""
        re, im = self.lam.re.to(torch.float64), self.lam.im.to(torch.float64)
        return torch.sqrt(re * re + im * im).max().item()

    def run(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.to(self.dtype)
        if self.schedule == "sequential":
            return self._sequential(inputs)
        return torch.cat(
            [
                self._chunk(inputs[i : i + self.chunk])
                for i in range(0, len(inputs), self.chunk)
            ]
        )

    def _output(self, re: torch.Tensor, im: torch.Tensor, u: torch.Tensor):
        # Re(c . z) = sum(c_re z_re - c_im z_im)
        return (self.c.re * re - self.c.im * im).sum(-1) + self.d * u

    def _sequential(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.empty(len(inputs), dtype=self.dtype)
        re, im = self.state.re, self.state.im
        lam, b = self.lam, self.b
        for t, u in enumerate(inputs):
            outputs[t] = self._output(re, im, u)
            re, im = _complex_mul(lam.re, lam.im, re, im)
            re, im = re + b.re * u, im + b.im * u
        self.state = Modes(re, im)
        return outputs

    def _chunk(self, u: torch.Tensor) -> torch.Tensor:
        n = len(u)  # the last chunk may be shorter
        z0 = self.state
        p = Modes(self.powers.re[:, : n + 1], self.powers.im[:, : n + 1])
        k = Modes(self.kernel.re[:, : n + 1, :n], self.kernel.im[:, : n + 1, :n])
        # Free response lambda^j z0 for j = 0..L, shape (modes, L+1).
        free_re, free_im = _complex_mul(p.re, p.im, z0.re[:, None], z0.im[:, None])
        # Forced response sum_i K[j, i] (b u_i), shape (modes, L+1).
        bu_re, bu_im = self.b.re[:, None] * u, self.b.im[:, None] * u  # (modes, L)
        forced_re = k.re @ bu_re[:, :, None] - k.im @ bu_im[:, :, None]
        forced_im = k.re @ bu_im[:, :, None] + k.im @ bu_re[:, :, None]
        re = free_re + forced_re[..., 0]
        im = free_im + forced_im[..., 0]
        outputs = self._output(re[:, :-1].T, im[:, :-1].T, u)
        self.state = Modes(re[:, -1], im[:, -1])
        return outputs

    def snapshot(self) -> dict[str, Any]:
        return {
            "re": self.state.re.to(torch.float64).tolist(),
            "im": self.state.im.to(torch.float64).tolist(),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.state = Modes(
            torch.tensor(snapshot["re"], dtype=torch.float64).to(self.dtype),
            torch.tensor(snapshot["im"], dtype=torch.float64).to(self.dtype),
        )
