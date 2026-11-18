"""Tensor implementation of ``optimizer-memory/v1`` for E07.

The specification and the float64 reference are in
``machines/optimizer_memory.py``. This backend is written separately: the
linear model's direction is the closed form ``(y - W k) k^T``, and the MLP's
comes from PyTorch autograd rather than hand-written backpropagation. Only
the specification's initial parameters are shared.

All parameters, momentum buffers, and the chunk-start copy are held in the
chosen dtype; Python scalars (``lr``, ``momentum``, ``decay``) are applied
by PyTorch's ordinary type promotion, so a ``decay`` that the dtype cannot
resolve behaves as the dtype dictates. ``readout``, ``snapshot``, and
``restore`` exist for conformance checks; a controller must not observe them.
"""

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F

from latent_machines.machines.events import Event, EventKind
from latent_machines.machines.memory import (
    Action,
    FaultReason,
    Halt,
    Idle,
    Read,
    Write,
)
from latent_machines.machines.optimizer_memory import (
    NORM_EPS,
    VERSION,
    WORK_PER_ACTION,
    Optimizer,
    initial_params,
    shapes,
)

type Tensors = dict[str, torch.Tensor]


@dataclass(frozen=True)
class TensorState:
    params: Tensors
    momentum: Tensors
    chunk_start: Tensors | None
    steps: int = 0
    halted: bool = False


@dataclass(frozen=True)
class TensorTransition:
    next_state: TensorState
    event: Event
    output: torch.Tensor | None
    work: int


def _forward(params: Tensors, keys: torch.Tensor, model: str) -> torch.Tensor:
    """``f(k)`` for keys of shape ``(..., keys)``."""
    if model == "linear":
        return keys @ params["W"].T
    hidden = F.relu(keys @ params["W1"].T + params["b1"])
    return hidden @ params["W2"].T + params["b2"]


@dataclass(frozen=True)
class OptimizerMemory:
    keys: int
    values: int
    optimizer: Optimizer
    dtype: torch.dtype = torch.float32

    def initial_state(self) -> TensorState:
        flat = initial_params(self.keys, self.values, self.optimizer)
        params = {
            name: torch.tensor(flat[name], dtype=self.dtype).reshape(shape)
            for name, shape in shapes(self.keys, self.values, self.optimizer).items()
        }
        momentum = {n: torch.zeros_like(p) for n, p in params.items()}
        chunk = dict(params) if self.optimizer.chunk > 1 else None
        return TensorState(params, momentum, chunk)

    def predict(self, state: TensorState, key: torch.Tensor) -> torch.Tensor:
        return _forward(state.params, key, self.optimizer.model)

    def readout(self, state: TensorState) -> torch.Tensor:
        """``f(e_a)`` for every one-hot key, as columns ``(values, keys)``."""
        eye = torch.eye(self.keys, dtype=self.dtype)
        return _forward(state.params, eye, self.optimizer.model).T

    def direction(
        self, params: Tensors, key: torch.Tensor, target: torch.Tensor
    ) -> Tensors:
        if self.optimizer.model == "linear":
            return {"W": torch.outer(target - params["W"] @ key, key)}
        with torch.enable_grad():
            leaves = {n: p.detach().requires_grad_(True) for n, p in params.items()}
            loss = 0.5 * (_forward(leaves, key, "mlp") - target).pow(2).sum()
            grads = torch.autograd.grad(loss, list(leaves.values()))
        return {n: -g for n, g in zip(leaves, grads, strict=True)}

    def update(
        self,
        state: TensorState,
        key: torch.Tensor | None,
        target: torch.Tensor | None,
    ) -> TensorState:
        opt = self.optimizer
        chunk_start = state.chunk_start
        if opt.chunk > 1 and state.steps % opt.chunk == 0:
            chunk_start = dict(state.params)
        if key is None or target is None:
            g = {n: torch.zeros_like(p) for n, p in state.params.items()}
        else:
            at = chunk_start if opt.chunk > 1 and chunk_start else state.params
            g = self.direction(at, key, target)
        if opt.normalize:
            norm = torch.sqrt(sum((x * x).sum() for x in g.values()))
            g = {n: x / torch.clamp(norm, min=NORM_EPS) for n, x in g.items()}
        momentum = {n: opt.momentum * state.momentum[n] + g[n] for n in g}
        params = {
            n: (1 - opt.decay) * state.params[n] + opt.lr * momentum[n] for n in g
        }
        return TensorState(params, momentum, chunk_start, state.steps + 1, state.halted)

    # ---------- memory/v1 actions ----------

    def one_hot(self, address: int) -> torch.Tensor:
        vector = torch.zeros(self.keys, dtype=self.dtype)
        vector[address] = 1
        return vector

    def bits(self, word: int) -> torch.Tensor:
        return torch.tensor(
            [(word >> i) & 1 for i in range(self.values)], dtype=self.dtype
        )

    def step(self, state: TensorState, action: Action) -> TensorTransition:
        if not isinstance(action, Read | Write | Idle | Halt):
            raise TypeError(f"not a memory/v1 action: {action!r}")
        reason = None
        if state.halted:
            reason = FaultReason.HALTED
        elif isinstance(action, Read | Write) and not 0 <= action.address < self.keys:
            reason = FaultReason.ADDRESS_OUT_OF_RANGE
        elif isinstance(action, Write) and not 0 <= action.value < 2**self.values:
            reason = FaultReason.VALUE_OUT_OF_RANGE
        if reason is not None:
            fault = Event(EventKind.FAULT, reason=reason)
            return TensorTransition(state, fault, None, WORK_PER_ACTION)
        if isinstance(action, Write):
            nxt = self.update(
                state, self.one_hot(action.address), self.bits(action.value)
            )
            return TensorTransition(nxt, Event(EventKind.CONTINUE), None, 1)
        nxt = self.update(state, None, None)
        if isinstance(action, Read):
            out = self.predict(nxt, self.one_hot(action.address))
            event = Event(EventKind.OUTPUT, value=decode(out))
            return TensorTransition(nxt, event, out, WORK_PER_ACTION)
        if isinstance(action, Halt):
            return TensorTransition(
                replace(nxt, halted=True), Event(EventKind.HALT), None, 1
            )
        return TensorTransition(nxt, Event(EventKind.CONTINUE), None, 1)

    # ---------- conformance and snapshots ----------

    def snapshot(self, state: TensorState) -> dict[str, Any]:
        """Full optimizer state as JSON-compatible floats (exact for these dtypes)."""

        def flat(tensors: Tensors | None) -> dict[str, list[float]] | None:
            if tensors is None:
                return None
            return {n: t.reshape(-1).tolist() for n, t in tensors.items()}

        return {
            "version": VERSION,
            "keys": self.keys,
            "values": self.values,
            "params": flat(state.params),
            "momentum": flat(state.momentum),
            "chunk_start": flat(state.chunk_start),
            "steps": state.steps,
            "halted": state.halted,
        }

    def restore(self, snapshot: dict[str, Any]) -> TensorState:
        if (snapshot["version"], snapshot["keys"], snapshot["values"]) != (
            VERSION,
            self.keys,
            self.values,
        ):
            raise ValueError("snapshot is for a different memory")
        layout = shapes(self.keys, self.values, self.optimizer)

        def tensors(flat: dict[str, list[float]] | None) -> Tensors | None:
            if flat is None:
                return None
            if set(flat) != set(layout):
                raise ValueError(f"snapshot parameters are {sorted(flat)}")
            return {
                n: torch.tensor(flat[n], dtype=self.dtype).reshape(layout[n])
                for n in layout
            }

        params, momentum = tensors(snapshot["params"]), tensors(snapshot["momentum"])
        assert params is not None and momentum is not None
        return TensorState(
            params,
            momentum,
            tensors(snapshot["chunk_start"]),
            int(snapshot["steps"]),
            bool(snapshot["halted"]),
        )

    def from_words(self, words: tuple[int, ...], steps: int) -> TensorState:
        """A linear state holding ``words`` exactly, with empty optimizer buffers.

        This is what a ``memory/v1`` snapshot can restore: the canonical words,
        not the momentum, the chunk-start copy, or any residue.
        """
        if self.optimizer.model != "linear":
            raise ValueError("canonical words define only a linear state")
        matrix = torch.stack([self.bits(w) for w in words], dim=1)
        momentum = {"W": torch.zeros_like(matrix)}
        chunk = {"W": matrix} if self.optimizer.chunk > 1 else None
        return TensorState({"W": matrix}, momentum, chunk, steps)

    def state_bytes(self, state: TensorState) -> dict[str, int]:
        def size(tensors: Tensors | None) -> int:
            if not tensors:
                return 0
            return sum(t.numel() * t.element_size() for t in tensors.values())

        params = size(state.params)
        buffers = (size(state.momentum) if self.optimizer.momentum else 0) + size(
            state.chunk_start
        )
        return {"parameters": params, "optimizer_buffers": buffers}


def decode(vector: torch.Tensor) -> int:
    """Threshold each entry at 0.5; NaN decodes as 0, as in the reference."""
    return sum(1 << i for i, bit in enumerate((vector > 0.5).tolist()) if bit)
