"""Optimizer-defined associative memory: the float64 reference for E07.

Specification ``optimizer-memory/v1``. A parametric map ``f_theta`` from keys
(length ``keys``) to values (length ``values``) is updated by one optimizer
step per action on the loss ``L = 1/2 |f_theta(k) - y|^2``:

- Models: ``linear``, ``f(k) = W k`` with ``W`` (values x keys) starting at
  zero; or ``mlp``, ``f(k) = W2 relu(W1 k + b1) + b2`` with ``hidden`` units,
  ``W1`` and ``W2`` drawn from N(0, 1/hidden) by ``random.Random(f"{seed}:mlp")``
  and zero biases.
- Step. For a sample ``(k, y)`` the update direction is the negative gradient
  ``g = -dL/dtheta``; a step without a sample has ``g = 0``. Then, in order:

      if chunk > 1 and steps % chunk == 0:  theta0 <- theta  (chunk start)
      g  <- -dL/dtheta evaluated at theta0 if chunk > 1, else at theta
      g  <- g / max(|g|, 1e-12)             if normalize (|g| over all parameters)
      m  <- momentum * m + g                (m starts at zero)
      theta <- (1 - decay) * theta + lr * m
      steps <- steps + 1

  ``relu`` has derivative 0 at 0. With one-hot keys, ``lr = 1``, and no other
  feature, the linear model's write is delta replacement: column ``k`` becomes
  ``y`` and every other column is unchanged.
- ``memory/v1`` actions: ``Write(a, v)`` is a step on the sample
  ``(onehot(a), bits(v))``, bits least significant first as 0.0 or 1.0.
  ``Read(a)``, ``Idle()``, and ``Halt()`` are steps without a sample, so decay
  and momentum still act. A read returns ``f(onehot(a))`` after its step and
  decodes it by thresholding each entry at 0.5. Validation is that of
  ``memory/v1`` and precedes any effect; a faulted action changes nothing,
  not even ``steps``. Every action costs one work unit.

This module is the independent reference: plain Python floats (IEEE
float64) and lists, written separately from the tensor backend. Whether an
optimizer setting behaves like random-access memory is an experimental
question; this specification only fixes what the optimizer does.
"""

import math
import random
from dataclasses import dataclass, replace
from typing import Any

from latent_machines.machines.events import Event, EventKind
from latent_machines.machines.memory import (
    Action,
    FaultReason,
    Halt,
    Idle,
    Memory,
    Read,
    Write,
)

VERSION = "optimizer-memory/v1"
WORK_PER_ACTION = 1
NORM_EPS = 1e-12
MODELS = ("linear", "mlp")

type Params = dict[str, list[float]]  # flat, row-major


@dataclass(frozen=True)
class Optimizer:
    lr: float = 1.0
    momentum: float = 0.0
    decay: float = 0.0
    chunk: int = 1
    normalize: bool = False
    model: str = "linear"
    hidden: int = 16
    seed: int = 0

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}")
        if self.chunk < 1 or self.hidden < 1:
            raise ValueError("chunk and hidden must be positive")


def shapes(keys: int, values: int, optimizer: Optimizer) -> dict[str, tuple[int, ...]]:
    """Parameter names and shapes, in a fixed order."""
    if optimizer.model == "linear":
        return {"W": (values, keys)}
    h = optimizer.hidden
    return {"W1": (h, keys), "b1": (h,), "W2": (values, h), "b2": (values,)}


def initial_params(keys: int, values: int, optimizer: Optimizer) -> Params:
    sizes = {n: math.prod(s) for n, s in shapes(keys, values, optimizer).items()}
    if optimizer.model == "linear":
        return {"W": [0.0] * sizes["W"]}
    rng = random.Random(f"{optimizer.seed}:mlp")
    scale = 1 / math.sqrt(optimizer.hidden)
    return {
        "W1": [rng.gauss(0.0, scale) for _ in range(sizes["W1"])],
        "b1": [0.0] * sizes["b1"],
        "W2": [rng.gauss(0.0, scale) for _ in range(sizes["W2"])],
        "b2": [0.0] * sizes["b2"],
    }


@dataclass(frozen=True)
class ReferenceState:
    params: Params
    momentum: Params
    chunk_start: Params | None
    steps: int = 0
    halted: bool = False


@dataclass(frozen=True)
class ReferenceTransition:
    next_state: ReferenceState
    event: Event
    output: list[float] | None  # the read's value vector before decoding
    work: int


def _matvec(m: list[float], rows: int, cols: int, x: list[float]) -> list[float]:
    return [sum(m[r * cols + c] * x[c] for c in range(cols)) for r in range(rows)]


def _outer(a: list[float], b: list[float]) -> list[float]:
    return [x * y for x in a for y in b]


def decode(vector: list[float]) -> int:
    return sum(1 << i for i, x in enumerate(vector) if x > 0.5)


@dataclass(frozen=True)
class OptimizerMemoryReference:
    keys: int
    values: int
    optimizer: Optimizer

    def initial_state(self) -> ReferenceState:
        params = initial_params(self.keys, self.values, self.optimizer)
        zeros = {n: [0.0] * len(p) for n, p in params.items()}
        chunk = dict(params) if self.optimizer.chunk > 1 else None
        return ReferenceState(params, zeros, chunk)

    def predict(self, params: Params, key: list[float]) -> list[float]:
        if self.optimizer.model == "linear":
            return _matvec(params["W"], self.values, self.keys, key)
        h = self.optimizer.hidden
        z = [
            a + b
            for a, b in zip(
                _matvec(params["W1"], h, self.keys, key), params["b1"], strict=True
            )
        ]
        hidden = [max(x, 0.0) for x in z]
        out = _matvec(params["W2"], self.values, h, hidden)
        return [a + b for a, b in zip(out, params["b2"], strict=True)]

    def direction(
        self, params: Params, key: list[float], target: list[float]
    ) -> Params:
        """The negative gradient of 1/2 |f(k) - y|^2 at ``params``."""
        error = [y - o for y, o in zip(target, self.predict(params, key), strict=True)]
        if self.optimizer.model == "linear":
            return {"W": _outer(error, key)}
        h = self.optimizer.hidden
        z = [
            a + b
            for a, b in zip(
                _matvec(params["W1"], h, self.keys, key), params["b1"], strict=True
            )
        ]
        hidden = [max(x, 0.0) for x in z]
        w2 = params["W2"]
        back = [
            sum(w2[r * h + j] * error[r] for r in range(self.values))
            if z[j] > 0
            else 0.0
            for j in range(h)
        ]
        return {
            "W1": _outer(back, key),
            "b1": back,
            "W2": _outer(error, hidden),
            "b2": error,
        }

    def update(
        self, state: ReferenceState, key: list[float] | None, target: list[float] | None
    ) -> ReferenceState:
        """One optimizer step on ``(key, target)``, or without a sample if None."""
        opt = self.optimizer
        chunk_start = state.chunk_start
        if opt.chunk > 1 and state.steps % opt.chunk == 0:
            chunk_start = dict(state.params)
        if key is None or target is None:
            g = {n: [0.0] * len(p) for n, p in state.params.items()}
        else:
            at = (
                chunk_start
                if opt.chunk > 1 and chunk_start is not None
                else state.params
            )
            g = self.direction(at, key, target)
        if opt.normalize:
            norm = math.sqrt(sum(x * x for p in g.values() for x in p))
            g = {n: [x / max(norm, NORM_EPS) for x in p] for n, p in g.items()}
        momentum = {
            n: [
                opt.momentum * m + x
                for m, x in zip(state.momentum[n], g[n], strict=True)
            ]
            for n in g
        }
        params = {
            n: [
                (1 - opt.decay) * p + opt.lr * m
                for p, m in zip(state.params[n], momentum[n], strict=True)
            ]
            for n in g
        }
        return ReferenceState(
            params, momentum, chunk_start, state.steps + 1, state.halted
        )

    # ---------- memory/v1 actions ----------

    def machine(self) -> Memory:
        return Memory(self.keys, self.values)

    def one_hot(self, address: int) -> list[float]:
        return [1.0 if i == address else 0.0 for i in range(self.keys)]

    def bits(self, word: int) -> list[float]:
        return [float((word >> i) & 1) for i in range(self.values)]

    def step(self, state: ReferenceState, action: Action) -> ReferenceTransition:
        if not isinstance(action, Read | Write | Idle | Halt):
            raise TypeError(f"not a memory/v1 action: {action!r}")
        reason = self._invalid(state, action)
        if reason is not None:
            return ReferenceTransition(
                state, Event(EventKind.FAULT, reason=reason), None, WORK_PER_ACTION
            )
        match action:
            case Write(address, value):
                nxt = self.update(state, self.one_hot(address), self.bits(value))
                return ReferenceTransition(nxt, Event(EventKind.CONTINUE), None, 1)
            case Read(address):
                nxt = self.update(state, None, None)
                out = self.predict(nxt.params, self.one_hot(address))
                event = Event(EventKind.OUTPUT, value=decode(out))
                return ReferenceTransition(nxt, event, out, WORK_PER_ACTION)
            case Idle():
                nxt = self.update(state, None, None)
                return ReferenceTransition(nxt, Event(EventKind.CONTINUE), None, 1)
            case _:
                nxt = replace(self.update(state, None, None), halted=True)
                return ReferenceTransition(nxt, Event(EventKind.HALT), None, 1)

    def _invalid(self, state: ReferenceState, action: Action) -> FaultReason | None:
        if state.halted:
            return FaultReason.HALTED
        if isinstance(action, Read | Write) and not 0 <= action.address < self.keys:
            return FaultReason.ADDRESS_OUT_OF_RANGE
        if isinstance(action, Write) and not 0 <= action.value < 2**self.values:
            return FaultReason.VALUE_OUT_OF_RANGE
        return None

    def restore(self, snapshot: dict[str, Any]) -> ReferenceState:
        """A state from ``snapshot``; also accepts the tensor backend's."""
        if (snapshot["version"], snapshot["keys"], snapshot["values"]) != (
            VERSION,
            self.keys,
            self.values,
        ):
            raise ValueError("snapshot is for a different memory")
        sizes = {
            n: math.prod(s)
            for n, s in shapes(self.keys, self.values, self.optimizer).items()
        }

        def params(flat: Params | None) -> Params | None:
            if flat is None:
                return None
            if {n: len(v) for n, v in flat.items()} != sizes:
                raise ValueError("snapshot parameters do not match the model")
            return {n: [float(x) for x in flat[n]] for n in sizes}

        restored, momentum = params(snapshot["params"]), params(snapshot["momentum"])
        if restored is None or momentum is None:
            raise ValueError("snapshot lacks parameters or momentum")
        return ReferenceState(
            restored,
            momentum,
            params(snapshot["chunk_start"]),
            int(snapshot["steps"]),
            bool(snapshot["halted"]),
        )

    def snapshot(self, state: ReferenceState) -> dict[str, Any]:
        return {
            "version": VERSION,
            "keys": self.keys,
            "values": self.values,
            "params": state.params,
            "momentum": state.momentum,
            "chunk_start": state.chunk_start,
            "steps": state.steps,
            "halted": state.halted,
        }
