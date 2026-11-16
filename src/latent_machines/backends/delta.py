"""Eager delta-rule tensor backend for ``memory/v1``.

Runs the Gated DeltaNet recurrence (https://arxiv.org/abs/2412.06464) in its
explicit recurrent form, one step per action:

    S_t = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    o_t = S_t q_t

computed as ``alpha * (S - beta * (S k) k^T) + beta * v k^T`` without forming
the ``d_k x d_k`` transition matrix.

Encoding:

- ``S`` has shape ``(word_bits, capacity)``. Column ``a`` holds slot ``a``;
  row ``i`` holds bit ``i`` of each word as 0.0 or 1.0.
- Addresses are one-hot keys and queries, L2-normalized as ``x / max(|x|,
  eps)``, which leaves one-hot vectors unchanged. Kernels that add ``eps``
  inside the square root do not, and are a separate variant to measure.
- Gates are set directly to their endpoints: ``alpha = 1`` always, ``beta =
  1`` for a write and ``beta = 0`` for every other valid action, which still
  runs the update. Whether a trained gate parameterization reaches these
  values exactly is a separate check.
- A read decodes ``o`` by thresholding each bit at 0.5. There is no output
  normalization or output gate; those belong to a full layer path.

Host code validates actions, tracks the halted flag, and skips the update on
a fault, so a faulted action leaves ``S`` untouched. It is written
independently of the oracle; only the machine's parameters are shared.

``decode``, ``encode``, and ``residue`` exist for conformance checks and
snapshots. A controller must never observe them.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from latent_machines.machines.memory import (
    Action,
    Event,
    EventKind,
    FaultReason,
    Halt,
    Idle,
    Memory,
    MemoryState,
    Read,
    Write,
)

WORK_PER_ACTION = 1  # one recurrence step
_NORM_EPS = 1e-6


@dataclass(frozen=True)
class DeltaState:
    matrix: torch.Tensor  # (word_bits, capacity), never modified in place
    halted: bool = False


@dataclass(frozen=True)
class DeltaInputs:
    """Layer inputs for one step; ``key`` and ``query`` are not yet normalized."""

    key: torch.Tensor
    value: torch.Tensor
    alpha: float
    beta: float
    query: torch.Tensor | None


@dataclass(frozen=True)
class DeltaTransition:
    next_state: DeltaState
    event: Event
    work: int


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    return F.normalize(vector, dim=0, eps=_NORM_EPS)


def delta_step(matrix: torch.Tensor, inputs: DeltaInputs) -> torch.Tensor:
    """One gated delta-rule update of ``matrix``."""
    key = _normalize(inputs.key)
    retained = matrix - inputs.beta * torch.outer(matrix @ key, key)
    return inputs.alpha * retained + inputs.beta * torch.outer(inputs.value, key)


@dataclass(frozen=True)
class DeltaMemory:
    """``memory/v1`` stored in a delta-rule state matrix."""

    machine: Memory
    dtype: torch.dtype = torch.float32
    device: torch.device | str = "cpu"

    def initial_state(self) -> DeltaState:
        return DeltaState(self._zeros(self.machine.word_bits, self.machine.capacity))

    def step(self, state: DeltaState, action: Action) -> DeltaTransition:
        if not isinstance(action, Read | Write | Idle | Halt):
            raise TypeError(f"not a memory/v1 action: {action!r}")
        self._check_shape(state.matrix)
        if state.halted:
            return self._fault(state, FaultReason.HALTED)
        if isinstance(action, Read | Write):
            if not 0 <= action.address < self.machine.capacity:
                return self._fault(state, FaultReason.ADDRESS_OUT_OF_RANGE)
        if isinstance(action, Write):
            if not 0 <= action.value <= self.machine.max_word:
                return self._fault(state, FaultReason.VALUE_OUT_OF_RANGE)

        inputs = self.lower(action)
        matrix = delta_step(state.matrix, inputs)
        if inputs.query is not None:
            output = matrix @ _normalize(inputs.query)
            event = Event(EventKind.OUTPUT, value=self._decode(output[:, None])[0])
        elif isinstance(action, Halt):
            event = Event(EventKind.HALT)
        else:
            event = Event(EventKind.CONTINUE)
        next_state = DeltaState(matrix, halted=isinstance(action, Halt))
        return DeltaTransition(next_state, event, WORK_PER_ACTION)

    def lower(self, action: Action) -> DeltaInputs:
        """Map a valid action to layer inputs with gates at their endpoints."""
        no_key = self._zeros(self.machine.capacity)
        no_value = self._zeros(self.machine.word_bits)
        match action:
            case Write(address, value):
                key = self._one_hot(address)
                return DeltaInputs(key, self._bits(value), 1.0, 1.0, query=None)
            case Read(address):
                query = self._one_hot(address)
                return DeltaInputs(no_key, no_value, 1.0, 0.0, query=query)
            case Idle() | Halt():
                return DeltaInputs(no_key, no_value, 1.0, 0.0, query=None)
        raise TypeError(f"not a memory/v1 action: {action!r}")

    def decode(self, state: DeltaState) -> MemoryState:
        """Canonical state by thresholding every bit, for conformance only."""
        self._check_shape(state.matrix)
        return MemoryState(tuple(self._decode(state.matrix)), state.halted)

    def encode(self, state: MemoryState) -> DeltaState:
        """Tensor state holding the words of a canonical state exactly."""
        if len(state.words) != self.machine.capacity:
            raise ValueError(f"expected {self.machine.capacity} words")
        for word in state.words:
            if not 0 <= word <= self.machine.max_word:
                raise ValueError(f"word {word!r} out of range")
        columns = [self._bits(word) for word in state.words]
        return DeltaState(torch.stack(columns, dim=1), state.halted)

    def residue(self, state: DeltaState) -> float:
        """Largest distance of any state entry from the nearest of 0 and 1.

        Zero means thresholding changed nothing; a positive value is margin
        lost before canonicalization, even when decoding still succeeds.
        """
        matrix = state.matrix.to(torch.float64)
        return torch.minimum(matrix.abs(), (matrix - 1).abs()).max().item()

    def _decode(self, columns: torch.Tensor) -> list[int]:
        if not torch.isfinite(columns).all():
            raise ValueError("non-finite entry in delta-rule state or output")
        words = []
        for bits in (columns > 0.5).T.tolist():
            words.append(sum(1 << i for i, bit in enumerate(bits) if bit))
        return words

    def _check_shape(self, matrix: torch.Tensor) -> None:
        expected = (self.machine.word_bits, self.machine.capacity)
        if tuple(matrix.shape) != expected:
            raise ValueError(f"state shape {tuple(matrix.shape)} != {expected}")

    def _zeros(self, *shape: int) -> torch.Tensor:
        return torch.zeros(shape, dtype=self.dtype, device=self.device)

    def _one_hot(self, address: int) -> torch.Tensor:
        vector = self._zeros(self.machine.capacity)
        vector[address] = 1
        return vector

    def _bits(self, word: int) -> torch.Tensor:
        bits = [(word >> i) & 1 for i in range(self.machine.word_bits)]
        return torch.tensor(bits, dtype=self.dtype, device=self.device)

    def _fault(self, state: DeltaState, reason: FaultReason) -> DeltaTransition:
        event = Event(EventKind.FAULT, reason=reason)
        return DeltaTransition(state, event, WORK_PER_ACTION)
