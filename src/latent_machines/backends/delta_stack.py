"""Delta-rule stack backend for ``stack/v1``.

The stack lives in two tensors:

- ``memory``, a delta-rule state matrix of shape ``(word_bits, capacity)``:
  column ``d`` holds stack slot ``d`` as bits, as in the E00 memory backend.
- ``pointer``, a one-hot vector over depths ``0..capacity``, moved by shift
  matrices ``U`` (up) and ``U^T`` (down).

``Push(v)`` writes ``bits(v)`` with key ``pointer[:capacity]`` and ``beta = 1``,
then ``pointer = U pointer``. ``Pop()`` moves ``pointer = U^T pointer``, reads
with query ``pointer[:capacity]``, and clears the slot with a ``beta = 1``
write of zeros, so freed slots hold no stale words.

Host code checks the value argument, as the memory backend does. Overflow
and underflow are decided from the tensor itself: the pointer's last and
first entries. ``decode``, ``encode``, and ``residue`` exist for conformance
checks and snapshots; a controller must never observe them.
"""

from dataclasses import dataclass

import torch

from latent_machines.backends.delta import DeltaInputs, delta_step
from latent_machines.machines.events import Event, EventKind
from latent_machines.machines.stack import (
    Pop,
    Push,
    Stack,
    StackAction,
    StackFault,
    StackState,
)

WORK_PER_ACTION = 1  # one delta write (and, for Pop, one read) per action


@dataclass(frozen=True)
class DeltaStackState:
    memory: torch.Tensor  # (word_bits, capacity)
    pointer: torch.Tensor  # one-hot over depths 0..capacity


@dataclass(frozen=True)
class DeltaStackTransition:
    next_state: DeltaStackState
    event: Event
    work: int


class DeltaStack:
    """``stack/v1`` stored in a delta-rule matrix with a one-hot depth pointer."""

    def __init__(
        self,
        machine: Stack,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.machine, self.dtype, self.device = machine, dtype, device
        ones = torch.ones(machine.capacity, dtype=dtype, device=device)
        self.shift_up = torch.diag(ones, -1)  # shift_up[d + 1, d] = 1

    def initial_state(self) -> DeltaStackState:
        return self.encode(self.machine.initial_state())

    def step(self, state: DeltaStackState, action: StackAction) -> DeltaStackTransition:
        if not isinstance(action, Push | Pop):
            raise TypeError(f"not a stack/v1 action: {action!r}")
        self._check_shapes(state)
        capacity = self.machine.capacity
        match action:
            case Push(value):
                if not 0 <= value <= self.machine.max_word:
                    return self._fault(state, StackFault.VALUE_OUT_OF_RANGE)
                if state.pointer[capacity] > 0.5:
                    return self._fault(state, StackFault.OVERFLOW)
                key = state.pointer[:capacity]
                write = DeltaInputs(key, self._bits(value), 1.0, 1.0, query=None)
                memory = delta_step(state.memory, write)
                pointer = self.shift_up @ state.pointer
                next_state = DeltaStackState(memory, pointer)
                return DeltaStackTransition(
                    next_state, Event(EventKind.CONTINUE), WORK_PER_ACTION
                )
            case Pop():
                if state.pointer[0] > 0.5:
                    return self._fault(state, StackFault.UNDERFLOW)
                pointer = self.shift_up.T @ state.pointer
                slot = pointer[:capacity]
                value = self._words(state.memory @ slot[:, None])[0]
                clear = DeltaInputs(
                    slot, self._zeros(self.machine.word_bits), 1.0, 1.0, None
                )
                memory = delta_step(state.memory, clear)
                event = Event(EventKind.OUTPUT, value=value)
                return DeltaStackTransition(
                    DeltaStackState(memory, pointer), event, WORK_PER_ACTION
                )
        raise TypeError(f"not a stack/v1 action: {action!r}")

    def decode(self, state: DeltaStackState) -> StackState:
        """Canonical items below the pointer, for conformance only."""
        self._check_shapes(state)
        depth = self._index(state.pointer)
        return StackState(tuple(self._words(state.memory[:, :depth])))

    def encode(self, state: StackState) -> DeltaStackState:
        if len(state.items) > self.machine.capacity:
            raise ValueError(f"{len(state.items)} items exceed capacity")
        memory = self._zeros(self.machine.word_bits, self.machine.capacity)
        for slot, word in enumerate(state.items):
            if not 0 <= word <= self.machine.max_word:
                raise ValueError(f"word {word!r} out of range")
            memory[:, slot] = self._bits(word)
        pointer = self._zeros(self.machine.capacity + 1)
        pointer[len(state.items)] = 1
        return DeltaStackState(memory, pointer)

    def residue(self, state: DeltaStackState) -> float:
        """Largest distance of any memory or pointer entry from 0 or 1."""
        entries = torch.cat([state.memory.flatten(), state.pointer]).to(torch.float64)
        return torch.minimum(entries.abs(), (entries - 1).abs()).max().item()

    def _words(self, columns: torch.Tensor) -> list[int]:
        if not torch.isfinite(columns).all():
            raise ValueError("non-finite entry in stack memory")
        return [
            sum(1 << i for i, bit in enumerate(bits) if bit)
            for bits in (columns > 0.5).T.tolist()
        ]

    def _index(self, vector: torch.Tensor) -> int:
        if not torch.isfinite(vector).all():
            raise ValueError("non-finite entry in stack pointer")
        active = (vector > 0.5).nonzero().flatten().tolist()
        if len(active) != 1:
            raise ValueError(f"expected one active pointer entry, found {len(active)}")
        return active[0]

    def _check_shapes(self, state: DeltaStackState) -> None:
        memory = (self.machine.word_bits, self.machine.capacity)
        pointer = (self.machine.capacity + 1,)
        if tuple(state.memory.shape) != memory or tuple(state.pointer.shape) != pointer:
            raise ValueError("state shapes do not match the machine")

    def _zeros(self, *shape: int) -> torch.Tensor:
        return torch.zeros(shape, dtype=self.dtype, device=self.device)

    def _bits(self, word: int) -> torch.Tensor:
        bits = [(word >> i) & 1 for i in range(self.machine.word_bits)]
        return torch.tensor(bits, dtype=self.dtype, device=self.device)

    def _fault(
        self, state: DeltaStackState, reason: StackFault
    ) -> DeltaStackTransition:
        event = Event(EventKind.FAULT, reason=reason)
        return DeltaStackTransition(state, event, WORK_PER_ACTION)
