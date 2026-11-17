"""ReLU lookup-circuit backend for ``transducer/v1``.

Compiles a transducer's tables into a two-layer ReLU network with explicit
feedback. The state is a one-hot vector ``s`` over states; each action feeds
a one-hot symbol ``x``:

    z  = relu(W_in [s; x] - 1)    one hidden unit per (state, symbol) pair
    s' = W_next z                 next one-hot state, fed back
    y  = W_out z                  one-hot output symbol

Unit ``(i, a)`` has weight 1 on ``s_i`` and ``x_a``, so it is 1 exactly when
both are active and 0 otherwise. Every intermediate is a small integer, so
the circuit is exact in every floating-point format with one-hot inputs.

Host code, written independently of the oracle, faults on out-of-range
symbols before touching the circuit. ``decode``, ``encode``, and ``residue``
exist for conformance checks and snapshots; a controller must never observe
them.
"""

from dataclasses import dataclass

import torch

from latent_machines.machines.events import Event, EventKind
from latent_machines.machines.transducer import (
    Feed,
    Transducer,
    TransducerFault,
    TransducerState,
)

WORK_PER_ACTION = 1  # one pass through the circuit


@dataclass(frozen=True)
class LookupState:
    vector: torch.Tensor  # one-hot over states, never modified in place


@dataclass(frozen=True)
class LookupTransition:
    next_state: LookupState
    event: Event
    work: int


class LookupTransducer:
    """``transducer/v1`` compiled into a ReLU lookup circuit."""

    def __init__(
        self,
        machine: Transducer,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.machine, self.dtype, self.device = machine, dtype, device
        states, symbols = machine.num_states, machine.alphabet_size
        hidden = states * symbols
        w_in = torch.zeros(hidden, states + symbols)
        w_next = torch.zeros(states, hidden)
        w_out = torch.zeros(machine.output_size, hidden)
        for state in range(states):
            for symbol in range(symbols):
                unit = state * symbols + symbol
                w_in[unit, state] = w_in[unit, states + symbol] = 1
                w_next[machine.next_state[state][symbol], unit] = 1
                w_out[machine.output[state][symbol], unit] = 1
        self.w_in, self.w_next, self.w_out = (
            weight.to(dtype=dtype, device=device) for weight in (w_in, w_next, w_out)
        )

    def circuit_stats(self) -> dict[str, int]:
        weights = (self.w_in, self.w_next, self.w_out)
        return {
            "hidden_width": self.w_in.shape[0],
            "depth": 2,
            "stored_weights": sum(w.numel() for w in weights),
            "nonzero_weights": sum(int(w.count_nonzero()) for w in weights),
        }

    def initial_state(self) -> LookupState:
        return self.encode(self.machine.initial_state())

    def step(self, state: LookupState, action: Feed) -> LookupTransition:
        if not isinstance(action, Feed):
            raise TypeError(f"not a transducer/v1 action: {action!r}")
        if tuple(state.vector.shape) != (self.machine.num_states,):
            raise ValueError(f"state shape {tuple(state.vector.shape)} is wrong")
        if not 0 <= action.symbol < self.machine.alphabet_size:
            fault = Event(EventKind.FAULT, reason=TransducerFault.SYMBOL_OUT_OF_RANGE)
            return LookupTransition(state, fault, WORK_PER_ACTION)

        symbol = self._one_hot(action.symbol, self.machine.alphabet_size)
        hidden = torch.relu(self.w_in @ torch.cat([state.vector, symbol]) - 1)
        next_vector = self.w_next @ hidden
        output = self._index(self.w_out @ hidden)
        event = Event(EventKind.OUTPUT, value=output)
        return LookupTransition(LookupState(next_vector), event, WORK_PER_ACTION)

    def decode(self, state: LookupState) -> TransducerState:
        """Canonical state, for conformance only."""
        return TransducerState(self._index(state.vector))

    def encode(self, state: TransducerState) -> LookupState:
        if not 0 <= state.state < self.machine.num_states:
            raise ValueError(f"state {state.state} out of range")
        return LookupState(self._one_hot(state.state, self.machine.num_states))

    def residue(self, state: LookupState) -> float:
        """Largest distance of any state entry from the nearest of 0 and 1."""
        vector = state.vector.to(torch.float64)
        return torch.minimum(vector.abs(), (vector - 1).abs()).max().item()

    def _index(self, vector: torch.Tensor) -> int:
        if not torch.isfinite(vector).all():
            raise ValueError("non-finite entry in one-hot vector")
        active = (vector > 0.5).nonzero().flatten().tolist()
        if len(active) != 1:
            raise ValueError(f"expected one active entry, found {len(active)}")
        return active[0]

    def _one_hot(self, index: int, size: int) -> torch.Tensor:
        vector = torch.zeros(size, dtype=self.dtype, device=self.device)
        vector[index] = 1
        return vector
