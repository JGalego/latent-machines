"""Chunked delta-rule tensor backend for ``memory/v1``.

The recurrent backend in ``delta.py`` applies one update per action. Fused
Gated DeltaNet kernels instead process a chunk of ``C`` actions at once with
the WY representation (https://arxiv.org/abs/2412.06464, section 3.3). This
module implements that form in plain PyTorch so its arithmetic can be checked
against the integer oracle, with the same encoding as ``DeltaMemory``:

- ``S`` has shape ``(word_bits, capacity)``; column ``a`` holds slot ``a``.
- Keys and queries are one-hot addresses, normalized as ``x / max(|x|, eps)``
  outside the chunk kernel. Gates are at their endpoints: ``alpha = 1``,
  ``beta = 1`` for a write, ``beta = 0`` for every other action.
- A read decodes its output by thresholding each bit at 0.5.

For a chunk with start state ``S0``, keys ``K``, values ``V``, gates ``beta``,
and queries ``Q`` (one row per action), with ``u_i = beta_i (v_i - S_{i-1}
k_i)``:

    W_i = beta_i k_i - beta_i sum_{j<i} (k_j . k_i) W_j
    U_i = beta_i v_i - beta_i sum_{j<i} (k_j . k_i) U_j
    u   = U - W S0^T
    O   = Q S0^T + tril(Q K^T) u          (outputs o_i = S_i q_i)
    S   = S0 + u^T K                      (state after the chunk)

``W`` and ``U`` are built by forward substitution in the working dtype, so
every intermediate is rounded as a kernel would round it. The state exists
only at chunk boundaries; ``Chunk`` carries it there, together with every
action's output.

Host code validates actions and tracks the halted flag before lowering, as in
``DeltaMemory``: a faulted action is lowered to a no-op (``beta = 0``, no
query) and its event comes from the host.

``lower_sequence`` and ``dense_inputs`` are shared with the fused-kernel
check in ``fla_kernels.py``. Like ``DeltaMemory.decode``, ``decode`` and
``residue`` are for conformance only; a controller must never observe them.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch

from latent_machines.machines.memory import (
    Action,
    Event,
    EventKind,
    FaultReason,
    Halt,
    Idle,
    Memory,
    Read,
    Write,
)

WORK_PER_ACTION = 1


@dataclass(frozen=True)
class Lowered:
    """A validated action sequence in the form every kernel consumes.

    Entry ``i`` describes action ``i``: the key address of a valid write
    (``-1`` otherwise), its value, ``beta``, the query address of a valid read
    (``-1`` otherwise), and the host's event for actions the tensor does not
    answer (``None`` for a valid read).
    """

    key: list[int]
    value: list[int]
    beta: list[float]
    query: list[int]
    host_events: list[Event | None]

    def __len__(self) -> int:
        return len(self.key)


@dataclass(frozen=True)
class Chunk:
    start: int
    stop: int
    outputs: torch.Tensor  # (stop - start, word_bits); meaningful at reads
    state: torch.Tensor  # (word_bits, capacity) after action stop - 1


def lower_sequence(machine: Memory, actions: Sequence[Action]) -> Lowered:
    """Validate and lower ``actions``; faults and halts follow ``memory/v1``."""
    key, value, beta, query, events = [], [], [], [], []
    halted = False
    for action in actions:
        if not isinstance(action, Read | Write | Idle | Halt):
            raise TypeError(f"not a memory/v1 action: {action!r}")
        reason = _fault(machine, action, halted)
        k, v, b, q, event = -1, 0, 0.0, -1, Event(EventKind.CONTINUE)
        if reason is not None:
            event = Event(EventKind.FAULT, reason=reason)
        elif isinstance(action, Write):
            k, v, b = action.address, action.value, 1.0
        elif isinstance(action, Read):
            q, event = action.address, None
        elif isinstance(action, Halt):
            halted, event = True, Event(EventKind.HALT)
        key.append(k)
        value.append(v)
        beta.append(b)
        query.append(q)
        events.append(event)
    return Lowered(key, value, beta, query, events)


def _fault(machine: Memory, action: Action, halted: bool) -> FaultReason | None:
    if halted:
        return FaultReason.HALTED
    if isinstance(action, Read | Write):
        if not 0 <= action.address < machine.capacity:
            return FaultReason.ADDRESS_OUT_OF_RANGE
    if isinstance(action, Write):
        if not 0 <= action.value <= machine.max_word:
            return FaultReason.VALUE_OUT_OF_RANGE
    return None


def dense_inputs(
    machine: Memory,
    lowered: Lowered,
    start: int,
    stop: int,
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, ...]:
    """Dense ``K, V, beta, Q`` for actions ``start`` to ``stop``.

    One-hot keys and queries are already unit length, so normalization
    outside the kernel leaves them unchanged.
    """
    count, bits = stop - start, machine.word_bits
    key = torch.tensor(lowered.key[start:stop], device=device)
    query = torch.tensor(lowered.query[start:stop], device=device)
    word = torch.tensor(lowered.value[start:stop], device=device)
    rows = torch.arange(count, device=device)
    keys = torch.zeros(count, machine.capacity, dtype=dtype, device=device)
    queries = torch.zeros_like(keys)
    has_key, has_query = key >= 0, query >= 0
    keys[rows[has_key], key[has_key]] = 1
    queries[rows[has_query], query[has_query]] = 1
    shifts = torch.arange(bits, device=device)
    values = ((word[:, None] >> shifts) & 1).to(dtype) * has_key[:, None].to(dtype)
    beta = torch.tensor(lowered.beta[start:stop], dtype=dtype, device=device)
    return keys, values, beta, queries


def chunk_update(
    state: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    beta: torch.Tensor,
    queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One chunk of the delta rule: outputs ``(C, d_v)`` and the next state."""
    count = keys.shape[0]
    gram = keys @ keys.T
    w, u = torch.empty_like(keys), torch.empty_like(values)
    for i in range(count):
        w_i, u_i = beta[i] * keys[i], beta[i] * values[i]
        if i:
            coefficient = beta[i] * gram[i, :i]
            w_i = w_i - coefficient @ w[:i]
            u_i = u_i - coefficient @ u[:i]
        w[i], u[i] = w_i, u_i
    pseudo = u - w @ state.T
    outputs = queries @ state.T + torch.tril(queries @ keys.T) @ pseudo
    return outputs, state + pseudo.T @ keys


@dataclass(frozen=True)
class ChunkedDeltaMemory:
    """``memory/v1`` stored in a delta-rule state matrix, updated in chunks."""

    machine: Memory
    chunk: int = 16
    dtype: torch.dtype = torch.float32
    device: torch.device | str = "cpu"

    def initial_state(self) -> torch.Tensor:
        shape = (self.machine.word_bits, self.machine.capacity)
        return torch.zeros(shape, dtype=self.dtype, device=self.device)

    def run(
        self, lowered: Lowered, state: torch.Tensor | None = None
    ) -> Iterator[Chunk]:
        """Process ``lowered`` chunk by chunk, yielding each chunk's results."""
        state = self.initial_state() if state is None else state
        for start in range(0, len(lowered), self.chunk):
            stop = min(start + self.chunk, len(lowered))
            keys, values, beta, queries = dense_inputs(
                self.machine, lowered, start, stop, self.dtype, self.device
            )
            outputs, state = chunk_update(state, keys, values, beta, queries)
            yield Chunk(start, stop, outputs, state)

    def decode(self, matrix: torch.Tensor) -> tuple[int, ...]:
        """Canonical words by thresholding every bit, for conformance only."""
        return decode_words(matrix.T)

    def events(self, lowered: Lowered, chunk: Chunk) -> list[Event]:
        """The events of a chunk's actions, reads decoded from its outputs."""
        return chunk_events(lowered, chunk)


def decode_words(columns: torch.Tensor) -> tuple[int, ...]:
    """Words from rows of bits (``(n, word_bits)``), thresholded at 0.5."""
    return tuple(word_tensor(columns).tolist())


def word_tensor(columns: torch.Tensor) -> torch.Tensor:
    """Integer words from rows of bits, as a tensor, thresholded at 0.5."""
    if not torch.isfinite(columns).all():
        raise ValueError("non-finite entry in delta-rule state or output")
    shifts = torch.arange(columns.shape[-1], device=columns.device)
    return ((columns > 0.5).to(torch.int64) << shifts).sum(-1)


def chunk_events(lowered: Lowered, chunk: Chunk) -> list[Event]:
    events = []
    for row, i in enumerate(range(chunk.start, chunk.stop)):
        event = lowered.host_events[i]
        if event is None:
            (value,) = decode_words(chunk.outputs[row : row + 1])
            event = Event(EventKind.OUTPUT, value=value)
        events.append(event)
    return events


def residue(*tensors: torch.Tensor) -> float:
    """Largest distance of any entry from the nearest of 0 and 1."""
    worst = 0.0
    for tensor in tensors:
        t = tensor.to(torch.float64)
        worst = max(worst, torch.minimum(t.abs(), (t - 1).abs()).max().item())
    return worst
