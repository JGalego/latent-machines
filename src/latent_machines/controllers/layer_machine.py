"""A delta-rule memory inside one layer of a frozen decoder model (E10).

The model stays frozen. A forward hook on the output of decoder layer ``k``
(1-indexed) gives trained projections the residual stream at a window's last
token. The projections produce a macro (``set``, ``get``, ``copy``, and
optionally ``reject``), a record, a source record, and a value. Addressing
is a straight-through argmax: the forward pass is one-hot and the backward
pass uses the softmax, so the memory stays exact while gradients reach the
projections.

The memory is a ``(16, 8)`` matrix per sequence: column ``r`` is the one-hot
code of record ``r``'s value, and all columns start as the code of 0. A
write is the delta-rule update ``S - beta (S k) k^T + beta v k^T`` with a
one-hot key ``k`` and ``beta = 1``, which replaces one column and leaves the
others untouched. A read is ``S q``. A ``copy`` reads the source and writes
the value read, both inside the same forward pass.

The read value can re-enter the residual stream at layer ``k`` through a
trained projection (arm M), so the layers above ``k`` compute with it.
``LayerHook`` can also run the machine live inside a single model pass, which
the tests use to check that the cached evaluation path is the same function.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from latent_machines.controllers.frozen_lm import FrozenLM
from latent_machines.controllers.heads import mlp
from latent_machines.tasks import records as record_task

RECORDS = len(record_task.NAMES)
VALUES = record_task.MAX_VALUE + 1
MACROS = ("set", "get", "copy", "reject")
SET, GET, COPY, REJECT = range(4)


# ---------- the memory ----------


def initial_state(batch: int = 1) -> torch.Tensor:
    """Every record holds the code of 0: ``(batch, VALUES, RECORDS)``."""
    state = torch.zeros(batch, VALUES, RECORDS)
    state[:, 0, :] = 1.0
    return state


def delta_update(
    state: torch.Tensor, key: torch.Tensor, value: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """``S - beta (S k) k^T + beta v k^T`` for a batch; ``key`` is one-hot."""
    stored = torch.einsum("bvr,br->bv", state, key)
    b = beta[:, None, None]
    return (
        state
        - b * stored[:, :, None] * key[:, None, :]
        + (b * value[:, :, None] * key[:, None, :])
    )


def apply(
    state: torch.Tensor,
    macro: torch.Tensor,
    record: torch.Tensor,
    source: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute one macro per sequence; return the new state and the value read.

    The value read is nonzero only for ``get``; a copy's read stays inside
    the layer. ``reject`` leaves the state unchanged.
    """
    is_set, is_get, is_copy = macro == SET, macro == GET, macro == COPY
    query = F.one_hot(torch.where(is_copy, source, record), RECORDS).float()
    read = torch.einsum("bvr,br->bv", state, query)
    key = F.one_hot(record, RECORDS).float()
    written = torch.where(is_copy[:, None], read, F.one_hot(value, VALUES).float())
    beta = (is_set | is_copy).float()
    return delta_update(state, key, written, beta), read * is_get[:, None].float()


def decode(state: torch.Tensor) -> list[list[int]]:
    """Each record's stored value, for conformance checks only."""
    return state.argmax(1).tolist()


def st_onehot(logits: torch.Tensor) -> torch.Tensor:
    """One-hot argmax in the forward pass, softmax gradient in the backward."""
    soft = logits.softmax(-1)
    hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(soft.dtype)
    return hard + (soft - soft.detach())


# ---------- trained parts ----------


class LayerMachine(nn.Module):
    """Projections of the residual stream, and optionally the read's way back."""

    def __init__(
        self, features: int, hidden: int, reject: bool, residual: bool
    ) -> None:
        super().__init__()
        self.sizes = (4 if reject else 3, RECORDS, RECORDS, VALUES)
        self.net = mlp(features, hidden, sum(self.sizes))
        self.residual = nn.Linear(VALUES, features, bias=False) if residual else None

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.net(residual).split(self.sizes, dim=-1))

    def choose(self, residual: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Macro, record, source, and value by argmax."""
        return tuple(logits.argmax(-1) for logits in self(residual))


class TopAnswer(nn.Module):
    """Answer head on the last layer's feature (arm M)."""

    def __init__(self, features: int, hidden: int) -> None:
        super().__init__()
        self.net = mlp(features, hidden, VALUES)

    def forward(self, top: torch.Tensor) -> torch.Tensor:
        return self.net(top)


class ReadAnswer(nn.Module):
    """Answer head that also gets the read as an embedded value (arm Mh)."""

    def __init__(self, features: int, hidden: int) -> None:
        super().__init__()
        self.embed = nn.Linear(VALUES, 32, bias=False)
        self.net = mlp(features + 32, hidden, VALUES)

    def forward(self, top: torch.Tensor, read: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([top, self.embed(read)], dim=-1))


# ---------- the layer hook ----------


class LayerHook:
    """Forward hook on the output of decoder layer ``layer`` (1-indexed).

    ``captured`` holds the residual stream at the last token. ``add`` is
    added to it, or ``callback(residual)`` returns what to add; the machine
    runs inside the pass that way. ``calls`` counts the model passes seen.
    The hook is removed on exit, also after an exception.
    """

    def __init__(self, lm: FrozenLM, layer: int) -> None:
        self.module = lm.model.layers[layer - 1]
        self.handle: Any = None
        self.calls = 0
        self.captured: torch.Tensor | None = None
        self.add: torch.Tensor | None = None
        self.callback: Callable[[torch.Tensor], torch.Tensor | None] | None = None

    def __enter__(self) -> "LayerHook":
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: object) -> None:
        self.handle.remove()
        self.handle = None

    def _hook(self, module: nn.Module, args: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        self.calls += 1
        last = hidden[:, -1]
        self.captured = last.detach()
        add = self.callback(last) if self.callback is not None else self.add
        if add is None:
            return None
        patched = hidden.clone()
        patched[:, -1] = patched[:, -1] + add.to(patched.dtype)
        return (patched, *output[1:]) if isinstance(output, tuple) else patched


def window_ids(lm: FrozenLM, preamble: str, text: str) -> list[int]:
    return [token for ids in lm.tokens([preamble, text]) for token in ids]


class WindowCache:
    """Layer-``k`` and last-layer features of statement windows, by text.

    A window is the preamble plus one statement. Layers up to ``k`` do not
    see anything added at ``k``, so the layer-``k`` feature is the same
    with and without an injection and is computed once per text.
    """

    def __init__(self, lm: FrozenLM, layer: int, preamble: str) -> None:
        self.lm, self.layer, self.preamble = lm, layer, preamble
        self.mid: dict[str, torch.Tensor] = {}
        self.top: dict[str, torch.Tensor] = {}
        self.length: dict[str, int] = {}

    @torch.no_grad()
    def add(self, texts: list[str]) -> None:
        for text in texts:
            if text in self.mid:
                continue
            ids = window_ids(self.lm, self.preamble, text)
            with LayerHook(self.lm, self.layer) as hook:
                out = self.lm.model(
                    input_ids=torch.tensor(ids, device=self.lm.device)[None]
                )
            assert hook.captured is not None
            self.mid[text] = hook.captured[0].cpu()
            self.top[text] = out.last_hidden_state[0, -1].cpu()
            self.length[text] = len(ids)

    def mid_features(self, texts: list[str]) -> torch.Tensor:
        self.add(texts)
        return torch.stack([self.mid[t] for t in texts])

    def top_features(self, texts: list[str]) -> torch.Tensor:
        self.add(texts)
        return torch.stack([self.top[t] for t in texts])


def inject_forward(
    lm: FrozenLM, layer: int, ids: torch.Tensor, add: torch.Tensor
) -> torch.Tensor:
    """Last-layer features at the last token with ``add`` added after layer ``k``.

    Differentiable in ``add``; the frozen layers get no gradient.
    """
    with LayerHook(lm, layer) as hook:
        hook.add = add.to(lm.device)
        out = lm.model(input_ids=ids.to(lm.device))
    return out.last_hidden_state[:, -1].cpu()


@dataclass
class LiveStep:
    macro: int
    record: int
    source: int
    value: int
    state: torch.Tensor
    read: torch.Tensor
    top: torch.Tensor
    passes: int


@torch.no_grad()
def live_statement(
    lm: FrozenLM,
    layer: int,
    machine: LayerMachine,
    preamble: str,
    text: str,
    state: torch.Tensor,
) -> LiveStep:
    """Run one statement in a single model pass with the machine inside layer ``k``.

    The hook reads the residual stream at the last token, chooses the macro,
    updates the memory, and adds the projected read to the residual stream
    before the upper layers run.
    """
    ids = torch.tensor(window_ids(lm, preamble, text), device=lm.device)[None]
    step: dict[str, Any] = {}

    def callback(residual: torch.Tensor) -> torch.Tensor | None:
        macro, record, source, value = machine.choose(residual.cpu())
        step["choice"] = (macro, record, source, value)
        step["state"], step["read"] = apply(state, macro, record, source, value)
        if machine.residual is None:
            return None
        return machine.residual(step["read"]).to(residual.device)

    with LayerHook(lm, layer) as hook:
        hook.callback = callback
        out = lm.model(input_ids=ids)
    macro, record, source, value = (int(x) for x in step["choice"])
    return LiveStep(
        macro,
        record,
        source,
        value,
        step["state"],
        step["read"],
        out.last_hidden_state[:, -1].cpu(),
        hook.calls,
    )
