"""Trainable heads that read frozen language-model features.

``ActionHead`` chooses a memory action: an operation, a record, and a value.
``AnswerHead`` predicts a question's answer, optionally from a typed
observation embedded alongside the features. These are the only trained
parameters in E02.

E03 adds ``TickHead``, a head with any number of categorical outputs that can
also read an embedded feedback code, and ``FeedbackEmbedding``, which embeds a
code as a sum of per-field vectors: small for the heads, or the language
model's width for latent feedback.
"""

import torch
from torch import nn


def mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(inputs),
        nn.Linear(inputs, hidden),
        nn.GELU(),
        nn.Linear(hidden, outputs),
    )


class ActionHead(nn.Module):
    def __init__(
        self, features: int, hidden: int, operations: int, records: int, values: int
    ):
        super().__init__()
        self.sizes = (operations, records, values)
        self.net = mlp(features, hidden, sum(self.sizes))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.net(features).split(self.sizes, dim=-1))


class AnswerHead(nn.Module):
    def __init__(self, features: int, hidden: int, values: int, observations: int = 0):
        super().__init__()
        self.embed = nn.Embedding(observations, 32) if observations else None
        extra = 32 if observations else 0
        self.net = mlp(features + extra, hidden, values)

    def forward(
        self,
        features: torch.Tensor,
        observation: torch.Tensor | None = None,
        observation_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``observation_probs`` ``(..., observations)`` is a soft or one-hot
        observation; it embeds as ``probs @ embedding`` and equals the integer
        path for one-hot rows, so gradients can reach whatever produced it."""
        if self.embed is not None:
            if observation_probs is not None:
                embedded = observation_probs @ self.embed.weight
            elif observation is not None:
                embedded = self.embed(observation)
            else:
                raise ValueError("this head needs an observation")
            features = torch.cat([features, embedded], dim=-1)
        return self.net(features)


class FeedbackEmbedding(nn.Module):
    """Embeds integer codes ``(..., fields)`` as the sum of per-field vectors."""

    def __init__(self, sizes: tuple[int, ...], dim: int, std: float = 1.0):
        super().__init__()
        self.tables = nn.ModuleList(nn.Embedding(n, dim) for n in sizes)
        for table in self.tables:
            nn.init.normal_(table.weight, std=std)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [table(codes[..., i]) for i, table in enumerate(self.tables)]
        ).sum(0)


class TickHead(nn.Module):
    """Categorical outputs of ``sizes`` from features and an optional code."""

    def __init__(
        self,
        features: int,
        hidden: int,
        sizes: tuple[int, ...],
        codes: tuple[int, ...] | None = None,
        code_dim: int = 32,
    ):
        super().__init__()
        self.sizes = sizes
        self.feedback = FeedbackEmbedding(codes, code_dim) if codes else None
        extra = code_dim if codes else 0
        self.net = mlp(features + extra, hidden, sum(sizes))

    def forward(
        self, features: torch.Tensor, codes: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, ...]:
        if self.feedback is not None:
            if codes is None:
                raise ValueError("this head needs a feedback code")
            features = torch.cat([features, self.feedback(codes)], dim=-1)
        return tuple(self.net(features).split(self.sizes, dim=-1))


def parameter_count(*modules: nn.Module | None) -> int:
    return sum(p.numel() for m in modules if m is not None for p in m.parameters())
