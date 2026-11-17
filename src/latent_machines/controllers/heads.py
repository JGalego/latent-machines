"""Trainable heads that read frozen language-model features.

``ActionHead`` chooses a memory action: an operation, a record, and a value.
``AnswerHead`` predicts a question's answer, optionally from a typed
observation embedded alongside the features. These are the only trained
parameters in E02.
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
        self, features: torch.Tensor, observation: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.embed is not None:
            if observation is None:
                raise ValueError("this head needs an observation")
            features = torch.cat([features, self.embed(observation)], dim=-1)
        return self.net(features)


def parameter_count(*modules: nn.Module | None) -> int:
    return sum(p.numel() for m in modules if m is not None for p in m.parameters())
