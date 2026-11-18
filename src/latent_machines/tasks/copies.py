"""Record programs with copies: E03's multi-tick task.

Extends E02's record-update programs with a statement whose write depends on
a read made in the same statement:

    Records a to h start at 0. Set c to 7. Copy c to e. What is e?

``Copy x to y`` is two machine ticks on ``memory/v1``: ``Read(x)``, then
``Write(y, v)`` with the value ``v`` that the read returned. The value is not
in the text, so a controller must route the read's reply back into its next
decision. ``set`` is one ``Write`` and a question one ``Read``, as in E02.
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass

from latent_machines.machines.memory import Action, Read, Write
from latent_machines.tasks.records import MAX_VALUE, NAMES, PREAMBLE, machine

__all__ = ["MAX_VALUE", "NAMES", "PREAMBLE", "machine"]


@dataclass(frozen=True)
class Statement:
    kind: str  # "set", "copy", or "get"
    record: int  # the record written (set, copy) or asked about (get)
    value: int | None = None  # for "set" only
    source: int | None = None  # for "copy" only

    @property
    def text(self) -> str:
        name = NAMES[self.record]
        if self.kind == "set":
            return f" Set {name} to {self.value}."
        if self.kind == "copy":
            assert self.source is not None
            return f" Copy {NAMES[self.source]} to {name}."
        return f" What is {name}?"


@dataclass(frozen=True)
class Program:
    statements: tuple[Statement, ...]

    def gold_actions(self) -> list[list[Action]]:
        """Each statement's machine actions under correct control."""
        values = [0] * len(NAMES)
        actions: list[list[Action]] = []
        for s in self.statements:
            if s.kind == "set":
                assert s.value is not None
                values[s.record] = s.value
                actions.append([Write(s.record, s.value)])
            elif s.kind == "copy":
                assert s.source is not None
                value = values[s.source]
                values[s.record] = value
                actions.append([Read(s.source), Write(s.record, value)])
            else:
                actions.append([Read(s.record)])
        return actions

    def answers(self) -> list[int | None]:
        """Each statement's answer: the value for a question, else None."""
        values = [0] * len(NAMES)
        answers: list[int | None] = []
        for s in self.statements:
            answers.append(values[s.record] if s.kind == "get" else None)
            if s.kind == "set":
                assert s.value is not None
                values[s.record] = s.value
            elif s.kind == "copy":
                assert s.source is not None
                values[s.record] = values[s.source]
        return answers


@dataclass(frozen=True)
class Split:
    name: str
    count: int
    min_statements: int
    max_statements: int


KINDS = ("set", "copy", "get")
WEIGHTS = (0.4, 0.3, 0.3)


def random_program(rng: random.Random, split: Split) -> Program:
    length = rng.randint(split.min_statements, split.max_statements)
    statements, assigned = [], []
    for _ in range(length):
        kind = rng.choices(KINDS, WEIGHTS)[0]
        if kind == "set":
            record = rng.randrange(len(NAMES))
            statements.append(
                Statement("set", record, value=rng.randrange(MAX_VALUE + 1))
            )
            assigned.append(record)
            continue
        # Mostly read assigned records, so values are rarely 0.
        if assigned and rng.random() < 0.8:
            source = rng.choice(assigned)
        else:
            source = rng.randrange(len(NAMES))
        if kind == "copy":
            target = rng.choice([r for r in range(len(NAMES)) if r != source])
            statements.append(Statement("copy", target, source=source))
            assigned.append(target)
        else:
            statements.append(Statement("get", source))
    return Program(tuple(statements))


def programs(split: Split, seed: int) -> Iterator[Program]:
    """``split.count`` programs, reproducible from ``(split.name, seed)``."""
    rng = random.Random(f"{seed}:copies:{split.name}")
    for _ in range(split.count):
        yield random_program(rng, split)
