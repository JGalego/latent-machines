"""Record-update programs: E02's state-tracking task.

A program is a preamble followed by statements over records ``a`` to ``h``,
each holding a value from 0 to 15 and starting at 0:

    Records a to h start at 0. Set c to 7. What is c? Set a to 3. What is b?

``set`` statements map to ``Write(record, value)`` and questions to
``Read(record)`` on a ``memory/v1`` machine with one slot per record. The
answer to a question is the record's current value. Each statement is a
separate text chunk, so every interface tokenizes the same boundaries.
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass

from latent_machines.machines.memory import Action, Memory, Read, Write

NAMES = "abcdefgh"
VALUE_BITS = 4
MAX_VALUE = 2**VALUE_BITS - 1
PREAMBLE = f"Records {NAMES[0]} to {NAMES[-1]} start at 0."


def machine() -> Memory:
    """The memory the task assumes: one slot per record, 4-bit values."""
    return Memory(capacity=len(NAMES), word_bits=VALUE_BITS)


@dataclass(frozen=True)
class Statement:
    kind: str  # "set" or "get"
    record: int  # index into NAMES
    value: int | None = None  # for "set" only

    @property
    def text(self) -> str:
        name = NAMES[self.record]
        return (
            f" Set {name} to {self.value}."
            if self.kind == "set"
            else f" What is {name}?"
        )

    @property
    def action(self) -> Action:
        if self.kind == "set":
            assert self.value is not None
            return Write(self.record, self.value)
        return Read(self.record)


@dataclass(frozen=True)
class Program:
    statements: tuple[Statement, ...]

    def answers(self) -> list[int | None]:
        """Each statement's answer: the value for a question, else None."""
        values = [0] * len(NAMES)
        answers: list[int | None] = []
        for statement in self.statements:
            if statement.kind == "set":
                assert statement.value is not None
                values[statement.record] = statement.value
                answers.append(None)
            else:
                answers.append(values[statement.record])
        return answers


@dataclass(frozen=True)
class Split:
    """How a set of programs is drawn.

    ``held_out`` lists (record, value) pairs that ``set`` statements avoid
    (when ``exclude_held_out``) or must include at least once (when not).
    """

    name: str
    count: int
    min_statements: int
    max_statements: int
    held_out: frozenset[tuple[int, int]] = frozenset()
    exclude_held_out: bool = True


# Records g and h never take values 12-15 in training; one test split requires them.
HELD_OUT = frozenset((record, value) for record in (6, 7) for value in range(12, 16))


def random_program(rng: random.Random, split: Split) -> Program:
    length = rng.randint(split.min_statements, split.max_statements)
    while True:
        statements, assigned = [], []
        for _ in range(length):
            if rng.random() < 0.5:
                record = rng.randrange(len(NAMES))
                value = rng.randrange(MAX_VALUE + 1)
                while split.exclude_held_out and (record, value) in split.held_out:
                    value = rng.randrange(MAX_VALUE + 1)
                statements.append(Statement("set", record, value))
                assigned.append(record)
            else:
                # Mostly ask about assigned records, so answers are rarely 0.
                if assigned and rng.random() < 0.8:
                    record = rng.choice(assigned)
                else:
                    record = rng.randrange(len(NAMES))
                statements.append(Statement("get", record))
        program = Program(tuple(statements))
        if split.exclude_held_out or _uses_held_out(program, split.held_out):
            return program


def _uses_held_out(program: Program, held_out: frozenset[tuple[int, int]]) -> bool:
    return any(
        s.kind == "set" and (s.record, s.value) in held_out for s in program.statements
    )


def programs(split: Split, seed: int) -> Iterator[Program]:
    """``split.count`` programs, reproducible from ``(split.name, seed)``."""
    rng = random.Random(f"{seed}:{split.name}")
    for _ in range(split.count):
        yield random_program(rng, split)
