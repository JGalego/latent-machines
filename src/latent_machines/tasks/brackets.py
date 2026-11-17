"""Bracket matching with a bounded stack: E01's stack-parser task.

A text over ``()[]{}`` is checked by pushing a code for each opening bracket
and popping for each closing one. ``parse`` is the ground truth under a
stack capacity; ``actions_for`` is the ``stack/v1`` action stream a parser
issues; ``verdict_from_events`` reads a verdict back from the stack's events.
"""

import random
from collections.abc import Iterator, Sequence

from latent_machines.machines.events import Event, EventKind
from latent_machines.machines.stack import Pop, Push, Stack, StackAction

OPEN, CLOSE = "([{", ")]}"


def random_text(rng: random.Random, length: int) -> str:
    """Mostly well-nested text: closes favor the most recent open bracket."""
    text, open_stack = [], []
    for _ in range(length):
        if open_stack and rng.random() < 0.45:
            kind = open_stack.pop() if rng.random() < 0.9 else rng.randrange(3)
            text.append(CLOSE[kind])
        elif rng.random() < 0.05:
            text.append(CLOSE[rng.randrange(3)])  # likely unmatched
        else:
            kind = rng.randrange(3)
            open_stack.append(kind)
            text.append(OPEN[kind])
    return "".join(text)


def parse(text: str, capacity: int) -> tuple[str, int]:
    """Verdict and deciding position under a stack capacity.

    Verdicts: balanced, unclosed, mismatch, underflow, or overflow.
    """
    depth: list[int] = []
    for position, char in enumerate(text):
        if char in OPEN:
            if len(depth) == capacity:
                return "overflow", position
            depth.append(OPEN.index(char))
        elif not depth:
            return "underflow", position
        elif depth.pop() != CLOSE.index(char):
            return "mismatch", position
    return ("balanced", len(text)) if not depth else ("unclosed", len(text))


def actions_for(text: str, capacity: int) -> list[StackAction]:
    """Stack actions up to the deciding character, then pops that empty the stack."""
    verdict, stop = parse(text, capacity)
    actions: list[StackAction] = []
    depth = 0
    for char in text[: stop + 1] if verdict != "balanced" else text:
        if char in OPEN:
            actions.append(Push(OPEN.index(char)))
            depth = min(depth + 1, capacity)
        else:
            actions.append(Pop())
            depth = max(depth - 1, 0)
    return actions + [Pop()] * depth


def verdict_from_events(text: str, events: Sequence[Event]) -> str:
    """The verdict a parser reaches from the stack's events for ``text``."""
    depth = 0
    for char, event in zip(text, events, strict=False):
        if event.kind is EventKind.FAULT:
            return event.reason.value if event.reason else "fault"
        if char in OPEN:
            depth += 1
        else:
            depth -= 1
            if event.value != CLOSE.index(char):
                return "mismatch"
    return "balanced" if depth == 0 else "unclosed"


def stream(machine: Stack, rng: random.Random) -> Iterator[StackAction]:
    """Endless actions for random texts of up to twice the stack capacity."""
    while True:
        text = random_text(rng, rng.randrange(1, 2 * machine.capacity + 1))
        yield from actions_for(text, machine.capacity)
