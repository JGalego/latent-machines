"""Record programs with copies: E03's task."""

import random

from latent_machines.machines.memory import Read, Write
from latent_machines.tasks import copies


def run_gold(program: copies.Program) -> list[int | None]:
    """Answers from executing the gold actions on the oracle."""
    machine = copies.machine()
    state, answers = machine.initial_state(), []
    for statement, actions in zip(
        program.statements, program.gold_actions(), strict=True
    ):
        events = []
        for action in actions:
            transition = machine.step(state, action)
            state = transition.next_state
            events.append(transition.event)
        answers.append(events[-1].value if statement.kind == "get" else None)
    return answers


def test_copy_reads_the_source_then_writes_its_value() -> None:
    program = copies.Program(
        (
            copies.Statement("set", 2, value=7),
            copies.Statement("copy", 4, source=2),
            copies.Statement("get", 4),
        )
    )
    assert program.statements[1].text == " Copy c to e."
    assert program.gold_actions() == [[Write(2, 7)], [Read(2), Write(4, 7)], [Read(4)]]
    assert program.answers() == [None, None, 7]


def test_gold_actions_reproduce_the_answers() -> None:
    split = copies.Split("check", 200, 2, 32)
    for program in copies.programs(split, seed=0):
        assert run_gold(program) == program.answers()


def test_programs_are_reproducible_and_mix_kinds() -> None:
    split = copies.Split("train", 50, 2, 8)
    first = list(copies.programs(split, seed=0))
    assert first == list(copies.programs(split, seed=0))
    assert first != list(copies.programs(split, seed=1))
    kinds = {s.kind for p in first for s in p.statements}
    assert kinds == {"set", "copy", "get"}
    for p in first:
        assert 2 <= len(p.statements) <= 8
        for s in p.statements:
            if s.kind == "copy":
                assert s.source != s.record


def test_fixed_length_split() -> None:
    program = copies.random_program(random.Random(0), copies.Split("x", 1, 32, 32))
    assert len(program.statements) == 32
