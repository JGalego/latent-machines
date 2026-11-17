"""Record-update programs: answers, actions, and held-out pairs."""

import pytest

from latent_machines.machines.memory import Read, Write
from latent_machines.tasks import records as task


def test_answers_track_the_latest_value_and_default_to_zero() -> None:
    program = task.Program(
        (
            task.Statement("get", 2),
            task.Statement("set", 2, 7),
            task.Statement("get", 2),
            task.Statement("set", 2, 9),
            task.Statement("get", 2),
        )
    )
    assert program.answers() == [0, None, 7, None, 9]
    assert program.statements[1].action == Write(2, 7)
    assert program.statements[0].action == Read(2)


def test_statement_text() -> None:
    assert task.Statement("set", 0, 15).text == " Set a to 15."
    assert task.Statement("get", 7).text == " What is h?"


def test_programs_are_reproducible_and_sized() -> None:
    split = task.Split("train", 50, 2, 8, task.HELD_OUT)
    first = list(task.programs(split, seed=0))
    assert first == list(task.programs(split, seed=0))
    assert all(2 <= len(p.statements) <= 8 for p in first)


def pairs(programs: list[task.Program]) -> set[tuple[int, int]]:
    return {
        (s.record, s.value) for p in programs for s in p.statements if s.kind == "set"
    }


def test_held_out_pairs_are_excluded_or_required() -> None:
    train = list(task.programs(task.Split("train", 500, 2, 8, task.HELD_OUT), 0))
    assert not pairs(train) & task.HELD_OUT
    split = task.Split("held", 100, 2, 8, task.HELD_OUT, exclude_held_out=False)
    for program in task.programs(split, 0):
        assert pairs([program]) & task.HELD_OUT


@pytest.mark.parametrize("length", [32, 64])
def test_long_programs_have_exact_length(length: int) -> None:
    split = task.Split("long", 5, length, length, task.HELD_OUT)
    assert all(len(p.statements) == length for p in task.programs(split, 0))
