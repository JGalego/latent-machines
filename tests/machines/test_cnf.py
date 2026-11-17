"""The cnf/v1 assignment verifier."""

import pytest

from latent_machines.machines import cnf
from latent_machines.machines.cnf import FaultReason, Formula
from latent_machines.machines.events import EventKind

FORMULA = Formula(3, ((1, -2), (2, 3), (-1, -3)))


def test_accepts_only_assignments_that_satisfy_every_clause() -> None:
    accepted = cnf.check(FORMULA, (1, 1, 0))
    assert accepted.accepted and accepted.unsatisfied == 0
    assert accepted.first_unsatisfied is None and accepted.work == 3
    rejected = cnf.check(FORMULA, (1, 0, 1))
    assert rejected.event.kind is EventKind.OUTPUT and rejected.event.value == 0
    assert (rejected.unsatisfied, rejected.first_unsatisfied, rejected.work) == (
        1,
        2,
        3,
    )


def test_malformed_assignments_fault_before_evaluation() -> None:
    short = cnf.check(FORMULA, (1, 1))
    assert short.event.reason is FaultReason.WRONG_LENGTH and short.work == 0
    for bad in ((1, 2, 0), (1, -1, 0)):
        verdict = cnf.check(FORMULA, bad)
        assert verdict.event.reason is FaultReason.VALUE_OUT_OF_RANGE
        assert not verdict.accepted
    with pytest.raises(TypeError):
        cnf.check(FORMULA, (True, 1, 0))


def test_empty_clauses_and_tautologies() -> None:
    assert not cnf.check(Formula(1, ((),)), (1,)).accepted
    assert cnf.check(Formula(1, ((1, -1),)), (0,)).accepted
    assert cnf.check(Formula(2, ()), (0, 1)).accepted  # no clauses


@pytest.mark.parametrize(
    ("num_vars", "clauses"),
    [(0, ()), (2, ((3,),)), (2, ((0,),)), (2, ([1],)), (2, ((True,),))],
)
def test_malformed_formulas_are_rejected(num_vars: int, clauses: tuple) -> None:
    with pytest.raises(ValueError):
        Formula(num_vars, clauses)
