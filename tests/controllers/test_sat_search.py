"""Search procedures: DPLL, WalkSAT, and verified proposals."""

import random

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends.cnf_tensor import TensorVerifier  # noqa: E402
from latent_machines.controllers import sat_search  # noqa: E402
from latent_machines.machines import cnf  # noqa: E402
from latent_machines.machines.cnf import Formula  # noqa: E402
from latent_machines.tasks import sat_instances  # noqa: E402


def test_dpll_agrees_with_enumeration() -> None:
    rng = random.Random(0)
    for _ in range(60):
        n = rng.randint(3, 10)
        formula = sat_instances.uniform(rng, n, rng.choice((3.0, 6.0)))
        outcome = sat_search.dpll(formula, 10**6)
        satisfiable = bool(sat_instances.satisfying(formula))
        assert outcome.status == ("sat" if satisfiable else "unsat")
        if satisfiable:
            assert cnf.check(formula, outcome.assignment).accepted


def test_budget_exhaustion_is_unknown_not_unsat() -> None:
    formula = sat_instances.uniform(random.Random(2), 20, 6.0)
    assert sat_search.dpll(formula, 3).status == "unknown"
    unsat = Formula(1, ((1,), (-1,)))
    assert sat_search.walksat(unsat, 100, random.Random(0)).status == "unknown"
    assert sat_search.dpll(unsat, 100).status == "unsat"
    propose = sat_search.random_proposals(unsat, 4, seed=0)
    assert sat_search.run_proposals(unsat, propose, None, 50).status == "unknown"


def test_walksat_and_energy_find_planted_solutions() -> None:
    formula, _ = sat_instances.planted(random.Random(3), 20, 3.5)
    walk = sat_search.walksat(formula, 5_000, random.Random(0))
    assert walk.status == "sat" and cnf.check(formula, walk.assignment).accepted
    propose = sat_search.energy_proposals(formula, 16, seed=0)
    found = sat_search.run_proposals(formula, propose, None, 5_000)
    assert found.status == "sat" and cnf.check(formula, found.assignment).accepted


def test_verifiers_make_identical_decisions_on_the_same_proposals() -> None:
    formula = sat_instances.uniform(random.Random(4), 20, 6.0)
    verifiers = (None, TensorVerifier(formula), TensorVerifier(formula, "folded"))
    outcomes = [
        sat_search.run_proposals(
            formula, sat_search.energy_proposals(formula, 8, seed=1), verifier, 400
        )
        for verifier in verifiers
    ]
    assert len({o.decisions for o in outcomes}) == 1
    assert {o.status for o in outcomes} == {"unknown"}
    assert all(o.verifier_calls == o.steps == 400 for o in outcomes)
