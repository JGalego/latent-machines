"""3-SAT instances and verifier workloads for E08.

Instance families, each reproducible from a ``random.Random``:

- ``uniform``: ``round(ratio * n)`` clauses of three distinct variables with
  random signs. Near ratio 4.26 about half are satisfiable; at ratio 6 most
  small instances are not. Satisfiability is unknown until a solver decides.
- ``planted``: uniform clauses, keeping only those a hidden assignment
  satisfies, so the instance is satisfiable by construction.
- ``wide``: a few clauses of ``width`` literals, for testing count precision
  beyond the exact-integer range of small floating-point formats.

Verifier workloads produce candidate assignments, including malformed ones,
for checking the tensor verifier against the ``cnf/v1`` oracle.
"""

import itertools
import random
from collections.abc import Iterator
from dataclasses import dataclass

from latent_machines.machines import cnf
from latent_machines.machines.cnf import Formula

WORKLOADS = ("random", "near-miss", "enumeration", "malformed", "wide-clauses")


def uniform(rng: random.Random, n: int, ratio: float) -> Formula:
    return Formula(n, tuple(_clause(rng, n) for _ in range(round(ratio * n))))


def planted(rng: random.Random, n: int, ratio: float) -> tuple[Formula, list[int]]:
    hidden = [rng.randrange(2) for _ in range(n)]
    clauses: list[tuple[int, ...]] = []
    while len(clauses) < round(ratio * n):
        clause = _clause(rng, n)
        if any(hidden[abs(lit) - 1] == (lit > 0) for lit in clause):
            clauses.append(clause)
    return Formula(n, tuple(clauses)), hidden


def wide(rng: random.Random, n: int, clauses: int, width: int) -> Formula:
    return Formula(
        n,
        tuple(
            tuple(
                v if rng.random() < 0.5 else -v
                for v in rng.sample(range(1, n + 1), width)
            )
            for _ in range(clauses)
        ),
    )


def _clause(rng: random.Random, n: int) -> tuple[int, ...]:
    return tuple(
        v if rng.random() < 0.5 else -v for v in rng.sample(range(1, n + 1), 3)
    )


@dataclass(frozen=True)
class Case:
    formula: Formula
    assignment: tuple[int, ...]


def satisfying(formula: Formula) -> list[tuple[int, ...]]:
    """Every satisfying assignment, by enumeration; for small formulas only."""
    if formula.num_vars > 16:
        raise ValueError("enumeration is limited to 16 variables")
    return [
        bits
        for bits in itertools.product((0, 1), repeat=formula.num_vars)
        if cnf.check(formula, bits).accepted
    ]


def near_misses(formula: Formula, solution: list[int]) -> list[tuple[int, ...]]:
    """Single flips of ``solution`` that leave exactly one clause unsatisfied."""
    found = []
    for v in range(formula.num_vars):
        flipped = list(solution)
        flipped[v] ^= 1
        if cnf.check(formula, flipped).unsatisfied == 1:
            found.append(tuple(flipped))
    return found


def cases(workload: str, horizon: int, rng: random.Random) -> Iterator[Case]:
    """``horizon`` candidate assignments (and their formulas) for a workload."""
    produced = 0
    while produced < horizon:
        for case in _batch(workload, rng):
            if produced == horizon:
                return
            yield case
            produced += 1


def _batch(workload: str, rng: random.Random) -> list[Case]:
    if workload == "random":
        formula = uniform(rng, rng.randint(5, 60), rng.choice((3.0, 4.26, 6.0)))
        return [
            Case(formula, tuple(rng.randrange(2) for _ in range(formula.num_vars)))
            for _ in range(20)
        ]
    if workload == "near-miss":
        formula, hidden = planted(rng, rng.randint(10, 60), 4.2)
        misses = near_misses(formula, hidden)
        return [Case(formula, tuple(hidden))] + [Case(formula, m) for m in misses]
    if workload == "enumeration":
        formula = uniform(rng, rng.randint(3, 10), rng.choice((2.0, 4.26, 6.0)))
        return [
            Case(formula, bits)
            for bits in itertools.product((0, 1), repeat=formula.num_vars)
        ]
    if workload == "malformed":
        formula = uniform(rng, rng.randint(3, 30), 4.26)
        n = formula.num_vars
        good = [rng.randrange(2) for _ in range(n)]
        bad_value = list(good)
        bad_value[rng.randrange(n)] = rng.choice((-1, 2, 3, 255, 1 << 20))
        return [
            Case(formula, tuple(good[: rng.randrange(n)])),
            Case(formula, tuple(good + [rng.randrange(2)])),
            Case(formula, tuple(bad_value)),
            Case(formula, tuple(good)),
        ]
    if workload == "wide-clauses":
        # Clause widths beyond bfloat16's exact-integer range (256).
        formula = wide(rng, 600, 4, rng.choice((300, 513)))
        n = formula.num_vars
        cases_ = [Case(formula, tuple(rng.randrange(2) for _ in range(n)))]
        # Assignments that falsify a clause or leave exactly one true literal.
        for clause in formula.clauses:
            falsify = [rng.randrange(2) for _ in range(n)]
            for lit in clause:
                falsify[abs(lit) - 1] = 0 if lit > 0 else 1
            cases_.append(Case(formula, tuple(falsify)))
            one = list(falsify)
            lit = rng.choice(clause)
            one[abs(lit) - 1] ^= 1
            cases_.append(Case(formula, tuple(one)))
        return cases_
    raise ValueError(f"unknown workload {workload!r}")
