"""CNF assignment verifier: the reference oracle for E08.

Specification ``cnf/v1``:

- Formula: ``num_vars >= 1`` Boolean variables numbered 1 to ``num_vars``
  and a tuple of clauses. A clause is a tuple of nonzero ``int`` literals
  (DIMACS style): ``v`` means variable ``v`` is true, ``-v`` that it is
  false, with ``1 <= |literal| <= num_vars``. Empty clauses, repeated
  literals, and tautologies are allowed; an empty clause is never satisfied.
  A malformed formula raises ``ValueError`` at construction.
- Action: ``check(assignment)``. An assignment is a sequence of ``int``
  values, one per variable, where ``assignment[v - 1]`` is 1 for true and 0
  for false. Elements that are not ``int`` (``bool`` included) are outside
  the schema and raise ``TypeError``.
- Validation precedes evaluation and checks, in order: the length equals
  ``num_vars``, then every value is 0 or 1. The first failed check yields a
  ``fault`` event naming it; no clause is evaluated.
- Result: a clause is satisfied when at least one of its literals is true.
  A valid assignment yields an ``output`` event with value 1 (accept) when
  every clause is satisfied and 0 (reject) otherwise, plus the number of
  unsatisfied clauses and the index of the first one (``None`` when none).
- Work: one unit per clause evaluated. Every clause is evaluated, so a valid
  check costs ``len(clauses)`` units whatever the outcome; a fault costs 0.
- The verifier is stateless: there is nothing to snapshot between checks.

Accepting is the only claim the verifier makes. A rejection says nothing
about whether another assignment satisfies the formula.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from latent_machines.machines.events import Event, EventKind

VERSION = "cnf/v1"


class FaultReason(Enum):
    WRONG_LENGTH = "wrong_length"
    VALUE_OUT_OF_RANGE = "value_out_of_range"


@dataclass(frozen=True, slots=True)
class Verdict:
    event: Event
    unsatisfied: int | None  # None for faults
    first_unsatisfied: int | None
    work: int

    @property
    def accepted(self) -> bool:
        return self.event.kind is EventKind.OUTPUT and self.event.value == 1


@dataclass(frozen=True, slots=True)
class Formula:
    num_vars: int
    clauses: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.num_vars) is not int or self.num_vars < 1:
            raise ValueError(f"num_vars must be a positive int, got {self.num_vars!r}")
        for index, clause in enumerate(self.clauses):
            if not isinstance(clause, tuple):
                raise ValueError(f"clause {index} is not a tuple")
            for literal in clause:
                if type(literal) is not int or not 1 <= abs(literal) <= self.num_vars:
                    raise ValueError(f"clause {index} has bad literal {literal!r}")

    @property
    def num_clauses(self) -> int:
        return len(self.clauses)


def check(formula: Formula, assignment: Sequence[int]) -> Verdict:
    """Evaluate ``assignment`` against every clause of ``formula``."""
    for value in assignment:
        if type(value) is not int:
            raise TypeError(f"assignment values must be int, got {value!r}")
    if len(assignment) != formula.num_vars:
        return _fault(FaultReason.WRONG_LENGTH)
    if any(value not in (0, 1) for value in assignment):
        return _fault(FaultReason.VALUE_OUT_OF_RANGE)
    unsatisfied, first = 0, None
    for index, clause in enumerate(formula.clauses):
        if not any(
            assignment[abs(literal) - 1] == (1 if literal > 0 else 0)
            for literal in clause
        ):
            unsatisfied += 1
            if first is None:
                first = index
    event = Event(EventKind.OUTPUT, value=1 if unsatisfied == 0 else 0)
    return Verdict(event, unsatisfied, first, formula.num_clauses)


def _fault(reason: FaultReason) -> Verdict:
    return Verdict(Event(EventKind.FAULT, reason=reason), None, None, 0)
