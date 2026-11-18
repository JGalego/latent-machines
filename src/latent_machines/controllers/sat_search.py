"""Search procedures for E08: conventional solvers and candidate proposers.

All procedures share one budget unit, a *step*:

- ``dpll``: one variable assignment, by decision or unit propagation. DPLL is
  complete: within its budget it returns a satisfying assignment or proves
  the formula unsatisfiable (every branch ends in a conflict).
- ``walksat``: one variable flip. Each flip yields a new full assignment,
  checked incrementally by the procedure's own clause counts.
- Proposers (``random`` and ``energy``): one candidate assignment handed to
  a verifier. The energy proposer runs ``chains`` gradient-descent chains
  with Langevin noise on a continuous relaxation of the clause energy and
  rounds every iterate to a candidate.

A procedure's own claim of a solution is never trusted: callers verify it
with the ``cnf/v1`` oracle. Only DPLL can report ``unsat``; the others report
``unknown`` when the budget runs out, which is not evidence of
unsatisfiability.
"""

import hashlib
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from latent_machines.backends.cnf_tensor import TensorVerifier
from latent_machines.machines import cnf
from latent_machines.machines.cnf import Formula


@dataclass
class Outcome:
    status: str  # "sat", "unsat" (DPLL only), or "unknown" (budget exhausted)
    steps: int
    assignment: list[int] | None = None
    verifier_calls: int = 0
    rejected: int = 0
    unsatisfied_total: int = 0  # summed over verified candidates
    propose_seconds: float = 0.0
    verify_seconds: float = 0.0
    decisions: str = ""  # sha256 of every (accepted, unsatisfied) decision
    extra: dict = field(default_factory=dict)


class BudgetExhausted(Exception):
    pass


# ---------- DPLL ----------


def dpll(formula: Formula, budget: int) -> Outcome:
    """Complete DPLL with unit propagation; branches on the shortest clause."""
    n = formula.num_vars
    values: list[int | None] = [None] * (n + 1)
    trail: list[int] = []
    steps = 0

    def assign(var: int, value: int) -> None:
        nonlocal steps
        if steps >= budget:
            raise BudgetExhausted
        steps += 1
        values[var] = value
        trail.append(var)

    def propagate() -> tuple[bool, tuple[int, ...] | None]:
        """Unit-propagate; return (no conflict, shortest open clause)."""
        while True:
            changed, shortest, shortest_open = False, None, n + 1
            for clause in formula.clauses:
                open_literals = []
                satisfied = False
                for lit in clause:
                    value = values[abs(lit)]
                    if value is None:
                        open_literals.append(lit)
                    elif value == (lit > 0):
                        satisfied = True
                        break
                if satisfied:
                    continue
                if not open_literals:
                    return False, None
                if len(open_literals) == 1:
                    lit = open_literals[0]
                    assign(abs(lit), 1 if lit > 0 else 0)
                    changed = True
                elif len(open_literals) < shortest_open:
                    shortest, shortest_open = tuple(open_literals), len(open_literals)
            if not changed:
                return True, shortest

    def undo(mark: int) -> None:
        while len(trail) > mark:
            values[trail.pop()] = None

    def solve() -> bool:
        ok, clause = propagate()
        if not ok:
            return False
        if clause is None:
            return True  # every clause satisfied
        lit = clause[0]
        for value in (1, 0) if lit > 0 else (0, 1):
            mark = len(trail)
            assign(abs(lit), value)
            if solve():
                return True
            undo(mark)
        return False

    try:
        found = solve()
    except BudgetExhausted:
        return Outcome("unknown", steps)
    if not found:
        return Outcome("unsat", steps)
    return Outcome("sat", steps, [values[v] or 0 for v in range(1, n + 1)])


# ---------- WalkSAT ----------


def walksat(
    formula: Formula, budget: int, rng: random.Random, noise: float = 0.5
) -> Outcome:
    n = formula.num_vars
    clauses = formula.clauses
    occurs: list[list[int]] = [[] for _ in range(n + 1)]
    for c, clause in enumerate(clauses):
        for lit in clause:
            occurs[abs(lit)].append(c)
    x = [rng.randrange(2) for _ in range(n + 1)]

    def true(lit: int) -> bool:
        return x[abs(lit)] == (lit > 0)

    counts = [sum(true(lit) for lit in clause) for clause in clauses]
    unsat = [c for c, k in enumerate(counts) if k == 0]
    where = {c: i for i, c in enumerate(unsat)}

    def flip(var: int) -> None:
        x[var] ^= 1
        for c in occurs[var]:
            before = counts[c]
            counts[c] = sum(true(lit) for lit in clauses[c])
            if before == 0 and counts[c] > 0:
                i = where.pop(c)
                last = unsat.pop()
                if last != c:
                    unsat[i], where[last] = last, i
            elif before > 0 and counts[c] == 0:
                where[c] = len(unsat)
                unsat.append(c)

    def breaks(var: int) -> int:
        # Clauses whose only true literal is on ``var``.
        return sum(
            1
            for c in occurs[var]
            if counts[c] == 1
            and any(abs(lit) == var and true(lit) for lit in clauses[c])
        )

    steps = 0
    while unsat:
        if steps >= budget:
            return Outcome("unknown", steps)
        clause = clauses[rng.choice(unsat)]
        if not clause:
            return Outcome("unknown", budget)  # an empty clause can never be fixed
        variables = [abs(lit) for lit in clause]
        if rng.random() < noise:
            var = rng.choice(variables)
        else:
            scored = [(breaks(v), v) for v in variables]
            best = min(score for score, _ in scored)
            var = rng.choice([v for score, v in scored if score == best])
        flip(var)
        steps += 1
    return Outcome("sat", steps, x[1:])


# ---------- proposers plus a verifier ----------


def random_proposals(
    formula: Formula, chains: int, seed: int
) -> Callable[[], torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)

    def propose() -> torch.Tensor:
        return torch.randint(0, 2, (chains, formula.num_vars), generator=generator)

    return propose


def energy_proposals(
    formula: Formula,
    chains: int,
    seed: int,
    learning_rate: float = 1.0,
    temperature: float = 0.02,
) -> Callable[[], torch.Tensor]:
    """Langevin descent on E(y) = sum_c prod_j (1 - q_cj), q = P(literal true).

    Variables are ``p = sigmoid(y)``. Each call takes one step on every chain
    and returns the rounded iterates ``y > 0`` as candidates.
    """
    widths = {len(clause) for clause in formula.clauses}
    if len(widths) != 1:
        raise ValueError("energy proposals need clauses of one width")
    generator = torch.Generator().manual_seed(seed)
    index = torch.tensor([[abs(lit) - 1 for lit in c] for c in formula.clauses])
    positive = torch.tensor([[lit > 0 for lit in c] for c in formula.clauses])
    y = torch.randn(chains, formula.num_vars, generator=generator)
    scale = (2 * learning_rate * temperature) ** 0.5

    def propose() -> torch.Tensor:
        nonlocal y
        y = y.detach().requires_grad_(True)
        p = torch.sigmoid(y)[:, index]  # (chains, m, k)
        q = torch.where(positive, p, 1 - p)
        energy = (1 - q).prod(dim=2).sum()
        (grad,) = torch.autograd.grad(energy, y)
        noise = torch.randn(y.shape, generator=generator)
        y = (y - learning_rate * grad + scale * noise).clamp(-6, 6).detach()
        return (y > 0).to(torch.int64)

    return propose


def run_proposals(
    formula: Formula,
    propose: Callable[[], torch.Tensor],
    verifier: TensorVerifier | None,
    budget: int,
) -> Outcome:
    """Verify proposals until one is accepted or ``budget`` candidates are used.

    ``verifier=None`` uses the ``cnf/v1`` oracle on Python lists; otherwise the
    compiled verifier checks each batch in one evaluation. Candidates within a
    batch are taken in order, so both verifiers see the same decision sequence.
    """
    digest = hashlib.sha256()
    outcome = Outcome("unknown", 0)
    while outcome.steps < budget:
        started = time.perf_counter()
        batch = propose()[: budget - outcome.steps]
        outcome.propose_seconds += time.perf_counter() - started

        started = time.perf_counter()
        if verifier is None:
            rows = batch.tolist()
            unsatisfied = [cnf.check(formula, row).unsatisfied for row in rows]
        else:
            unsatisfied = verifier.unsatisfied(batch).tolist()
        outcome.verify_seconds += time.perf_counter() - started

        for row, count in enumerate(unsatisfied):
            assert count is not None
            outcome.steps += 1
            outcome.verifier_calls += 1
            outcome.unsatisfied_total += count
            digest.update(f"{int(count == 0)},{count};".encode())
            if count == 0:
                outcome.status = "sat"
                outcome.assignment = batch[row].tolist()
                outcome.decisions = digest.hexdigest()
                return outcome
            outcome.rejected += 1
    outcome.decisions = digest.hexdigest()
    return outcome
