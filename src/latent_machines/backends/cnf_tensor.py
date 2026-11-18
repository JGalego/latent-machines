"""Compiled tensor verifier for ``cnf/v1``.

A formula with ``m`` clauses over ``n`` variables compiles to clause-literal
incidence matrices: ``P[c, v]`` counts the occurrences of literal ``v`` in
clause ``c`` and ``N[c, v]`` those of ``-v``. For a 0/1 assignment ``x``,
the number of true literal occurrences in each clause is

    incidence:  count = P x + N (1 - x)        two products, no subtraction
    folded:     count = (P - N) x + N 1        one product plus a constant

and the assignment is accepted iff every count is at least 1. Both lowerings
are exact while every partial sum is an integer the dtype represents exactly:
counts up to 256 in bfloat16, 2048 in float16, and 2^24 in float32. The
folded lowering adds signed terms, so rounding of a large intermediate can
change a small final count; the incidence lowering only adds nonnegative
terms, so a zero count stays zero.

Validation (length, then 0/1 values) runs on an integer tensor before the
cast, so out-of-range values are never rounded into range. ``residue`` is
the largest distance of a count from an integer, measured before
thresholding.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from latent_machines.machines.cnf import FaultReason, Formula, Verdict
from latent_machines.machines.events import Event, EventKind

LOWERINGS = ("incidence", "folded")
THRESHOLD = 0.5  # counts are integers; a satisfied clause has count >= 1


@dataclass(frozen=True)
class BatchResult:
    verdicts: list[Verdict]
    residue: float


class TensorVerifier:
    def __init__(
        self,
        formula: Formula,
        lowering: str = "incidence",
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        if lowering not in LOWERINGS:
            raise ValueError(f"unknown lowering {lowering!r}")
        self.formula, self.lowering, self.dtype = formula, lowering, dtype
        self.device = torch.device(device)
        m, n = formula.num_clauses, formula.num_vars
        positive = torch.zeros(m, n, dtype=torch.float64)
        negative = torch.zeros(m, n, dtype=torch.float64)
        for c, clause in enumerate(formula.clauses):
            for literal in clause:
                target = positive if literal > 0 else negative
                target[c, abs(literal) - 1] += 1
        if lowering == "incidence":
            self.weights = (
                positive.T.to(dtype).to(self.device),
                negative.T.to(dtype).to(self.device),
            )
            self.bias = None
        else:
            self.weights = ((positive - negative).T.to(dtype).to(self.device),)
            self.bias = negative.sum(1).to(dtype).to(self.device)

    def stats(self) -> dict[str, int]:
        tensors = [*self.weights] + ([self.bias] if self.bias is not None else [])
        return {
            "stored_weights": sum(t.numel() for t in tensors),
            "nonzero_weights": sum(int(t.count_nonzero()) for t in tensors),
            "weight_bytes": sum(t.numel() * t.element_size() for t in tensors),
        }

    def counts(self, x: torch.Tensor) -> torch.Tensor:
        """True-literal counts ``(batch, m)`` for a 0/1 tensor ``(batch, n)``."""
        x = x.to(self.dtype)
        if self.lowering == "incidence":
            positive, negative = self.weights
            return x @ positive + (1 - x) @ negative
        (signed,) = self.weights
        assert self.bias is not None
        return x @ signed + self.bias

    def check_batch(self, assignments: Sequence[Sequence[int]]) -> BatchResult:
        """Verdicts for several assignments, with one batched evaluation."""
        n, m = self.formula.num_vars, self.formula.num_clauses
        verdicts: list[Verdict | None] = [None] * len(assignments)
        rows, valid = [], []
        for i, assignment in enumerate(assignments):
            for value in assignment:
                if type(value) is not int:
                    raise TypeError(f"assignment values must be int, got {value!r}")
            if len(assignment) != n:
                verdicts[i] = _fault(FaultReason.WRONG_LENGTH)
                continue
            rows.append(list(assignment))
            valid.append(i)
        residue = 0.0
        if rows:
            x = torch.tensor(rows, dtype=torch.int64, device=self.device)
            in_range = ((x == 0) | (x == 1)).all(dim=1)
            counts = self.counts(x)
            exact = counts.double()
            residue_rows = (exact - exact.round()).abs()
            unsatisfied = counts < THRESHOLD
            totals = unsatisfied.sum(dim=1)
            # First unsatisfied clause: argmax returns the first maximum.
            firsts = (
                unsatisfied.to(torch.uint8).argmax(dim=1)
                if m
                else torch.zeros(len(rows), dtype=torch.int64)
            )
            for row, i in enumerate(valid):
                if not in_range[row]:
                    verdicts[i] = _fault(FaultReason.VALUE_OUT_OF_RANGE)
                    continue
                if m:
                    residue = max(residue, float(residue_rows[row].max()))
                total = int(totals[row])
                verdicts[i] = Verdict(
                    Event(EventKind.OUTPUT, value=1 if total == 0 else 0),
                    total,
                    int(firsts[row]) if total else None,
                    m,
                )
        assert all(v is not None for v in verdicts)
        return BatchResult([v for v in verdicts if v is not None], residue)

    def unsatisfied(self, x: torch.Tensor) -> torch.Tensor:
        """Unsatisfied-clause counts ``(batch,)`` for an integer 0/1 tensor.

        The fast path for search, with the same arithmetic as ``check_batch``
        but no per-row verdict objects. Values other than 0 and 1 raise.
        """
        if (
            x.dtype != torch.int64
            or x.dim() != 2
            or x.shape[1] != self.formula.num_vars
        ):
            raise ValueError("expected an int64 tensor of shape (batch, num_vars)")
        if not ((x == 0) | (x == 1)).all():
            raise ValueError("assignment values must be 0 or 1")
        return (self.counts(x) < THRESHOLD).sum(dim=1)

    def check(self, assignment: Sequence[int]) -> Verdict:
        return self.check_batch([assignment]).verdicts[0]


def _fault(reason: FaultReason) -> Verdict:
    return Verdict(Event(EventKind.FAULT, reason=reason), None, None, 0)
