"""The compiled tensor verifier against the cnf/v1 oracle."""

import random

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends.cnf_tensor import LOWERINGS, TensorVerifier  # noqa: E402
from latent_machines.machines import cnf  # noqa: E402
from latent_machines.machines.cnf import Formula  # noqa: E402
from latent_machines.tasks import sat_instances  # noqa: E402

DTYPES = (torch.float32, torch.bfloat16, torch.float16)


@pytest.mark.parametrize("lowering", LOWERINGS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "workload", ["random", "near-miss", "enumeration", "malformed"]
)
def test_matches_oracle_within_limits(
    lowering: str, dtype: torch.dtype, workload: str
) -> None:
    rng = random.Random(f"test:{workload}")
    backend = None
    for case in sat_instances.cases(workload, 100, rng):
        if backend is None or backend.formula is not case.formula:
            backend = TensorVerifier(case.formula, lowering, dtype)
        expected = cnf.check(case.formula, case.assignment)
        assert backend.check(case.assignment) == expected


def test_edge_formulas() -> None:
    for formula, assignment in [
        (Formula(1, ((),)), (1,)),
        (Formula(1, ((1, -1),)), (0,)),
        (Formula(2, ((1, 1, 1),)), (0, 0)),
        (Formula(2, ()), (1, 0)),
    ]:
        for lowering in LOWERINGS:
            backend = TensorVerifier(formula, lowering)
            assert backend.check(assignment) == cnf.check(formula, assignment)


def test_folded_bfloat16_loses_wide_clauses_but_incidence_does_not() -> None:
    # 300 negative literals plus one positive literal, its only true literal:
    # the count is 1, but 300 is beyond bfloat16's exact-integer range, so the
    # folded sum can round the 1 away.
    n = 301
    formula = Formula(n, (tuple(-v for v in range(1, n)) + (n,),))
    assignment = (1,) * n
    assert cnf.check(formula, assignment).accepted
    incidence = TensorVerifier(formula, "incidence", torch.bfloat16)
    assert incidence.check(assignment).accepted
    folded = TensorVerifier(formula, "folded", torch.bfloat16)
    assert not folded.check(assignment).accepted  # the documented limit


def test_fast_path_matches_batch_verdicts() -> None:
    rng = random.Random(1)
    formula = sat_instances.uniform(rng, 30, 4.26)
    rows = [[rng.randrange(2) for _ in range(30)] for _ in range(64)]
    backend = TensorVerifier(formula)
    fast = backend.unsatisfied(torch.tensor(rows)).tolist()
    assert fast == [v.unsatisfied for v in backend.check_batch(rows).verdicts]
    with pytest.raises(ValueError):
        backend.unsatisfied(torch.full((1, 30), 2))
