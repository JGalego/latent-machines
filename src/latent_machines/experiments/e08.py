"""E08: heuristic search with an exact verifier, on 3-SAT.

Two parts, recorded in one run:

1. Verifier conformance. The compiled tensor verifier (``backends.cnf_tensor``,
   two lowerings, several dtypes) checks the same candidate assignments as the
   ``cnf/v1`` oracle; every verdict (event, unsatisfied-clause count, first
   unsatisfied clause, work) must match. A variant is one lowering, dtype, and
   workload. Each variant records whether its formulas stay within the
   lowering's declared exact-integer range; the run passes only if every
   variant within that range matches, and mismatches outside it are reported
   as findings.
2. Search at a fixed step budget per instance (see ``controllers.sat_search``):
   DPLL, WalkSAT, random proposals with the oracle verifier, and energy-based
   proposals with the oracle verifier and with the compiled verifier. Every
   claimed solution is re-checked by the oracle; the two energy arms must make
   identical accept decisions; DPLL's unsatisfiability claims are confirmed by
   exhaustive enumeration where the instance has at most 20 variables.

    uv run --locked python -m latent_machines.experiments.e08

Exit status: 0 when the run passes, 1 on a failed check, 2 if the run failed.
"""

import argparse
import itertools
import json
import random
import resource
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any

import torch

from latent_machines.backends.cnf_tensor import LOWERINGS, TensorVerifier
from latent_machines.controllers import sat_search
from latent_machines.experiments import lockstep, records
from latent_machines.machines import cnf
from latent_machines.machines.cnf import Formula, Verdict
from latent_machines.tasks import sat_instances

EXPERIMENT = "E08"
MODULE = "latent_machines.experiments.e08"
DTYPES = ("float32", "bfloat16", "float16")
EXACT_INTEGER_LIMIT = {
    "float32": 2**24,
    "float64": 2**53,
    "bfloat16": 256,
    "float16": 2048,
}
CLASSES = {
    # name: (family, variables, clause/variable ratio)
    "planted-50": ("planted", 50, 4.2),
    "planted-100": ("planted", 100, 4.2),
    "uniform-40": ("uniform", 40, 4.26),
    "uniform-20-dense": ("uniform", 20, 6.0),
}
ARMS = (
    "dpll",
    "walksat",
    "random+oracle",
    "energy+oracle",
    "energy+tensor/float32",
    "energy+tensor/bfloat16",
)
ENUMERATION_LIMIT = 20  # variables; DPLL unsat claims above this are unchecked
ACCEPTANCE = (
    "verifier: zero verdict mismatches against cnf/v1 on every variant within its "
    "declared exact-integer range, in particular no invalid assignment accepted; "
    "search: every claimed solution accepted by the oracle, identical accept "
    "decisions for the energy proposals under both verifiers, and every checked "
    "DPLL unsatisfiability claim confirmed by enumeration"
)
NOT_MEASURED = [
    "optimized conventional verifier: the oracle is a plain Python loop, so "
    "verification-time ratios are against that implementation only",
    "CDCL or other modern complete solvers; DPLL here has no clause learning",
    "DPLL unsatisfiability claims for instances above 20 variables are not "
    "independently confirmed",
    "learned (trained) proposers; the energy proposer is a fixed relaxation",
    "GPU execution and batched verification across instances",
]


@dataclass(frozen=True)
class Variant:
    lowering: str
    dtype: str
    workload: str
    horizon: int
    seed: int
    trace_limit: int

    @property
    def name(self) -> str:
        return f"cnf-tensor-{self.lowering}/{self.dtype}/{self.workload}"


def verdict_json(verdict: Verdict) -> dict[str, Any]:
    return {
        "event": lockstep.event_json(verdict.event),
        "unsatisfied": verdict.unsatisfied,
        "first_unsatisfied": verdict.first_unsatisfied,
        "work": verdict.work,
    }


def formula_json(formula: Formula) -> dict[str, Any]:
    return {"num_vars": formula.num_vars, "clauses": [list(c) for c in formula.clauses]}


def reproduce_command(variant: Variant) -> list[str]:
    return [
        "python", "-m", MODULE, "--lowerings", variant.lowering,
        "--dtypes", variant.dtype, "--workloads", variant.workload,
        "--horizon", str(variant.horizon), "--seed", str(variant.seed),
        "--instances", "0",
    ]  # fmt: skip


def run_variant(variant: Variant) -> dict[str, Any]:
    """Check one lowering, dtype, and workload against the oracle."""
    dtype = getattr(torch, variant.dtype)
    rng = random.Random(f"{variant.seed}:{variant.workload}")
    groups: list[tuple[Formula, list[tuple[int, ...]]]] = []
    for case in sat_instances.cases(variant.workload, variant.horizon, rng):
        if groups and groups[-1][0] is case.formula:
            groups[-1][1].append(case.assignment)
        else:
            groups.append((case.formula, [case.assignment]))

    trace: list[dict[str, Any]] = []
    failure = None
    checked = mismatches = invalid_accepted = valid_rejected = 0
    first_mismatch = None
    max_residue, max_width, compile_seconds, verify_seconds = 0.0, 0, 0.0, 0.0
    kinds: dict[str, int] = {}
    started = time.perf_counter()
    for formula, assignments in groups:
        max_width = max([max_width, *(len(c) for c in formula.clauses)])
        t = time.perf_counter()
        backend = TensorVerifier(formula, variant.lowering, dtype)
        compile_seconds += time.perf_counter() - t
        t = time.perf_counter()
        batch = backend.check_batch(assignments)
        verify_seconds += time.perf_counter() - t
        max_residue = max(max_residue, batch.residue)
        for assignment, actual in zip(assignments, batch.verdicts, strict=True):
            expected = cnf.check(formula, assignment)
            kind = (
                expected.event.reason.value
                if expected.event.reason is not None
                else ("accept" if expected.accepted else "reject")
            )
            kinds[kind] = kinds.get(kind, 0) + 1
            ok = actual == expected
            if actual.accepted and not expected.accepted:
                invalid_accepted += 1
            if expected.accepted and not actual.accepted:
                valid_rejected += 1
            if checked < variant.trace_limit or not ok:
                trace.append(
                    {
                        "variant": variant.name,
                        "index": checked,
                        "num_vars": formula.num_vars,
                        "num_clauses": formula.num_clauses,
                        "verdict": verdict_json(actual),
                    }
                )
            if not ok:
                mismatches += 1
                if failure is None:
                    first_mismatch = checked
                    failure = {
                        "variant": variant.name,
                        "action_index": checked,
                        "formula": formula_json(formula),
                        "assignment": list(assignment),
                        "expected": verdict_json(expected),
                        "actual": verdict_json(actual),
                        "dtype": variant.dtype,
                        "lowering": variant.lowering,
                        "reproduce": reproduce_command(variant),
                    }
            checked += 1
    seconds = time.perf_counter() - started
    within = max_width <= EXACT_INTEGER_LIMIT[variant.dtype]
    if mismatches:
        classification = "mismatch"
    elif max_residue == 0.0:
        classification = "exact"
    else:
        classification = "conforms with residue"
    stats = (
        TensorVerifier(groups[0][0], variant.lowering, dtype).stats() if groups else {}
    )
    return {
        "metrics": {
            "variant": variant.name,
            "machine": "cnf/v1",
            "backend": f"cnf-tensor-{variant.lowering}",
            "lowering": variant.lowering,
            "workload": variant.workload,
            "dtype": variant.dtype,
            "unit": "candidate assignment",
            "actions_checked": checked,
            "mismatches": mismatches,
            "first_mismatch": first_mismatch,
            "invalid_accepted": invalid_accepted,
            "valid_rejected": valid_rejected,
            "max_residue": max_residue,
            "classification": classification,
            "snapshot_resume": "not applicable",
            "max_clause_width": max_width,
            "exact_integer_limit": EXACT_INTEGER_LIMIT[variant.dtype],
            "within_declared_limits": within,
            "outcomes": dict(sorted(kinds.items())),
            "formulas": len(groups),
            "compile_seconds": round(compile_seconds, 4),
            "verify_seconds": round(verify_seconds, 4),
            "end_to_end_seconds": round(seconds, 3),
            "first_formula_weights": stats,
        },
        "trace": trace,
        "failure": failure,
    }


# ---------- search ----------


def instance(name: str, index: int, seed: int) -> tuple[Formula, bool | None]:
    """The formula and its known satisfiability (True if planted, else None)."""
    family, n, ratio = CLASSES[name]
    rng = random.Random(f"{seed}:{name}:{index}")
    if family == "planted":
        formula, _ = sat_instances.planted(rng, n, ratio)
        return formula, True
    return sat_instances.uniform(rng, n, ratio), None


def unsat_by_enumeration(formula: Formula) -> bool:
    """True iff no assignment satisfies ``formula``, by checking all of them."""
    verifier = TensorVerifier(formula, "incidence", torch.float32)
    n = formula.num_vars
    chunk = 1 << min(n, 16)
    low = torch.tensor(list(itertools.product((0, 1), repeat=min(n, 16))))
    for high in itertools.product((0, 1), repeat=n - min(n, 16)):
        rows = torch.cat([torch.tensor(high).expand(chunk, -1), low], dim=1)
        if (verifier.unsatisfied(rows.to(torch.int64)) == 0).any():
            return False
    return True


def run_arm(arm: str, formula: Formula, budget: int, chains: int, seed: int) -> Any:
    if arm == "dpll":
        return sat_search.dpll(formula, budget)
    if arm == "walksat":
        return sat_search.walksat(formula, budget, random.Random(seed))
    proposer, _, verifier = arm.partition("+")
    propose = (
        sat_search.random_proposals(formula, chains, seed)
        if proposer == "random"
        else sat_search.energy_proposals(formula, chains, seed)
    )
    tensor = None
    if verifier.startswith("tensor/"):
        tensor = TensorVerifier(formula, "incidence", getattr(torch, verifier[7:]))
    return sat_search.run_proposals(formula, propose, tensor, budget)


def search(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    checks = {
        "claimed_solutions": 0,
        "invalid_solutions": 0,
        "energy_decision_mismatches": 0,
        "unsat_claims": 0,
        "unsat_claims_confirmed": 0,
        "unsat_claims_refuted": 0,
        "unsat_claims_unchecked": 0,
        "heuristic_unsat_claims": 0,
    }
    for name in CLASSES:
        for index in range(args.instances):
            formula, planted = instance(name, index, args.seed)
            seed = args.seed * 100_003 + index
            decisions = {}
            for arm in args.arms.split(","):
                started = time.perf_counter()
                outcome = run_arm(arm, formula, args.budget, args.chains, seed)
                seconds = time.perf_counter() - started
                verified = None
                if outcome.status == "sat":
                    checks["claimed_solutions"] += 1
                    verified = cnf.check(formula, outcome.assignment).accepted
                    checks["invalid_solutions"] += not verified
                if outcome.status == "unsat":
                    if arm != "dpll":
                        checks["heuristic_unsat_claims"] += 1
                    checks["unsat_claims"] += 1
                    if formula.num_vars <= ENUMERATION_LIMIT:
                        confirmed = unsat_by_enumeration(formula)
                        checks["unsat_claims_confirmed"] += confirmed
                        checks["unsat_claims_refuted"] += not confirmed
                    else:
                        checks["unsat_claims_unchecked"] += 1
                if arm.startswith("energy+"):
                    decisions[arm] = outcome.decisions
                rows.append(
                    {
                        "class": name,
                        "instance": index,
                        "num_vars": formula.num_vars,
                        "num_clauses": formula.num_clauses,
                        "planted": planted,
                        "arm": arm,
                        "status": outcome.status,
                        "verified": verified,
                        "steps": outcome.steps,
                        "verifier_calls": outcome.verifier_calls
                        + (
                            1
                            if arm in ("dpll", "walksat") and verified is not None
                            else 0
                        ),
                        "rejected": outcome.rejected,
                        "mean_unsatisfied": (
                            outcome.unsatisfied_total / outcome.verifier_calls
                            if outcome.verifier_calls
                            else None
                        ),
                        "propose_seconds": round(outcome.propose_seconds, 4),
                        "verify_seconds": round(outcome.verify_seconds, 4),
                        "seconds": round(seconds, 4),
                    }
                )
            if len(set(decisions.values())) > 1:
                checks["energy_decision_mismatches"] += 1
            print(
                f"{name:<17} {index:>3}  "
                + "  ".join(
                    f"{r['arm']}={r['status']}"
                    for r in rows
                    if r["class"] == name and r["instance"] == index
                ),
                flush=True,
            )
    return {"rows": rows, "checks": checks}


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["arm"], row["class"]), []).append(row)
    summary = []
    for (arm, name), group in groups.items():
        calls = sum(r["verifier_calls"] for r in group)
        quality = [
            r["mean_unsatisfied"] for r in group if r["mean_unsatisfied"] is not None
        ]
        summary.append(
            {
                "arm": arm,
                "class": name,
                "instances": len(group),
                "solved": sum(
                    r["status"] == "sat" and bool(r["verified"]) for r in group
                ),
                "proved_unsat": sum(r["status"] == "unsat" for r in group),
                "unknown": sum(r["status"] == "unknown" for r in group),
                "steps": sum(r["steps"] for r in group),
                "verifier_calls": calls,
                "rejected": sum(r["rejected"] for r in group),
                "mean_unsatisfied_per_candidate": (
                    sum(quality) / len(quality) if quality else None
                ),
                "propose_seconds": round(sum(r["propose_seconds"] for r in group), 3),
                "verify_seconds": round(sum(r["verify_seconds"] for r in group), 3),
                "seconds": round(sum(r["seconds"] for r in group), 3),
            }
        )
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="E08: tensor verifier conformance and verified 3-SAT search.",
    )
    parser.add_argument(
        "--horizon", type=int, default=2000, help="candidates per variant"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lowerings", default=",".join(LOWERINGS))
    parser.add_argument("--dtypes", default=",".join(DTYPES))
    parser.add_argument("--workloads", default=",".join(sat_instances.WORKLOADS))
    parser.add_argument("--instances", type=int, default=10, help="instances per class")
    parser.add_argument("--budget", type=int, default=20_000, help="steps per instance")
    parser.add_argument("--chains", type=int, default=32, help="proposals per batch")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--trace-limit", type=int, default=200)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", default="runs")
    args = parser.parse_args(argv)
    for option, names, allowed in (
        ("--lowerings", args.lowerings, LOWERINGS),
        ("--dtypes", args.dtypes, tuple(EXACT_INTEGER_LIMIT)),
        ("--workloads", args.workloads, sat_instances.WORKLOADS),
        ("--arms", args.arms, ARMS),
    ):
        for name in names.split(","):
            if name not in allowed:
                parser.error(f"{option}: unknown {name!r}; choose from {list(allowed)}")
    if args.horizon < 1 or args.budget < 1 or args.chains < 1 or args.instances < 0:
        parser.error("sizes must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    run_dir, source, root = lockstep.start(args.out, "e08")
    variants = [
        Variant(lowering, dtype, workload, args.horizon, args.seed, args.trace_limit)
        for lowering in args.lowerings.split(",")
        for dtype in args.dtypes.split(",")
        for workload in args.workloads.split(",")
    ]
    header = lockstep.run_header(
        experiment=EXPERIMENT,
        module=MODULE,
        argv=argv,
        run_dir=run_dir,
        source=source,
        root=root,
        config=vars(args),
        device="cpu",
    )
    header["environment"]["torch_threads_per_variant"] = args.threads
    records.write_json(
        run_dir / "run.json",
        header
        | {
            "seeds": {
                "seed": args.seed,
                "derivation": "workloads: random.Random(f'{seed}:{workload}'); "
                "instances: random.Random(f'{seed}:{class}:{index}'); "
                "proposals and WalkSAT: seed * 100003 + index",
            },
            "machine": {"cnf/v1": "machines.cnf; stateless assignment verifier"},
            "backends": {
                "cnf-tensor-incidence": "count = P x + N (1 - x); accept iff all >= 1",
                "cnf-tensor-folded": "count = (P - N) x + N 1; accept iff all >= 1",
                "exact_integer_limit": EXACT_INTEGER_LIMIT,
            },
            "search": {
                "classes": {
                    name: {"family": f, "variables": n, "ratio": r}
                    for name, (f, n, r) in CLASSES.items()
                },
                "budget_unit": "dpll: variable assignment; walksat: flip; "
                "proposers: verified candidate",
                "energy_proposer": {
                    "learning_rate": 1.0,
                    "temperature": 0.02,
                    "chains": args.chains,
                    "selection": "best of four settings on six planted-50 development "
                    "instances drawn with a separate seed, not these instances",
                },
                "walksat_noise": 0.5,
            },
            "variants": [asdict(v) | {"name": v.name} for v in variants],
        },
    )

    results: list[dict[str, Any]] = []
    found: dict[str, Any] = {"rows": [], "checks": {}}
    error = None
    try:
        for variant in variants:
            results.append(run_variant(variant))
            m = results[-1]["metrics"]
            print(
                f"{m['variant']:<46} {m['actions_checked']:>6} checked  "
                f"{m['mismatches']} mismatches  residue {m['max_residue']:.3g}  "
                f"{'within' if m['within_declared_limits'] else 'beyond'} limits",
                flush=True,
            )
        if args.instances:
            found = search(args)
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)

    metrics = [r["metrics"] for r in results]
    with (run_dir / "trace.jsonl").open("w") as handle:
        for result in results:
            for line in result["trace"]:
                handle.write(json.dumps(line) + "\n")
    failures = [r["failure"] for r in results if r["failure"]]
    if failures:
        records.write_json(run_dir / "failure.json", failures)
    checks = found["checks"]
    in_contract_ok = all(
        m["mismatches"] == 0 for m in metrics if m["within_declared_limits"]
    )
    search_ok = not checks or (
        checks["invalid_solutions"] == 0
        and checks["energy_decision_mismatches"] == 0
        and checks["unsat_claims_refuted"] == 0
        and checks["heuristic_unsat_claims"] == 0
    )
    status = (
        "error" if error else ("passed" if in_contract_ok and search_ok else "failed")
    )
    records.write_json(
        run_dir / "metrics.json",
        {
            "schema_version": records.SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": status,
            "finished_at": records.now(),
            "acceptance": ACCEPTANCE,
            "totals": {
                "variants": len(variants),
                "variants_completed": len(metrics),
                "actions_checked": sum(m["actions_checked"] for m in metrics),
                "mismatches": sum(m["mismatches"] for m in metrics),
                "mismatches_within_declared_limits": sum(
                    m["mismatches"] for m in metrics if m["within_declared_limits"]
                ),
                "invalid_accepted": sum(m["invalid_accepted"] for m in metrics),
                "max_residue": max((m["max_residue"] for m in metrics), default=None),
            },
            "variants": metrics,
            "search_checks": checks,
            "search": summarize(found["rows"]),
            "search_instances": found["rows"],
            "peak_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "not_measured": NOT_MEASURED,
            "error": error,
        },
    )
    print(f"status: {status}\nrecord: {run_dir}")
    return {"passed": 0, "failed": 1}.get(status, 2)


if __name__ == "__main__":
    sys.exit(main())
