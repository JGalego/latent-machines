"""Aggregate the four-fold unseen-statement runs of E09.

Each run is an ``e09 --held-out --fold k`` record. This module reads a set of
SmolLM2 runs and a set of random-model runs, and evaluates the predictions of
"Amendment 2026-11-25: more unseen statements" in EXPERIMENTS.md:

1. SmolLM2's S gets at least 0.90 of the 32 unseen statements right, in the
   mean over folds and seeds.
2. No fold falls below 0.50 in the mean over seeds.
3. SmolLM2's S beats the random model's S by more than 0.10, in the mean over
   folds and seeds.

The sanity check is S's action accuracy on seen statements, at least 0.99 in
every fold. The module also lists, for each unseen statement, how many seeds
chose its action correctly.

    uv run --locked python -m latent_machines.experiments.e09_folds \\
        --smollm results/E09/RUN ... --random results/E09/controls/RUN ...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from latent_machines.experiments.e09 import FOLDS, SEEN_MIN, UNSEEN_MIN, mean

MODULE = "latent_machines.experiments.e09_folds"
FOLD_FLOOR = 0.50  # prediction 2: no fold below this
OVER_RANDOM = 0.10  # prediction 3 margin


def load(paths: list[Path]) -> dict[int, list[dict[str, Any]]]:
    """Per-seed statement measures of each run, grouped by fold."""
    folds: dict[int, list[dict[str, Any]]] = {}
    for path in paths:
        run = json.loads((path / "run.json").read_text())
        metrics = json.loads((path / "metrics.json").read_text())
        if metrics["status"] != "completed":
            raise SystemExit(f"{path}: status {metrics['status']}")
        fold = run["task"].get("fold", 0)
        folds.setdefault(fold, []).extend(metrics["statement_accuracy"])
    return folds


def fold_mean(folds: dict[int, list[dict[str, Any]]], key: str) -> dict[int, float]:
    return {fold: mean([s[key] for s in rows]) for fold, rows in sorted(folds.items())}


def per_statement(rows: list[dict[str, Any]], fold: int) -> dict[str, int]:
    """Seeds that chose each unseen statement's action correctly."""
    statements = [
        f"Set {'abcdefgh'[record]} to {value}." for record, value in sorted(FOLDS[fold])
    ]
    failed = [f["statement"] for row in rows for f in row["failed_unseen"]]
    return {s: len(rows) - failed.count(s) for s in statements}


def evaluate(
    smollm: dict[int, list[dict[str, Any]]], random: dict[int, list[dict[str, Any]]]
) -> dict[str, Any]:
    if sorted(smollm) != sorted(FOLDS) or sorted(random) != sorted(FOLDS):
        raise SystemExit("every fold needs a SmolLM2 run and a random-model run")
    unseen, seen = fold_mean(smollm, "unseen"), fold_mean(smollm, "seen")
    unseen_random = fold_mean(random, "unseen")
    overall, overall_random = (
        mean(list(unseen.values())),
        mean(list(unseen_random.values())),
    )
    return {
        "1_s_reads_unseen_statements": {
            "unseen_action_accuracy": overall,
            "minimum": UNSEEN_MIN,
            "met": overall >= UNSEEN_MIN,
            "counts_against_reading": overall < 0.5,
            "inconclusive": 0.5 <= overall < UNSEEN_MIN,
        },
        "2_no_fold_below_half": {
            "per_fold": unseen,
            "floor": FOLD_FLOOR,
            "met": all(v >= FOLD_FLOOR for v in unseen.values()),
        },
        "3_pretraining_adds_over_random": {
            "smollm2": overall,
            "random": overall_random,
            "random_per_fold": unseen_random,
            "difference": overall - overall_random,
            "margin": OVER_RANDOM,
            "met": overall - overall_random > OVER_RANDOM,
        },
        "sanity_seen_statements": {
            "per_fold": seen,
            "minimum": SEEN_MIN,
            "ok": all(v >= SEEN_MIN for v in seen.values()),
        },
        "statements": {
            "smollm2": {f: per_statement(r, f) for f, r in smollm.items()},
            "random": {f: per_statement(r, f) for f, r in random.items()},
        },
        "seeds": {fold: len(rows) for fold, rows in smollm.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m {MODULE}")
    parser.add_argument("--smollm", nargs="+", type=Path, required=True)
    parser.add_argument("--random", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, help="write the evaluation here as JSON")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    result = evaluate(load(args.smollm), load(args.random))
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
