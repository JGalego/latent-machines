"""Evaluate the E03 reply-only amendment against the recorded run.

The amendment in EXPERIMENTS.md ("2026-11-25: reply-only vectors") fixes two
predictions on arm V, which carries only the value a read returned:

1. V's task accuracy exceeds S's by more than the conservative two-standard-
   error bound at 64 statements, in the mean over seeds.
2. V is within the bound of L at 64 statements.

S is rerun in the same job as V, so V is compared with an S from the same
code; the rerun S is also compared with the recorded S. If it differs by more
than the recorded seed range, prediction 2 is not evaluated.

    uv run --locked python -m latent_machines.experiments.e03_reply_only \\
        results/E03/e03-20261118T022125Z-fe9db1c-ff00/metrics.json \\
        results/E03/controls/<new run>/metrics.json
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

SPLITS = ("test_in_distribution", "test_length_32", "test_length_64")
LONG = "test_length_64"


def bound(questions: int) -> float:
    """E02's conservative two-standard-error bound for a difference of two
    proportions over ``questions`` questions."""
    return 2 * math.sqrt(0.5 / questions)


def by_seed(rows: list[dict[str, Any]], arm: str, split: str) -> dict[int, dict]:
    return {r["seed"]: r for r in rows if r["arm"] == arm and r["split"] == split}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def paired(
    a_rows: list[dict[str, Any]], a: str, b_rows: list[dict[str, Any]], b: str
) -> list[dict[str, Any]]:
    """Per-split paired differences ``a - b``, with the bound and the verdicts."""
    out = []
    for split in SPLITS:
        x, y = by_seed(a_rows, a, split), by_seed(b_rows, b, split)
        seeds = sorted(set(x) & set(y))
        if not seeds:
            continue
        diffs = [x[s]["task_accuracy"] - y[s]["task_accuracy"] for s in seeds]
        limit = bound(x[seeds[0]]["questions"])
        out.append(
            {
                "split": split,
                "seeds": seeds,
                "differences": diffs,
                "mean": mean(diffs),
                "bound": limit,
                "above_bound": mean(diffs) > limit,
                "within_bound": abs(mean(diffs)) <= limit,
            }
        )
    return out


def accuracy(rows: list[dict[str, Any]], arm: str, split: str) -> dict[str, Any]:
    values = [r["task_accuracy"] for r in by_seed(rows, arm, split).values()]
    return {"mean": mean(values), "min": min(values), "max": max(values)}


def evaluate(recorded: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_rows, new_rows = recorded["runs"], new["runs"]
    v_s = paired(new_rows, "V", new_rows, "S")
    v_l = paired(new_rows, "V", old_rows, "L")
    s_rerun = paired(new_rows, "S", old_rows, "S")
    old_s = accuracy(old_rows, "S", LONG)
    new_s = accuracy(new_rows, "S", LONG)
    s_matches = old_s["min"] <= new_s["mean"] <= old_s["max"]
    long = {p["split"]: p for p in v_s}[LONG]
    long_l = {p["split"]: p for p in v_l}[LONG]
    return {
        "arms": {
            arm: {split: accuracy(new_rows, arm, split) for split in SPLITS}
            for arm in ("V", "S")
        },
        "recorded": {
            arm: {split: accuracy(old_rows, arm, split) for split in SPLITS}
            for arm in ("L", "S")
        },
        "v_minus_s_rerun": v_s,
        "v_minus_l_recorded": v_l,
        "s_rerun_minus_s_recorded": s_rerun,
        "s_rerun_within_recorded_range_at_64": s_matches,
        "1_v_beats_s_at_64": {
            "mean_difference": long["mean"],
            "bound": long["bound"],
            "met": long["above_bound"],
        },
        "2_v_within_bound_of_l_at_64": {
            "evaluated": s_matches,
            "mean_difference": long_l["mean"],
            "bound": long_l["bound"],
            "met": long_l["within_bound"] if s_matches else None,
        },
    }


def main(argv: list[str]) -> int:
    recorded = json.loads(Path(argv[0]).read_text())
    new = json.loads(Path(argv[1]).read_text())
    result = evaluate(recorded, new)
    print(json.dumps(result, indent=2))
    if len(argv) > 2:
        Path(argv[2]).write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
