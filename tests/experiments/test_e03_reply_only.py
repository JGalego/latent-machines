"""The amendment's prediction logic on synthetic run records."""

from latent_machines.experiments import e03_reply_only as amendment

QUESTIONS = {"test_in_distribution": 471, "test_length_32": 937, "test_length_64": 1920}


def rows(arm: str, accuracy: float) -> list[dict]:
    return [
        {
            "arm": arm,
            "split": s,
            "seed": seed,
            "questions": q,
            "task_accuracy": accuracy,
        }
        for s, q in QUESTIONS.items()
        for seed in (0, 1, 2)
    ]


def test_bound_matches_the_recorded_table() -> None:
    assert round(amendment.bound(1920), 3) == 0.032
    assert round(amendment.bound(471), 3) == 0.065


def test_predictions_met_when_v_leads_s_and_matches_l() -> None:
    recorded = {"runs": rows("L", 0.62) + rows("S", 0.57)}
    new = {"runs": rows("V", 0.61) + rows("S", 0.57)}
    result = amendment.evaluate(recorded, new)
    assert result["1_v_beats_s_at_64"]["met"]
    assert result["2_v_within_bound_of_l_at_64"]["met"]
    assert result["s_rerun_within_recorded_range_at_64"]


def test_prediction_two_is_not_evaluated_when_s_does_not_reproduce() -> None:
    recorded = {"runs": rows("L", 0.62) + rows("S", 0.57)}
    new = {"runs": rows("V", 0.61) + rows("S", 0.40)}
    result = amendment.evaluate(recorded, new)
    assert not result["s_rerun_within_recorded_range_at_64"]
    assert result["2_v_within_bound_of_l_at_64"]["met"] is None
