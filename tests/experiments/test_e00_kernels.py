"""The E00 kernels runner: counted mismatches, errors, and predictions."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from latent_machines.experiments import e00_kernels  # noqa: E402


def run(out: Path, *args: str) -> tuple[int, Path]:
    code = e00_kernels.main(["--out", str(out), "--horizon", "300", *args])
    (run_dir,) = out.iterdir()
    return code, run_dir


def test_small_run_records_counts_and_predictions(tmp_path: Path) -> None:
    code, run_dir = run(
        tmp_path,
        "--kernels", "recurrent,chunked-4,chunked-16",
        "--dtypes", "float32,bfloat16",
        "--capacities", "8,40",
        "--workloads", "overwrite,random_mix",
    )  # fmt: skip
    assert code == 0
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["status"] == "completed"
    assert metrics["totals"]["variants"] == 3 * 2 * 2 * 2
    assert metrics["totals"]["mismatches"] == 0
    kernels = {m["kernel"] for m in metrics["variants"]}
    assert kernels == {"recurrent", "chunked-4", "chunked-16"}
    chunk = next(m for m in metrics["variants"] if m["kernel"] == "chunked-16")
    assert chunk["event_mismatches"] == 0 and chunk["state_mismatches"] == 0
    assert chunk["classification"] == "exact"
    predictions = metrics["predictions"]
    assert predictions["1_chunked_exact_in_float64_and_float32"]["met"] is True
    assert predictions["3_fused_bfloat16_has_a_mismatch"]["met"] is None
    assert json.loads((run_dir / "run.json").read_text())["environment"]["fla"]


def test_mismatches_are_counted_not_stopped_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = e00_kernels.chunked.chunk_update

    def broken(state, keys, values, beta, queries):
        outputs, state = real(state, keys, values, beta, queries)
        return outputs, state.flip(0)

    monkeypatch.setattr(e00_kernels.chunked, "chunk_update", broken)
    _, run_dir = run(
        tmp_path,
        "--kernels", "chunked-16",
        "--dtypes", "float64",
        "--capacities", "8",
        "--workloads", "overwrite",
    )  # fmt: skip
    metrics = json.loads((run_dir / "metrics.json").read_text())
    (variant,) = metrics["variants"]
    assert variant["state_mismatches"] > 1
    assert variant["classification"] == "mismatch"
    assert (run_dir / "failure.json").exists()


def test_a_kernel_that_cannot_run_is_recorded_as_an_error(tmp_path: Path) -> None:
    _, run_dir = run(
        tmp_path,
        "--kernels", "fla-chunk",
        "--dtypes", "float32",
        "--capacities", "8",
        "--workloads", "overwrite",
    )  # fmt: skip
    metrics = json.loads((run_dir / "metrics.json").read_text())
    (variant,) = metrics["variants"]
    assert variant["classification"] == "error"
    assert metrics["status"] == "completed with errors"
