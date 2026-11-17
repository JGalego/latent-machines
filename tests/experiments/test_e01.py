"""The E01 runner: run records for transducers and the stack."""

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends.lookup import LookupTransducer  # noqa: E402
from latent_machines.experiments import e01  # noqa: E402


def run(out: Path, *args: str) -> tuple[int, Path]:
    code = e01.main(["--out", str(out), "--horizon", "80", *args])
    (run_dir,) = out.iterdir()
    return code, run_dir


def load(run_dir: Path, name: str) -> Any:
    return json.loads((run_dir / name).read_text())


def test_small_run_passes_for_every_machine_and_workload(tmp_path: Path) -> None:
    code, run_dir = run(tmp_path, "--dtypes", "float32,float16")
    assert code == 0
    metrics = load(run_dir, "metrics.json")
    assert metrics["status"] == "passed"
    variants = metrics["variants"]
    assert len(variants) == 2 * (3 + 3)  # dtypes x (transducers + stack workloads)
    for m in variants:
        assert (m["classification"], m["snapshot_resume"]) == ("exact", "agrees")
    names = {m["variant"] for m in variants}
    assert "relu-lookup/float32/s3-tracker/symbols" in names
    assert "delta-stack/float16/stack/brackets" in names
    lookup = next(m for m in variants if m["machine"] == "s3-tracker")
    assert (lookup["hidden_width"], lookup["depth"]) == (12, 2)
    record = load(run_dir, "run.json")
    assert record["experiment"] == "E01"
    assert set(record["machines"]["transducer/v1"]) == set(e01.TRANSDUCERS)


def test_broken_circuit_writes_failure_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = LookupTransducer.__init__

    def miswired(self, machine, dtype=torch.float32, device="cpu") -> None:
        original(self, machine, dtype, device)
        self.w_next = self.w_next.flip(0)  # route every unit to the wrong state

    monkeypatch.setattr(LookupTransducer, "__init__", miswired)
    code, run_dir = run(tmp_path, "--machines", "s3-tracker", "--dtypes", "float32")
    assert code == 1
    (failure,) = load(run_dir, "failure.json")
    assert "--machines" in failure["reproduce"]
    assert failure["actual"]["decoded_state"] != failure["expected"]["state"]


def test_rejects_unknown_machine() -> None:
    with pytest.raises(SystemExit):
        e01.parse_args(["--machines", "turing"])
