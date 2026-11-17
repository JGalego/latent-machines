"""Run-record provenance: source state and runtime environment."""

import os
from pathlib import Path

import pytest

from latent_machines.experiments import records


def test_environment_records_hf_job_and_launch_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOB_ID", "abc123")
    monkeypatch.setenv("CPU_CORES", "8")
    monkeypatch.setenv("LM_RUN_FLAVOR", "cpu-upgrade")
    monkeypatch.setenv("LM_RUN_TIMEOUT", "3h")
    env = records.environment()
    assert env["hf_job"] == {"job_id": "abc123", "cpu_cores": "8"}
    assert env["launch"] == {"flavor": "cpu-upgrade", "timeout": "3h"}


def test_environment_outside_a_job(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JOB_ID", raising=False)
    for name in [n for n in os.environ if n.startswith("LM_RUN_")]:
        monkeypatch.delenv(name)
    env = records.environment()
    assert env["hf_job"] is None
    assert env["launch"] == {}
    assert env["cpu_quota"] is None or env["cpu_quota"] > 0


def test_source_reports_a_revision_in_this_checkout() -> None:
    root = records.repo_root(Path(__file__).resolve().parent)
    info, _ = records.source(root)
    assert info["revision"] and len(info["revision"]) == 40
    assert isinstance(info["dirty"], bool)
