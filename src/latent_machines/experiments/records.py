"""Run directories and provenance for experiment run records.

Implements the source and environment parts of the run-record contract in
EXPERIMENTS.md. Each run gets a fresh directory under ``runs/``.
"""

import hashlib
import json
import os
import platform
import secrets
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
HF_JOB_VARIABLES = ("JOB_ID", "ACCELERATOR", "CPU_CORES", "MEMORY")
LAUNCH_PREFIX = "LM_RUN_"  # set by whoever launches a remote run, e.g. LM_RUN_FLAVOR


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, check=True
    ).stdout


def repo_root(start: Path) -> Path | None:
    try:
        return Path(_git(start, "rev-parse", "--show-toplevel").decode().strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def source(root: Path | None) -> tuple[dict[str, Any], bytes]:
    """The checkout's revision and dirty-tree fingerprint, plus its patch.

    The fingerprint hashes the diff of tracked files against ``HEAD`` and the
    names and contents of untracked, unignored files. The patch covers the
    tracked changes only; untracked files are listed by name.
    """
    if root is None:
        return {"revision": None, "error": "not a git checkout"}, b""
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    patch = _git(root, "diff", "HEAD", "--binary")
    listed = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    untracked = sorted(path for path in listed.decode().split("\0") if path)
    digest = hashlib.sha256(patch)
    for path in untracked:
        digest.update(path.encode() + b"\0" + (root / path).read_bytes())
    dirty = bool(patch or untracked)
    info = {
        "revision": revision,
        "dirty": dirty,
        "dirty_fingerprint": digest.hexdigest() if dirty else None,
        "untracked": untracked,
    }
    return info, patch


def environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpu_quota": _cpu_quota(),
        "hf_job": _hf_job(),
        "launch": {
            name.removeprefix(LAUNCH_PREFIX).lower(): value
            for name, value in sorted(os.environ.items())
            if name.startswith(LAUNCH_PREFIX)
        },
    }


def _cpu_quota() -> float | None:
    """CPUs allowed by the cgroup v2 quota; containers often report host CPUs."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
    except (OSError, ValueError):
        return None
    return None if quota == "max" else int(quota) / int(period)


def _hf_job() -> dict[str, str] | None:
    """Variables Hugging Face Jobs sets inside the container, if running in one."""
    if "JOB_ID" not in os.environ:
        return None
    return {
        name.lower(): os.environ[name]
        for name in HF_JOB_VARIABLES
        if name in os.environ
    }


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def new_run_dir(base: Path, experiment: str, revision: str | None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    short = (revision or "norev")[:7]
    path = base / f"{experiment}-{stamp}-{short}-{secrets.token_hex(2)}"
    path.mkdir(parents=True)  # fails rather than reuse an existing directory
    return path


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
