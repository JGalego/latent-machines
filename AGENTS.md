# AGENTS.md

Guidance for coding agents working in this repository. [CONTRIBUTING.md](CONTRIBUTING.md) is the authoritative standard; this file covers orientation and the cross-file rules that are easy to miss.

## Current state

The uv project, Ruff, pytest, pre-commit hooks, and CI are configured; no experiment runner or result exists yet. The layout in EXPERIMENTS.md is a target, not existing files. Do not report a check as passing without running it.

```bash
uv sync --locked --extra torch-cpu                       # restore the locked environment (Python 3.13+)
uv run --locked pre-commit install                       # install git hooks once per clone
uv run --locked pre-commit run --all-files               # Ruff lint and format, hygiene checks
uv run --locked pytest                                   # bounded CPU test suite
uv run --locked pytest tests/path/test_x.py::test_name   # one test
```

CI (`.github/workflows/ci.yml`) runs the same locked sync, hooks, and tests on pull requests and pushes to `main`. The Ruff hooks run from the locked environment, so `uv.lock` is the only place its version is pinned. Add dependencies with `uv add` or `uv add --dev` and commit `uv.lock` with them. The `torch-cpu` extra installs CPU-only PyTorch for tensor backends; without it, a plain `uv sync` removes PyTorch and backend tests are skipped, which `-ra` reports.

## Documents

| File | Role |
|---|---|
| README.md | Pitch and scope |
| RESEARCH.md | Sources, attributed results, evidence boundaries, candidate layer families, open questions |
| EXPERIMENTS.md | Experiment program E00-E08, target layout, execution and evidence contracts, research coverage, scope gates |
| CONTRIBUTING.md | Code, tooling, commit, reproducibility, and verification standards |

Cross-file rules:

- Every claim, proposal, or open question in RESEARCH.md needs a row in the [Research coverage](EXPERIMENTS.md#research-coverage) table, mapping it to an experiment, a scope gate, or context only. Add the row in the same change.
- Headings are link targets. CONTRIBUTING.md and RESEARCH.md link into EXPERIMENTS.md by anchor (`#research-coverage`, `#run-records-and-results`, `#first-implementation-slice`, `#repository-organization`, `#gates-for-expanding-scope`). Grep for an anchor before renaming its heading.
- "Gate" means either a neural gate (gate endpoints, channel gates) or one of the numbered [scope gates](EXPERIMENTS.md#gates-for-expanding-scope). Link scope gates explicitly.

## Architecture

The program separates four axes, and the planned code mirrors them:

- **Machine** (`machines/`): specified transitions `(state, action) -> (next_state, event, work)` with an independent integer oracle and no dependency on a model family or tensor backend.
- **Backend** (`backends/`): native, tensor, and optimized implementations of a machine. Owns lowering, kernels, and decoding to canonical state for conformance; decoded state must not leak to a controller.
- **Interface** (`interfaces/`): what a controller observes and how it acts. Owns neither task answers nor executor arithmetic.
- **Task** (`tasks/`): generators, ground truth, and splits. Cannot change machine semantics.

`experiments/` composes these and writes run records; it must not reimplement machines or metrics. Not every combination is valid: a backend declares the semantics it supports and rejects the rest explicitly.

Build in the order of the [first implementation slice](EXPERIMENTS.md#first-implementation-slice): integer memory oracle, eager delta-rule tensor backend validated at every step, E00, E01 machines, then a learned controller for E02. Do not start with a general compiler, plugin registry, training framework, or full ISA; the scope gates decide when those are justified.

## Evidence language

The documents are precise about evidence class, and edits must preserve it:

- "Exact" means matching a specified machine within declared representation, precision, and resource limits, not merely deterministic output.
- Distinguish hypotheses, theorems under stated assumptions, upstream reports (attributed, not validated here), local reproductions, and independently checked results. Never turn a proposal into a claimed result.
- Passing a finite test suite is conformance evidence, not proof. Negative results are results.
- Cite primary sources with links. Absence from the literature review is not evidence of novelty.

## Working rules

- Keep runs small and CPU-only by default. Model downloads, GPU sweeps, training, paid compute, and network-dependent tests need explicit approval, except Hugging Face Jobs within the limit below.
- Agents may launch [Hugging Face Jobs](CONTRIBUTING.md#12-run-intensive-work-on-hugging-face-jobs), CPU or GPU, for work too slow for local CPU when the worst-case cost of all jobs for one task (hourly price × timeout, summed) is at most $5. State the flavor, timeout, and worst-case cost when launching, and cancel jobs that are no longer needed. Anything above the limit needs explicit approval.
- Follow the [commit message rules](CONTRIBUTING.md#6-write-informative-commit-messages): at most one experiment per commit, results committed separately after the code that produced them, and a `Co-Authored-By:` trailer for substantial agent-written changes.
- When explaining a new concept, machine, or experiment, in documents, pull requests, or replies, include a diagram as code per [CONTRIBUTING.md](CONTRIBUTING.md#11-explain-visually).
- Raw runs go under the ignored `runs/` directory; curated summaries go under `results/` per the [run-record contract](EXPERIMENTS.md#run-records-and-results).
