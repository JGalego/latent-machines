# AGENTS.md

> **TL;DR:** Follow CONTRIBUTING.md for standards; use this file for repository status and ownership boundaries.

Guidance for coding agents working in this repository. [CONTRIBUTING.md](CONTRIBUTING.md) is the authoritative standard; this file covers orientation and the cross-file rules that are easy to miss.

## Current state

> **TL;DR:** What exists now, and the commands to check it.

The uv project, Ruff, pytest, pre-commit hooks, and CI are configured. E00, E01, and E02 have runners and first results in `results/`. No other experiment exists yet. The layout in EXPERIMENTS.md is a target, not existing files. Do not report a check as passing without running it.

```bash
uv sync --locked --extra torch-cpu                       # restore the locked environment (Python 3.13+)
uv run --locked pre-commit install                       # install git hooks once per clone
uv run --locked pre-commit run --all-files               # Ruff lint and format, hygiene checks
uv run --locked pytest                                   # bounded CPU test suite
uv run --locked pytest tests/path/test_x.py::test_name   # one test
uv run --locked python -m latent_machines.experiments.e00   # E00 default run, record under runs/
uv run --locked python -m latent_machines.experiments.e01   # E01 default run, record under runs/
uv run --locked python -m latent_machines.experiments.e02 --model tiny-random   # E02 smoke run; the default downloads SmolLM2-135M
```

CI (`.github/workflows/ci.yml`) runs the same locked sync, hooks, and tests on pull requests and pushes to `main`. The Ruff hooks run from the locked environment, so `uv.lock` is the only place its version is pinned. Add dependencies with `uv add` or `uv add --dev` and commit `uv.lock` with them. The `torch-cpu` extra installs CPU-only PyTorch for tensor backends; without it, a plain `uv sync` removes PyTorch and backend tests are skipped, which `-ra` reports.

## Documents

> **TL;DR:** Keep source reviews, experiment plans, and result summaries in their designated files, with cross-links intact.

| File | Role |
|---|---|
| README.md | Pitch and scope |
| RESEARCH.md | Sources, attributed results, evidence boundaries, candidate layer families, open questions |
| EXPERIMENTS.md | Experiment program E00-E08, target layout, execution and evidence contracts, research coverage, scope gates |
| CONTRIBUTING.md | Code, tooling, writing, commit, reproducibility, and verification standards |
| site/ | Public [project page](https://jgalego.github.io/latent-machines/), deployed by `.github/workflows/pages.yml`; results tables are generated from `results/` manifests |

Cross-file rules:

- Follow the [writing guidelines](CONTRIBUTING.md#14-write-for-the-reader) for documentation, website copy, reports, and replies. Remove filler and gratuitous counts without dropping technical limits or evidence qualifiers.
- Every claim, proposal, or open question in RESEARCH.md needs a row in the [Research coverage](EXPERIMENTS.md#research-coverage) table, mapping it to an experiment, a scope gate, or context only. Add the row in the same change.
- Headings are link targets. CONTRIBUTING.md and RESEARCH.md link into EXPERIMENTS.md by anchor (`#research-coverage`, `#run-records-and-results`, `#first-implementation-slice`, `#repository-organization`, `#gates-for-expanding-scope`). Grep for an anchor before renaming its heading.
- `site/index.html` is a template that `site/build.py` renders. Results tables, table, figure, and equation numbers, citations, the abstract's result sentence, the conformance summary, and the experiment-status caption are generated from `results/`, `site/references.json`, and `{{...}}` markers; never type these numbers. Add a source to `references.json` and cite it with `{{cite:key}}`. A new record format needs a renderer in `build.py`; until then it appears as links.
- When a result changes the project's status, update the prose the build cannot generate, in the same change: the site's abstract and contributions, its setup (Section 4.1), in-progress section (Section 5), limitations, and conclusion, plus README.md and the EXPERIMENTS.md status line.
- "Gate" means either a neural gate (gate endpoints, channel gates) or one of the numbered [scope gates](EXPERIMENTS.md#gates-for-expanding-scope). Link scope gates explicitly.

## Architecture

> **TL;DR:** Keep machine semantics independent of the backend, controller interface, and task.

Code ownership follows these boundaries:

- **Machine** (`machines/`): specified transitions `(state, action) -> (next_state, event, work)` with an independent integer oracle and no dependency on a model family or tensor backend.
- **Backend** (`backends/`): native, tensor, and optimized implementations of a machine. Owns lowering, kernels, and decoding to canonical state for conformance; decoded state must not leak to a controller.
- **Interface** (`interfaces/`): what a controller observes and how it acts. Owns neither task answers nor executor arithmetic.
- **Task** (`tasks/`): generators, ground truth, and splits. Cannot change machine semantics.

`experiments/` composes these and writes run records; it must not reimplement machines or metrics. Not every combination is valid: a backend declares the semantics it supports and rejects the rest explicitly.

Build in the order of the [first implementation slice](EXPERIMENTS.md#first-implementation-slice): integer memory oracle, eager delta-rule tensor backend validated at every step, E00, E01 machines, then a learned controller for E02. Do not start with a general compiler, plugin registry, training framework, or full ISA; the scope gates decide when those are justified.

## Evidence language

> **TL;DR:** Distinguish proposals, upstream reports, local conformance tests, and proofs.

The documents are precise about evidence class, and edits must preserve it:

- "Exact" means matching a specified machine within declared representation, precision, and resource limits, not merely deterministic output.
- Distinguish hypotheses, theorems under stated assumptions, upstream reports (attributed, not validated here), local reproductions, and independently checked results. Never turn a proposal into a claimed result.
- Passing a finite test suite is conformance evidence, not proof. Negative results are results.
- Cite primary sources with links. Absence from the literature review is not evidence of novelty.

## Working rules

> **TL;DR:** Default to small CPU runs; follow the approval limits and documentation requirements below.

- Keep runs small and CPU-only by default. Model downloads, GPU sweeps, training, paid compute, and network-dependent tests need explicit approval, except Hugging Face Jobs within the limit below.
- Agents may launch [Hugging Face Jobs](CONTRIBUTING.md#12-run-intensive-work-on-hugging-face-jobs), CPU or GPU, for work too slow for local CPU when the worst-case cost of all jobs for one task (hourly price × timeout, summed) is at most $5. State the flavor, timeout, and worst-case cost when launching, and cancel jobs that are no longer needed. Anything above the limit needs explicit approval.
- Follow the [commit message rules](CONTRIBUTING.md#6-write-informative-commit-messages): at most one experiment per commit, results committed separately after the code that produced them, and a `Co-Authored-By:` trailer for substantial agent-written changes.
- When explaining a new concept, machine, or experiment, in documents, pull requests, or replies, include a diagram as code per [CONTRIBUTING.md](CONTRIBUTING.md#11-explain-visually).
- Start every Markdown page, section, and subsection, except the main README.md, with a one-line `> **TL;DR:**` per [CONTRIBUTING.md](CONTRIBUTING.md#13-lead-every-page-and-section-with-a-tldr).
- Raw runs go under the ignored `runs/` directory; curated summaries go under `results/` per the [run-record contract](EXPERIMENTS.md#run-records-and-results).
