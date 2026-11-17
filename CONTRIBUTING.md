# Contributing

> **TL;DR:** How to make changes that others can trust, rerun, and build on.

These guidelines apply to code, experiments, results, documentation, and automated contributions.

## 1. Clean, minimal code

> **TL;DR:** Write the simplest code that does the job, nothing speculative.

- Write the simplest implementation that satisfies the stated contract. Prefer small functions, explicit data flow, and descriptive names.
- Keep machine semantics separate from backend optimizations, controller interfaces, and task definitions.
- Use type annotations for interfaces and state/action representations. Make bounds, units, dtypes, and ownership explicit where they affect correctness.
- Reuse working patterns, but do not build a framework for hypothetical future experiments. Extract shared abstractions only after concrete implementations justify them.
- Comment non-obvious invariants and numerical assumptions, not what the code already says. Delete obsolete code rather than keeping alternate implementations commented out.

## 2. Minimal dependencies

> **TL;DR:** Add a library only when you truly need it.

- Prefer the standard library when it solves the problem clearly. Add a dependency only for a concrete requirement, and explain why existing tools are insufficient.
- Separate runtime, development, and optional accelerator/model dependencies. A small CPU experiment must not require the full research stack.
- Default to Ruff, pytest, and pre-commit for Python development; add overlapping tools only for a demonstrated gap.
- Do not require a hosted service for tests, run tracking, or basic reproduction.

## 3. Use uv

> **TL;DR:** uv manages Python and packages, so everyone gets the same environment.

- Manage Python environments, dependencies, and commands with uv. Declare dependencies in `pyproject.toml` and commit `uv.lock` with dependency changes.
- Use `uv add` and `uv remove` rather than manually editing the lockfile. Declare the supported Python version; record the actual interpreter used for results.
- Use `uv sync --locked` to restore the declared environment and `uv run --locked ...` for checks and experiments. CI must fail on an inconsistent lockfile rather than updating it.
- Do not rely on globally installed packages, ad hoc pip installs, or an undocumented activated environment.

## 4. Use pre-commit

> **TL;DR:** Automatic checks run before every commit.

- Add a version-pinned `.pre-commit-config.yaml` with the first executable implementation. Use it for formatting, linting, and cheap repository hygiene checks.
- Document hook installation and the full local check command when the configuration exists. Run the same checks in CI; local hooks are convenience, not the only enforcement.
- Keep commit hooks fast, deterministic, and CPU-only. No model downloads, training, network-dependent tests, or long benchmarks in hooks.
- Fix failures rather than routinely bypassing hooks. Any exceptional bypass needs an explanation and equivalent validation before integration.

## 5. Practice continuous integration

> **TL;DR:** Small, frequent, checked changes beat big surprises.

- Commit early and often, in small, coherent, validated increments. Integrate frequently rather than accumulating a large unreviewable branch.
- A commit should be understandable and reproducible on its own. Separate refactoring, behavior changes, and scientific conclusions where practical.
- Add CI with the first executable implementation: locked environment restoration, format/lint checks, and bounded CPU tests on every pull request and integration push.
- Keep GPU sweeps, training, and long numerical tests explicit and separately budgeted. Passing the fast CI suite does not validate an unrun accelerator path.
- Do not commit unrelated staged work, secrets, generated checkpoints, or raw run directories. Fix failures before merging; do not hide them by weakening assertions.

## 6. Write informative commit messages

> **TL;DR:** Each commit says what changed in one clear line, and why in the body.

- Write an imperative subject of at most 72 characters, ideally about 50, without a trailing period: `Add integer memory oracle`, not `Added oracle` or `wip`.
- Optionally prefix the subject with an experiment ID or a top-level directory from the [repository layout](EXPERIMENTS.md#repository-organization), or `docs` for the root documents: `E01: add bounded stack parser`, `backends: validate gate endpoints`. Do not invent a separate type taxonomy.
- After a blank line, explain why the change is needed and what it affects, wrapped at 72 characters. The diff already shows what changed.
- Keep each commit to one logical change and at most one experiment. Shared code used by several experiments takes its directory prefix, not an experiment ID. An experiment normally spans many commits; integrate them as they pass rather than holding the experiment on a long-lived branch.
- Commit code before running it for reported results, then record each result in its own commit that names the source revision the run used. A result committed together with its code was produced from uncommitted code, so its recorded revision does not contain that code.
- State results at their evidence strength: name the experiment, scope, and outcome, and put the run manifest path in the body. Write `E00: record zero GDN mismatches on bounded suite`, not `Prove delta memory is exact`.
- Commit negative and failed results the same way, without waiting for the experiment to succeed: `E00: record RWKV-7 idle-retention failure at 10^3 steps`.
- Credit substantial AI-agent or other automated contributions with a `Co-Authored-By:` trailer.
- No hook enforces these rules yet. Add a pre-commit `commit-msg` check only if the convention drifts.

## 7. Reproducibility is part of the deliverable

> **TL;DR:** Anyone should be able to rerun a result from a clean checkout.

- Provide a command that runs from a clean checkout, with declared inputs, budgets, and expected outputs. No undocumented notebook state or manual file edits.
- Record the source revision, resolved configuration, seeds, input/model hashes, environment, kernels, and relevant dtypes. Follow the [run-record contract](EXPERIMENTS.md#run-records-and-results).
- Seeds alone do not ensure reproducibility. State which outcomes are bit-exact, tolerance-based, statistically replicated, or device-dependent.
- Curated results must identify the exact source used. Prefer a committed clean revision; if a dirty tree is used, retain a retrievable patch and fingerprint.
- Keep small reports and manifests in Git; keep large artifacts outside it with stable references and integrity hashes. Preserve failed runs and first-mismatch evidence.

## 8. Verified results are mandatory

> **TL;DR:** No result counts until it is checked against an independent reference.

- Do not present an experiment as successful until the stated acceptance criteria pass on the actual execution path being reported.
- Test state transitions, events, and outputs against an independent reference. Final-answer agreement alone is insufficient for an exact-machine claim.
- Check correctness before timing optimizations. Report validation commands, scope, limitations, and failures alongside results.
- Label hypotheses, theoretical results, upstream reports, local reproductions, and independently checked results distinctly. Never turn a proposed experiment into a claimed result.
- Passing finite tests is not a proof of universal exactness. When verification is incomplete or unavailable, say so and withhold the stronger claim.

## 9. Keep comparisons honest

> **TL;DR:** Compare like with like, and report the losses too.

- Use matched information, data, capacity, and work budgets. Separate interface benefits from executor, specialization, or implementation benefits.
- Include a credible conventional baseline. Do not handicap textual tools with unnecessary subprocess or network overhead.
- Separate initialization/compilation, steady-state execution, and end-to-end costs. Synchronize asynchronous work before timing.
- Report uncertainty where relevant, negative results, invalid actions, timeouts, and exhausted budgets. Do not select only favorable seeds or silently discard failed cases.
- Distinguish exploration from held-out evaluation. Record tuning effort and do not repeatedly tune on the test set.

## 10. Make failures and limits explicit

> **TL;DR:** Say exactly what happens at the edges, and never hide a failure.

- Specify overflow, capacity, invalid actions, ties, stopping, and fault behavior. Never silently clamp, repair, or substitute a successful-looking output.
- Keep a simple reference implementation independent of optimized code. Add focused regression cases for bugs and counterexamples.
- Use small, local, bounded defaults. Downloads, paid compute, long jobs, and external services must require explicit opt-in.
- Cite primary sources and respect code, model, and dataset licenses. Do not copy upstream code without checking its reuse terms.

## 11. Explain visually

> **TL;DR:** Show structure with diagrams written as text, so they can be reviewed like code.

- Accompany each new concept, machine, experiment, or data flow with a diagram when structure carries the meaning: state transitions, event order, ownership between axes, experiment arms and controls, and dependencies.
- Write diagrams as code: [Mermaid](https://mermaid.js.org/) blocks in Markdown, which GitHub renders in files, issues, and pull requests. Place each next to the prose it illustrates. Do not commit exported or binary diagrams that cannot be reviewed in a diff.
- Show one idea per diagram, use the same names as the code and documents, and keep it small enough to read without zooming.
- The text stays authoritative. A diagram illustrates a specification but does not replace it; update both in the same change.
- Generate result figures from run records with a committed script, never by hand, and name the script and run in the report that uses them.

## 12. Run intensive work on Hugging Face Jobs

> **TL;DR:** Heavy runs go to Hugging Face Jobs, pinned to a pushed commit and a set timeout.

- Run tests and sweeps that are too slow for local CPU, or that need a GPU, as [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs). Keep the default test suite, hooks, and CI local and CPU-only.
- Run a pushed commit: clone the public repository at that revision inside the job and restore the environment with `uv sync --locked`, so the run record names source anyone can fetch. Do not ship uncommitted code to a job.
- Choose the smallest flavor that fits (`hf jobs hardware` lists flavors and prices) and always set `--timeout`. Label each job with its experiment and revision, for example `--label experiment=E00 --label revision=<sha>`.
- Pass credentials with `--secrets`, never in the command, plain environment variables, or logs.
- The job's filesystem is discarded when it ends. Upload the run directory, for example to a Hub dataset, and record the job ID, flavor, image, timeout, wall time, and artifact location in the run record. Runners record the job's own variables (`JOB_ID`, `CPU_CORES`, `MEMORY`, `ACCELERATOR`) and the cgroup CPU quota automatically; pass what the container cannot see as `-e LM_RUN_FLAVOR=… -e LM_RUN_IMAGE=… -e LM_RUN_TIMEOUT=…`. Keep the logs of failed, timed-out, and cancelled jobs.
- A job validates only the hardware, kernels, and dtypes it ran on; label results by flavor. GPU backends need a CUDA build of PyTorch, added as a separate extra when the first GPU job requires it.

## 13. Lead every page and section with a TL;DR

> **TL;DR:** One memorable line under every heading tells a newcomer what it is for.

- Start every Markdown page, section, and subsection with a one-line TL;DR directly under its heading, written as `> **TL;DR:** …`.
- Write it for a newcomer, including non-specialists: plain words, intuitive, and memorable. Use one sentence, ideally under 20 words, and no jargon the section does not explain.
- A TL;DR summarizes; it adds no claims. Keep the evidence class: a proposal stays a proposal, and a finite test stays a finite test.
- Update a section's TL;DR in the same change as the section. Pointer files such as CLAUDE.md are exempt.

## Current tooling status

> **TL;DR:** What is set up today, and the commands to use it.

The uv project, Ruff, pytest, pre-commit hooks, and CI are configured. Python 3.13 or later is required; `.python-version` selects 3.13 for development and CI.

```bash
uv sync --locked --extra torch-cpu           # restore the locked environment
uv run --locked pre-commit install           # install git hooks once per clone
uv run --locked pre-commit run --all-files   # Ruff lint and format, hygiene checks
uv run --locked pytest                       # bounded CPU tests
```

The `torch-cpu` extra adds CPU-only PyTorch for tensor backends; the machine oracles need no runtime dependencies. CI runs the same checks with the locked environment on every pull request and push to `main`. The E00 runner writes run records under the ignored `runs/` directory; see [experiments](src/latent_machines/experiments/README.md).

Start with the [first implementation slice](EXPERIMENTS.md#first-implementation-slice), not a speculative scaffolding exercise.
