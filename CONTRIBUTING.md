# Contributing

These guidelines apply to code, experiments, results, documentation, and automated contributions.

## 1. Clean, minimal code

- Write the simplest implementation that satisfies the stated contract. Prefer small functions, explicit data flow, and descriptive names.
- Keep machine semantics separate from backend optimizations, controller interfaces, and task definitions.
- Use type annotations for interfaces and state/action representations. Make bounds, units, dtypes, and ownership explicit where they affect correctness.
- Reuse working patterns, but do not build a framework for hypothetical future experiments. Extract shared abstractions only after concrete implementations justify them.
- Comment non-obvious invariants and numerical assumptions, not what the code already says. Delete obsolete code rather than keeping alternate implementations commented out.

## 2. Minimal dependencies

- Prefer the standard library when it solves the problem clearly. Add a dependency only for a concrete requirement, and explain why existing tools are insufficient.
- Separate runtime, development, and optional accelerator/model dependencies. A small CPU experiment must not require the full research stack.
- Default to Ruff, pytest, and pre-commit for Python development; add overlapping tools only for a demonstrated gap.
- Do not require a hosted service for tests, run tracking, or basic reproduction.

## 3. Use uv

- Manage Python environments, dependencies, and commands with uv. Declare dependencies in `pyproject.toml` and commit `uv.lock` with dependency changes.
- Use `uv add` and `uv remove` rather than manually editing the lockfile. Declare the supported Python version; record the actual interpreter used for results.
- Use `uv sync --locked` to restore the declared environment and `uv run --locked ...` for checks and experiments. CI must fail on an inconsistent lockfile rather than updating it.
- Do not rely on globally installed packages, ad hoc pip installs, or an undocumented activated environment.

## 4. Use pre-commit

- Add a version-pinned `.pre-commit-config.yaml` with the first executable implementation. Use it for formatting, linting, and cheap repository hygiene checks.
- Document hook installation and the full local check command when the configuration exists. Run the same checks in CI; local hooks are convenience, not the only enforcement.
- Keep commit hooks fast, deterministic, and CPU-only. No model downloads, training, network-dependent tests, or long benchmarks in hooks.
- Fix failures rather than routinely bypassing hooks. Any exceptional bypass needs an explanation and equivalent validation before integration.

## 5. Practice continuous integration

- Commit early and often, in small, coherent, validated increments. Integrate frequently rather than accumulating a large unreviewable branch.
- A commit should be understandable and reproducible on its own. Separate refactoring, behavior changes, and scientific conclusions where practical.
- Add CI with the first executable implementation: locked environment restoration, format/lint checks, and bounded CPU tests on every pull request and integration push.
- Keep GPU sweeps, training, and long numerical tests explicit and separately budgeted. Passing the fast CI suite does not validate an unrun accelerator path.
- Do not commit unrelated staged work, secrets, generated checkpoints, or raw run directories. Fix failures before merging; do not hide them by weakening assertions.

## 6. Write informative commit messages

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

- Provide a command that runs from a clean checkout, with declared inputs, budgets, and expected outputs. No undocumented notebook state or manual file edits.
- Record the source revision, resolved configuration, seeds, input/model hashes, environment, kernels, and relevant dtypes. Follow the [run-record contract](EXPERIMENTS.md#run-records-and-results).
- Seeds alone do not ensure reproducibility. State which outcomes are bit-exact, tolerance-based, statistically replicated, or device-dependent.
- Curated results must identify the exact source used. Prefer a committed clean revision; if a dirty tree is used, retain a retrievable patch and fingerprint.
- Keep small reports and manifests in Git; keep large artifacts outside it with stable references and integrity hashes. Preserve failed runs and first-mismatch evidence.

## 8. Verified results are mandatory

- Do not present an experiment as successful until the stated acceptance criteria pass on the actual execution path being reported.
- Test state transitions, events, and outputs against an independent reference. Final-answer agreement alone is insufficient for an exact-machine claim.
- Check correctness before timing optimizations. Report validation commands, scope, limitations, and failures alongside results.
- Label hypotheses, theoretical results, upstream reports, local reproductions, and independently checked results distinctly. Never turn a proposed experiment into a claimed result.
- Passing finite tests is not a proof of universal exactness. When verification is incomplete or unavailable, say so and withhold the stronger claim.

## 9. Keep comparisons honest

- Use matched information, data, capacity, and work budgets. Separate interface benefits from executor, specialization, or implementation benefits.
- Include a credible conventional baseline. Do not handicap textual tools with unnecessary subprocess or network overhead.
- Separate initialization/compilation, steady-state execution, and end-to-end costs. Synchronize asynchronous work before timing.
- Report uncertainty where relevant, negative results, invalid actions, timeouts, and exhausted budgets. Do not select only favorable seeds or silently discard failed cases.
- Distinguish exploration from held-out evaluation. Record tuning effort and do not repeatedly tune on the test set.

## 10. Make failures and limits explicit

- Specify overflow, capacity, invalid actions, ties, stopping, and fault behavior. Never silently clamp, repair, or substitute a successful-looking output.
- Keep a simple reference implementation independent of optimized code. Add focused regression cases for bugs and counterexamples.
- Use small, local, bounded defaults. Downloads, paid compute, long jobs, and external services must require explicit opt-in.
- Cite primary sources and respect code, model, and dataset licenses. Do not copy upstream code without checking its reuse terms.

## 11. Explain visually

- Accompany each new concept, machine, experiment, or data flow with a diagram when structure carries the meaning: state transitions, event order, ownership between axes, experiment arms and controls, and dependencies.
- Write diagrams as code: [Mermaid](https://mermaid.js.org/) blocks in Markdown, which GitHub renders in files, issues, and pull requests. Place each next to the prose it illustrates. Do not commit exported or binary diagrams that cannot be reviewed in a diff.
- Show one idea per diagram, use the same names as the code and documents, and keep it small enough to read without zooming.
- The text stays authoritative. A diagram illustrates a specification but does not replace it; update both in the same change.
- Generate result figures from run records with a committed script, never by hand, and name the script and run in the report that uses them.

## Current tooling status

The uv project, Ruff, pytest, pre-commit hooks, and CI are configured. Python 3.13 or later is required; `.python-version` selects 3.13 for development and CI.

```bash
uv sync --locked --extra torch-cpu           # restore the locked environment
uv run --locked pre-commit install           # install git hooks once per clone
uv run --locked pre-commit run --all-files   # Ruff lint and format, hygiene checks
uv run --locked pytest                       # bounded CPU tests
```

The `torch-cpu` extra adds CPU-only PyTorch for tensor backends; the machine oracles need no runtime dependencies. CI runs the same checks with the locked environment on every pull request and push to `main`. No experiment runner or result exists yet.

Start with the [first implementation slice](EXPERIMENTS.md#first-implementation-slice), not a speculative scaffolding exercise.
