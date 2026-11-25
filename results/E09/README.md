# E09 results: state in the machine, not the context

> **TL;DR:** A controller that read one statement at a time, with the machine holding all state, answered every question correctly at every length up to 256 statements while full-context controllers fell to 0.11-0.13; but the controller's job reduces to a 128-string lookup, so this shows that long context is what hurts, not that the language model does the work.

In plain terms: E02 found that every controller got worse on programs longer than its training programs, while the machine stayed exact. E09 asked whether that decline comes from reading a long context or from ordinary per-statement mistakes adding up. The answer is the first. When the same heads read each statement alone (S) and let the machine remember, they made no action errors on 2-8, 32, 64, or 256 statements, and the machine answered every question correctly. The two full-context controllers, run again on the same programs, fell from about 0.87-0.89 to 0.11-0.13 at 256 statements. Two of the three pre-registered predictions held, one of them vacuously. The third, a ceiling of 0.15 for a controller with no memory, failed: that controller scored 0.50 on short programs.

## Question and design

> **TL;DR:** The same task, heads, and machine, with the controller's context changed from the whole program to one statement.

[E09](../../EXPERIMENTS.md#e09---state-in-the-machine-not-the-context) uses [E02's](../E02/README.md) task: programs of `Set c to 7.` and `What is c?` over 8 records of 4 bits on [`memory/v1`](../../src/latent_machines/machines/README.md#memoryv1), a frozen [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), and E02's trained heads. Only the context each forward pass reads differs:

```mermaid
flowchart LR
    subgraph C["Arm C (full context)"]
        direction LR
        P1["preamble + statements 1..k"] --> LM1["frozen SmolLM2"] --> H1["heads"] --> M1[("memory/v1")]
    end
    subgraph S["Arm S (one statement)"]
        direction LR
        P2["preamble + statement k"] --> LM2["frozen SmolLM2"] --> H2["heads"] --> M2[("memory/v1<br/>holds all state")]
    end
```

| Arm | Context the model reads | Machine | Trainable parameters |
|---|---|---|---|
| B | Whole program, with text requests and replies (E02 B) | Integer oracle | 308,522 |
| C | Whole program (E02 C) | Integer oracle, typed interface | 317,290 |
| S | Preamble plus the current statement, a fresh window each time | Integer oracle, typed interface | 317,290 |
| S+delta | S's trained heads | Delta-rule memory ([E00](../E00/README.md)) with a shadow oracle | shared with S |
| W | Preamble plus the current statement | None; the answer head sees only the question | 152,976 |

B, C, and the new 256-statement split were run in the same job, so every arm saw the same test programs. A statement's feature depends only on its own text and the preamble, so it is identical wherever the statement sits in a program.

## Evidence class

> **TL;DR:** One GPU run, three training seeds, pre-registered predictions evaluated as written; the setting is one frozen small model and one task.

All arms share training programs and test programs, so comparisons are paired by program. Evaluation is free-running. The three predictions and the compounding baseline were fixed in [EXPERIMENTS.md](../../EXPERIMENTS.md#e09---state-in-the-machine-not-the-context) in commit [`d44e15b`](https://github.com/JGalego/latent-machines/commit/d44e15b) before the code existed. This is a single run by this project, not independently reproduced. Conformance of the delta-rule executor is finite-test evidence, not proof.

## Setup

> **TL;DR:** Committed revision `c7d252a`, one `l4x1` job, E02's training recipe, four test splits up to 256 statements.

| Item | Value |
|---|---|
| Source | [`c7d252a`](https://github.com/JGalego/latent-machines/commit/c7d252ac4ad26b7ffb8a407fd0a113e79bd28ea8), clean tree |
| Model | SmolLM2-135M, revision `93efa2f`, frozen |
| Hardware | HF Jobs `l4x1`, features on the GPU; heads and executors on the same job |
| Training | 2,000 programs of 2-8 statements, 30 epochs, hidden size 256, seeds 0-2, data seed 0 |
| Test splits | 300 programs of 2-8 statements (769 questions); 100 of 32 (1,592); 100 of 64 (3,216); 100 of 256 (12,935) |

## Results

> **TL;DR:** S was exact at every length; B and C fell to about 0.11-0.13 at 256 statements; W matched the share of questions on unwritten records.

Task accuracy, mean over three seeds with the range in brackets:

| Arm | 2-8 statements | 32 statements | 64 statements | 256 statements |
|---|---|---|---|---|
| B | 0.871 [0.866, 0.875] | 0.389 [0.380, 0.394] | 0.224 [0.220, 0.228] | 0.111 [0.110, 0.113] |
| C | 0.893 [0.874, 0.908] | 0.461 [0.455, 0.470] | 0.286 [0.273, 0.293] | 0.134 [0.127, 0.144] |
| S | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| S+delta | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| W | 0.501 | 0.205 | 0.136 | 0.080 |

B and C reproduce E02 to three decimals on the three shared splits, which confirms that this job ran the same training and evaluation. W is identical for all three seeds. Action accuracy (statements whose action matched gold):

| Arm | 2-8 | 32 | 64 | 256 |
|---|---|---|---|---|
| B | 0.949 | 0.728 | 0.593 | 0.368 |
| C | 0.873 | 0.538 | 0.393 | 0.235 |
| S | 1.000 | 1.000 | 1.000 | 1.000 |

### Pre-registered predictions

Each prediction as written in [EXPERIMENTS.md](../../EXPERIMENTS.md#e09---state-in-the-machine-not-the-context), evaluated by the runner:

| # | Prediction | Result | Met |
|---|---|---|---|
| 1 | S − C exceeds the conservative bound at 64 and 256 statements | +0.714 (bound 0.025) at 64; +0.866 (bound 0.012) at 256; every seed above +0.70 and +0.85 | yes |
| 2 | S within 0.05 of the compounding baseline at every length | difference 0.000 at every length; baseline 1.000 | yes, vacuous |
| 3 | W at or below 0.15 on every split | 0.501 at 2-8, 0.205 at 32, 0.136 at 64, 0.080 at 256 | **no** |
| sanity | S's action accuracy constant across lengths (drift under 0.03) | drift 0.000 | ok |
| sanity | zero executor mismatches | 0 for S+delta on every split and seed | ok |

- **Prediction 2 is vacuous.** S made no action error on the 2-8 split, so ε = 0, the answer head given the gold observation scored a = 1.000, and the replay baseline is 1.000 at every length. The prediction held, but the experiment could not test whether S's errors would compound independently.
- **Prediction 3 failed as written.** Under the pre-registered rule, a failure means "the answer head recovers state from the question alone, so the task leaks and the comparison is not valid". The data point to a mis-set ceiling instead. W's accuracy matches, to four decimals on every split, the share of questions whose record was never written, where the correct answer is the initial value 0 (0.5007, 0.2048, 0.1359, 0.0802, computed from the test programs). W is identical across seeds and falls with length exactly as that share does. It never recovers a written value. The first two splits are short enough that half or a fifth of questions touch an unwritten record, which I did not account for when I set 0.15. That reading is mine, made after seeing the result; the pre-registered outcome stands as not met. S beats W by 0.50 at 2-8 statements, so the comparison still separates the machine from the question text.

### Post-hoc control: a random 2-layer model

> **TL;DR:** The same S pipeline on a random 29,000-parameter model with no pretraining reached 0.85-0.94, so most of S's result is the lookup, and this control makes prediction 2 informative.

To test the lookup reading, I reran the unchanged runner at the same commit on CPU with the repository's `tiny-random` model: a frozen, randomly initialized 2-layer Llama with hidden size 32 and a byte tokenizer, which has seen no text. This control was not pre-registered, was run after the main results, and has one seed and 20 programs per long split (303, 640, and 2,543 questions), so only the contrasts are informative:

| Arm | 2-8 | 32 | 64 | 256 |
|---|---|---|---|---|
| B | 0.492 | 0.135 | 0.097 | 0.088 |
| C | 0.553 | 0.211 | 0.147 | 0.082 |
| S | 0.938 | 0.888 | 0.847 | 0.865 |
| W | 0.501 | 0.195 | 0.128 | 0.083 |
| Compounding baseline | 0.947 | 0.892 | 0.882 | 0.867 |

- S's action accuracy was 0.940, 0.930, 0.937, and 0.937 across lengths, so ε = 0.060 and `a` = 1.000. S stayed 0.002 to 0.035 from the baseline, inside the pre-registered 0.05, and the delta-rule arm matched S with zero mismatches. Prediction 2 therefore holds here non-vacuously: with a controller that makes mistakes, per-statement errors accumulate as the replay predicts and do not grow with position.
- Prediction 1 also holds (S − C = +0.700 at 64 and +0.783 at 256, bounds 0.056 and 0.028).
- A random model with no pretraining gets within 0.06 of SmolLM2's S on 2-8 statements, and its full-context arms fall as steeply. Most of S's accuracy therefore comes from distinguishing 128 strings, not from pretrained language knowledge. What the pretrained features add is the last 6% of action accuracy.
- Record: [e09-20261125T093316Z-c7d252a-28de](e09-20261125T093316Z-c7d252a-28de/), local CPU, no cost.

### Model positions

Maximum positions in one forward pass, seed-independent. B and C use the gold transcript of the test programs, so these are lower bounds on a free-running B's context.

| Arm | 2-8 | 32 | 64 | 256 |
|---|---|---|---|---|
| B | 132 | 472 | 911 | 3,556 |
| C | 59 | 192 | 359 | 1,383 |
| S, S+delta, W | 17 | 17 | 17 | 17 |

S reads about 17 positions per window at any length, and its total positions per program grow linearly with length: 3,878 on average at 256 statements, against 1,328 for C. It spends more total model work than C and shorter windows.

## Interpretation

> **TL;DR:** The length decline in E02 is a property of reading long contexts, not of the machine or the heads; whether the language model contributes anything to S is not shown.

- **Where E02's long-program loss comes from.** The same heads, task, and machine make no mistakes when each statement is read alone. The decline of B and C with length therefore is not a limit of the machine, of the typed interface, or of the head architecture. It appears when the model must pick out a statement's action inside a long context. This supports the thesis that state belongs in the machine, in this setting.
- **Why this is weaker than it looks.** The training programs contain 128 distinct statement texts, and so does each long test split (127 on the 2-8 split): 8 questions and 120 `set` sentences, since the 8 held-out record-value pairs are excluded from training and from every test split run here. S's job is to map each of the 128 strings to its action, and a frozen model's last-layer feature plus a small MLP separates them. The perfect score therefore shows that this lookup is learnable, not that the language model understands statements, and the random-model control above supports that reading. The held-out-pair split of E02, where C scored 0.827, is the one that would test unseen statements, and E09 omitted it.
- **What the compounding baseline showed.** With SmolLM2, nothing: ε = 0 makes it flat at 1.000. With the random-model control, where ε = 0.06, S tracked the baseline within 0.035 at every length, which is weak evidence (one seed, small splits) that a bounded-window controller's errors compound independently and do not grow with position.
- **Cost of the design.** S's model work grows linearly with program length and needs no context beyond one statement, but it assumes every statement can be read without earlier ones. The task has that property by construction; tasks where a statement's meaning depends on earlier statements (references, pronouns) are not measured.

## Costs

> **TL;DR:** About $1.97 for one job; arm B on the 256-statement split dominated the time.

- HF job [`6b062c26…`](https://huggingface.co/jobs/jgalego/6b062c261460ffe1d5e9650b): 8,860 s running on `l4x1` at $0.80/h, about $1.97. The pre-registered estimate was a few dollars at most; the job finished in 2.5 hours of a 4-hour timeout.
- Evaluating B on 100 programs of 256 statements took 1,972 s per seed (about 342,000 sequential model positions). C and S took about 5.5 s on the same split because their features are computed once per statement.
- Peak memory 2,225 MB RSS.

## Limits

> **TL;DR:** The task makes each statement independent and finite, the strongest arm is a lookup, and the run was single.

- **Lookup task:** 128 distinct statements; S saw all of them in training. Generalization to unseen statements is untested here (held-out pairs omitted). The random-model control reached 0.94 on the same task.
- **Gold observation:** the answer head receives the machine's read reply, as in E02 C, so answering is a copy; `a` = 1.000 follows.
- **Vacuous prediction 2:** the baseline did not constrain anything in the recorded run; the control above is a post-hoc substitute with one seed and 20 programs per long split.
- **Free-running B:** B's positions use the gold transcript, not the free-running one.
- **Preamble in the window:** each S window is the preamble plus the statement, not the statement alone.
- **Scope:** one frozen 135M model, one task, one training recipe, heads only, three seeds, no unfrozen or larger-model baseline.
- **Deviations from the pre-registration:** the 256-statement split has 100 programs; `a` is the answer head's accuracy given the gold observation on the 2-8 questions, not task accuracy on correct-action questions; S+delta was run unconditionally rather than after S passed; the held-out-pairs split was dropped. Each is recorded in a [dated amendment](../../EXPERIMENTS.md#amendment-2026-11-25-first-run).

## Artifacts

> **TL;DR:** The manifest is here; the full run directory is in the public Hub dataset.

The run manifest (`run.json`, `metrics.json`) is in [e09-20261125T003100Z-c7d252a-bd6a](e09-20261125T003100Z-c7d252a-bd6a/). The full run directory, including traces, is in the public Hub dataset [`jgalego/latent-machines-runs`](https://huggingface.co/datasets/jgalego/latent-machines-runs) under `runs/`, at commit `af0e7609`.

```bash
git checkout c7d252a && uv sync --locked --extra torch-cuda --extra lm
uv run --locked python -m latent_machines.experiments.e09 --device cuda
```

## Decision

> **TL;DR:** Record a supported result for "long context is the cause" and a null on whether the model carries weight; run the held-out-pair split before building on S.

E09 meets its acceptance criteria: zero executor mismatches, S's action accuracy constant across lengths, and predictions 1 and 2 held. Prediction 3 failed as written and is recorded as such, with the explanation above given as interpretation. Next steps, not run:

- S, W, and C on a held-out-pairs split, so S has to read statements it never trained on. If S falls there, the 100% is memorization, and the thesis needs a task whose statements vary.
- A harder statement language (varied phrasing, references to earlier statements) so that ε is above zero and the compounding baseline has something to predict.
- [E10](../../EXPERIMENTS.md#e10---a-machine-layer-inside-the-model) and [E11](../../EXPERIMENTS.md#e11---learning-to-use-the-machine-from-outcomes) wait on these; E10's discriminator against E03 arms S and H is unchanged.
