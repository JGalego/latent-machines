# E02 results: learned interface comparison

> **TL;DR:** A typed internal interface beat a text tool on longer and held-out programs; compiled and native executors gave identical results.

In plain terms: a small frozen language model, with small trained heads, tracked record updates by using a memory machine. Handing the machine's replies back as typed values worked better than appending them to the text, most clearly on programs longer than those seen in training. Replacing the plain-code memory with the neural delta-rule memory changed no answer. All controllers degraded sharply on long programs, so the controller, not the machine, limits accuracy.

## Question and design

> **TL;DR:** Same task, features, and heads; only the interface or the executor changes between arms.

[E02](../../EXPERIMENTS.md#e02---learned-interface-comparison) asks whether an internal interface helps a learned controller, and whether compiling the machine adds anything beyond native integration. Programs assign values from 0 to 15 to records a to h and ask for current values, for example "Set c to 7. What is c?". Each `set` maps to a write and each question to a read on [`memory/v1`](../../src/latent_machines/machines/README.md#memoryv1) with 8 slots of 4 bits.

A frozen [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) (revision `93efa2f`) reads each program. Only small MLP heads on its last-layer hidden states are trained:

| Arm | Interface | Executor | Trainable parameters |
|---|---|---|---|
| A | None; the answer head reads the question's features | None | 152,976 |
| B | Request and reply appended to the context as text, e.g. `[read c] [7]` | Integer oracle | 308,522 |
| C | Typed action; the reply value is embedded and given to the answer head | Integer oracle | 317,290 |
| D | C's trained heads | Delta-rule memory ([E00](../E00/README.md)), with a shadow oracle | 317,290 (shared with C) |

C versus B tests the interface; D versus C tests the implementation.

## Evidence class

> **TL;DR:** One local run with three training seeds on paired test programs; differences are reported with seed ranges, not significance tests.

All arms share the training data and the test programs, so comparisons are paired by program. Training uses 2,000 programs of 2 to 8 statements, 30 epochs, and three seeds per arm. Evaluation is free-running: predicted actions drive the machine, so a wrong write affects later answers. This is a single local experiment with one model, one task, and one training recipe; it is not independently reproduced.

## Setup

> **TL;DR:** Committed revision, one GPU job, test splits for length and for unseen record-value pairs.

| Item | Value |
|---|---|
| Source | [`463bb48`](https://github.com/JGalego/latent-machines/commit/463bb484a08d142568a38c850795dc4235fa496d), clean tree; method and settings as committed in `559b56b` |
| Hardware | HF Jobs `l4x1` (NVIDIA L4, CUDA 13.2) for the frozen model; heads and executors on CPU |
| Training | 2,000 programs, 10,037 action examples, 5,022 answer examples, 1,200 steps per arm and seed, seeds 0-2 |
| Test splits | 300 programs of 2-8 statements (769 questions); 100 of 32 (1,592); 100 of 64 (3,216); 300 using record-value pairs held out of training (604) |

## Results

> **TL;DR:** C beats B on every split and every seed; the gap is clear beyond training length and on held-out pairs.

Task accuracy, mean over three seeds with the range in brackets:

| Arm | 2-8 statements | 32 statements | 64 statements | Held-out pairs |
|---|---|---|---|---|
| A | 0.718 [0.710, 0.730] | 0.257 [0.255, 0.259] | 0.154 [0.145, 0.160] | 0.531 [0.517, 0.545] |
| B | 0.871 [0.866, 0.875] | 0.389 [0.380, 0.394] | 0.224 [0.220, 0.228] | 0.727 [0.720, 0.740] |
| C | 0.893 [0.874, 0.908] | 0.461 [0.455, 0.470] | 0.286 [0.273, 0.293] | 0.827 [0.825, 0.831] |
| D | 0.893 [0.874, 0.908] | 0.461 [0.455, 0.470] | 0.286 [0.273, 0.293] | 0.827 [0.825, 0.831] |

Paired differences in task accuracy on the same test programs, per seed. The last column is a conservative two-standard-error bound for a difference of two proportions at that number of questions:

| Comparison | Split | Seeds 0, 1, 2 | Mean | Bound |
|---|---|---|---|---|
| C − B | 2-8 statements | +0.025, +0.008, +0.033 | +0.022 | 0.051 |
| C − B | 32 statements | +0.066, +0.060, +0.090 | +0.072 | 0.035 |
| C − B | 64 statements | +0.070, +0.045, +0.072 | +0.062 | 0.025 |
| C − B | Held-out pairs | +0.104, +0.091, +0.104 | +0.100 | 0.058 |
| D − C | All splits | 0, 0, 0 | 0 | |

B and C both beat A on every split, by 0.07 to 0.30.

Action accuracy, the fraction of statements whose chosen action matched the gold action:

| Arm | 2-8 statements | 32 statements | 64 statements | Held-out pairs |
|---|---|---|---|---|
| B | 0.949 | 0.728 | 0.593 | 0.920 |
| C and D | 0.873 | 0.538 | 0.393 | 0.770 |

D's delta-rule executor matched its shadow oracle on every action: zero executor mismatches.

## Interpretation

> **TL;DR:** The typed interface helps after the action is chosen; long programs fail because the controller's actions degrade.

- **Interface.** C beat B on every split in every seed. Beyond the training length and on held-out pairs the gap exceeds the conservative bound; at the training length (+0.022) it does not. B chose actions more accurately than C, likely because B's action head also sees its own earlier requests and replies, yet C answered more questions correctly. C's advantage therefore arises after the action: a typed observation is easier for the answer head to use than a reply that the frozen model must read back from text. This explanation is a hypothesis; the experiment does not isolate it.
- **Implementation.** Replacing the oracle with the delta-rule memory changed no answer on any program or seed, as expected from E00's conformance results. Compiling the machine adds no accuracy here.
- **Length.** Every arm degraded with program length, and so did action accuracy. The machines execute exactly, so the loss comes from the controller: heads trained on short programs choose worse actions later in long programs. Why, for example because frozen features shift with position, is untested.

## Costs

> **TL;DR:** One GPU job for about 73 cents; the text tool processed 2.6 times as many tokens.

- HF job [`6afc57f2…`](https://huggingface.co/jobs/jgalego/6afc57f24643bac7ac628403): 3,286 s running on `l4x1` at $0.80/h, about $0.73.
- On the 64-statement split, arm B processed 86,845 language-model tokens per seed against 33,984 for A, C, and D: its requests and replies add 52,861 tokens.
- Evaluation times are not comparable across arms. A, C, and D reuse features computed once before evaluation, while B's evaluation includes its incremental model calls. D's time also includes the shadow oracle.

## Limits

> **TL;DR:** One small frozen model, one task, heads only, and no feedback into the model.

- **Controller:** a frozen 135M-parameter model with trained heads; no fine-tuning, no other model sizes. Unfreezing the model is a separate ablation, not run.
- **Interface:** in C and D the typed observation reaches only the answer head, not later layers of the model; latent feedback is E03.
- **Task:** one record-update task with 8 records and 4-bit values. Capacity changes, unseen operation compositions, and the validity-constraint ablation were not tested.
- **Statistics:** three training seeds, paired test programs, and a conservative unpaired bound; no formal paired test.
- **Exploration:** before this run, a one-seed pilot with 300 training programs, on uncommitted code, looked at the same test splits. The method and settings were not changed after it.

## Artifacts

> **TL;DR:** The manifest is here; the full run directory is in the public Hub dataset.

The run manifest (`run.json`, `metrics.json`) is in this directory. The full run directory, including traces of the first programs per split, is in the public Hub dataset [`jgalego/latent-machines-runs`](https://huggingface.co/datasets/jgalego/latent-machines-runs) at commit `15ad5b5a`.

```bash
git checkout 463bb48 && uv sync --locked --extra torch-cuda --extra lm
uv run --locked python -m latent_machines.experiments.e02 --device cuda
```

## Decision

> **TL;DR:** Record the interface result; next, make the controller hold up on long programs.

This is the program's first scientific milestone: an interface-versus-implementation result. For this task and controller, the typed internal interface outperforms the text tool beyond the training distribution, and the compiled executor matches the native one exactly. Next steps:
- A pre-registered E02 variant aimed at length generalization of the controller, evaluated on a validation split rather than these test splits.
- E03: latent feedback, so typed observations can reach later model computation.
