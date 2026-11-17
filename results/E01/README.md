# E01 results: compiled transducers and a delta-rule stack

> **TL;DR:** Small state machines and a stack, compiled into neural layers, matched their references exactly for 26 million actions.

In plain terms: we turned three tiny state machines (a parity tracker, a counter, and a tracker for shuffles of three items) into two-layer neural networks, and built a stack out of neural-network memory. Checked against plain references after every action, in four number formats, they never disagreed and never drifted. This holds for these small machines on CPU, not for neural networks in general.

## Question and hypothesis

> **TL;DR:** Can simple exact machines run directly in layers, without a general-purpose CPU? The hypothesis says yes.

[E01](../../EXPERIMENTS.md#e01---exact-streaming-machines-and-transition-algebras) asks whether useful specified algorithms can run directly without a general-purpose CPU. This run tests two of its variants:

> A ReLU lookup circuit with one hidden unit per (state, symbol) pair and explicit feedback implements [`transducer/v1`](../../src/latent_machines/machines/README.md#transducerv1) exactly, and a delta-rule matrix with a one-hot depth pointer implements [`stack/v1`](../../src/latent_machines/machines/README.md#stackv1) exactly, including overflow and underflow, in float32, float64, bfloat16, and float16 on CPU.

## Evidence class

> **TL;DR:** Our own finite tests: strong evidence for these machines, not a proof.

Local conformance on finite, declared workloads, run by this project; not independently checked. Every action is compared with the oracle: event, work, and decoded state, plus residue before decoding. The method is the shared [lockstep check](../../src/latent_machines/experiments/README.md#e01-exact-streaming-machines), the same one E00 uses.

## Setup

> **TL;DR:** Three transducers and one stack, six workloads, four number formats.

| Item | Value |
|---|---|
| Source | [`765c759`](https://github.com/JGalego/latent-machines/commit/765c759038867fecb4f609ff212ec35f5858cf40), clean tree |
| Transducers | parity (2 states, 2 symbols), mod-5 counter over increments 0-2 (5 states, 3 symbols), S3 tracker under adjacent swaps (6 states, 2 symbols) |
| Transducer backend | `relu-lookup`: `z = relu(W_in [s; x] − 1)`, `s′ = W_next z`, `y = W_out z`, eager PyTorch 2.15.0+cpu |
| Stack | `stack/v1`, capacity 8, 8-bit words |
| Stack backend | `delta-stack`: E00's delta-rule update plus a one-hot depth pointer moved by shift matrices; overflow and underflow read from the pointer |
| Workloads | transducers: random symbols, about 5% out of range; stack: bracket parser, random push/pop, fill-and-drain cycles ([generators](../../src/latent_machines/tasks/)) |
| Snapshot check | At the midpoint, decoded state → JSON → restore; the restored tensors must equal the live ones exactly |

A variant is one machine, workload, and number format: 24 per run.

## Results

> **TL;DR:** Five runs, zero mismatches, zero drift, and every snapshot restored.

| Actions per variant | Seed | Where | Actions checked | Mismatches | Max residue | Snapshots | Status | Record |
|---|---|---|---|---|---|---|---|---|
| 10^3 | 0 | local, Intel i7-1365U | 24,000 | 0 | 0 | 24/24 agree | passed | [cf85](e01-20261117T104348Z-765c759-cf85/metrics.json) |
| 10^3 | 1 | local, Intel i7-1365U | 24,000 | 0 | 0 | 24/24 agree | passed | [f4b4](e01-20261117T104400Z-765c759-f4b4/metrics.json) |
| 10^3 | 2 | local, Intel i7-1365U | 24,000 | 0 | 0 | 24/24 agree | passed | [a3b4](e01-20261117T104413Z-765c759-a3b4/metrics.json) |
| 10^5 | 0 | HF Jobs `cpu-upgrade` (8 CPUs), AMD EPYC 7R13 | 2,400,000 | 0 | 0 | 24/24 agree | passed | [0306](e01-20261117T104353Z-765c759-0306/metrics.json) |
| 10^6 | 0 | HF Jobs `cpu-upgrade` (8 CPUs), AMD EPYC 7R13 | 24,000,000 | 0 | 0 | 24/24 agree | passed | [331d](e01-20261117T104505Z-765c759-331d/metrics.json) |

Every variant in every run is classified `exact`. The seed-0 10^3 run, whose full traces were kept locally, triggered every fault kind: 592 out-of-range symbols, 8 out-of-range push values, 732 overflows, and 288 underflows, summed over its four number formats, which each see the same actions.

## Circuit and state sizes

> **TL;DR:** The compiled circuits are tiny: four nonzero weights per state-symbol pair.

| Machine | Hidden width | Depth | Nonzero weights | Stored weights | Logical state | State tensor (float32) |
|---|---|---|---|---|---|---|
| parity | 4 | 2 | 16 | 32 | 1 bit | 8 bytes |
| mod-5 counter | 15 | 2 | 60 | 270 | 3 bits | 20 bytes |
| S3 tracker | 12 | 2 | 48 | 240 | 3 bits | 24 bytes |
| stack (capacity 8) | none | none | 8 (shift matrix) | 81 | 68 bits | 292 bytes |

Each action is one tick. One-hot state costs one entry per state, so state bytes grow with the number of states, not with log(states).

## Costs

> **TL;DR:** About a minute of laptop time plus about one cent of cloud compute.

- Local 10^3 runs: about 12 s each, with 4 workers.
- HF job [`6afc2fdc…`](https://huggingface.co/jobs/jgalego/6afc2fdc884144d2f44f27ba): 1,017 s running on `cpu-upgrade` at $0.03/h, about $0.01, with 8 worker processes.
- End-to-end time on HF was 0.12–0.36 ms per action, including oracle stepping and checks. This is not a backend latency measurement.

## Limits

> **TL;DR:** Tiny machines, ReLU circuits only, CPU only, and some checks done by host code.

- **Not tested:** smooth (SwiGLU) lookup circuits, direct signed, reflection, or rotation recurrences (DeltaProduct, Complex KDA, RWKV-7, Mamba-3), LSTM/GRU cells, mutable-record assignment, larger state spaces or permutation groups, deeper stacks, GPUs, and fused kernels.
- **Margins:** residue covers the fed-back state only. Hidden-unit and output activations were not measured separately, although every one decoded correctly.
- **Host checks:** ordinary Python rejects out-of-range symbols and push values before the tensors run. The stack's overflow and underflow decisions come from the tensor pointer.
- **Workload coverage:** at capacity 8, bracket texts underflowed but never overflowed. Overflow is covered by the random push/pop and fill-and-drain workloads.
- **One seed for the long sweeps:** the 10^5 and 10^6 sweeps used seed 0 only.

## Artifacts

> **TL;DR:** Summaries live here; raw traces live in a public Hub dataset.

Run manifests (`run.json`, `metrics.json`) are in this directory. The HF runs' manifests now record the job ID, CPU quota, flavor, image, and timeout. Raw traces (first 1,000 transitions per variant) and snapshots from the HF runs are in the public Hub dataset [`jgalego/latent-machines-runs`](https://huggingface.co/datasets/jgalego/latent-machines-runs) at commit `96f98985`.

```bash
git checkout 765c759 && uv sync --locked --extra torch-cpu
uv run --locked python -m latent_machines.experiments.e01 --horizon 100000 --jobs 8 --trace-limit 1000
```

## Decision

> **TL;DR:** The first milestone is met; next comes E02, the first learning test.

With [E00](../E00/README.md), this meets the program's first milestone for the tested setups: one conforming memory backend, one useful exact streaming machine, and reproducible evidence. The backends also meet the conformance and snapshot condition of [scope gate 1](../../EXPERIMENTS.md#gates-for-expanding-scope).

Next steps:
- **Slice:** step 6 is E02, a small LM comparing no machine, textual tools, an internal native module, and the internal delta-rule backend.
- **E01 variants:** SwiGLU circuits and signed recurrences for parity remain open.
