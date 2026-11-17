# E00 results: delta-rule memory conformance

> **TL;DR:** Neural-network memory matched the reference exactly for 35 million actions, in the tested setup.

In plain terms: we stored a small computer memory inside the update rule of a neural-network layer (Gated DeltaNet) and checked it against a plain reference after every single action. Across 35 million checked actions in four number formats, it never disagreed and never drifted, even slightly. This holds for the tested setup only (16 slots of 8 bits, CPU, an explicit implementation of the update rule), not for neural networks in general.

## Question and hypothesis

> **TL;DR:** Can a Gated DeltaNet state matrix act as exact memory? The hypothesis says yes.

[E00](../../EXPERIMENTS.md#e00---memory-and-transition-conformance) asks which layer families implement the required storage semantics on the chosen execution path. This first run tests one variant, delta replacement:

> The explicit Gated DeltaNet recurrence, with one-hot address keys, bit-plane values, and gates set to their endpoints (`α = 1`, `β ∈ {0, 1}`), implements [`memory/v1`](../../src/latent_machines/machines/README.md#memoryv1) exactly in float32, float64, bfloat16, and float16 on CPU.

## Evidence class

> **TL;DR:** Our own finite tests: strong evidence for this setup, not a proof.

Local conformance on a finite, declared suite, run by this project; not independently checked. Every action is compared with the integer oracle: event, work, and decoded state, plus residue before decoding. Passing is evidence for the tested configuration, not a proof of exactness for other inputs, sizes, or kernels. The method is described in the [E00 runner](../../src/latent_machines/experiments/README.md#e00-memory-conformance).

## Setup

> **TL;DR:** 16 slots of 8 bits, 8 workloads, and 4 number formats, on CPU.

| Item | Value |
|---|---|
| Source | [`ad888ad`](https://github.com/JGalego/latent-machines/commit/ad888ad5bb357ae245aef946ce576bf02139a5f6), clean tree; `uv.lock` sha256 `7e49ea11…` |
| Machine | `memory/v1`, capacity 16, 8-bit words |
| Backend | `delta-eager`: eager PyTorch 2.15.0+cpu, one matmul and two outer products per action, key norm `x / max(‖x‖, 1e-6)`, output thresholded at 0.5 |
| Number formats | float32, float64, bfloat16, float16 |
| Workloads | overwrite, alternate, untouched, idle, clear_region, capacity, invalid, random_mix ([generators](../../src/latent_machines/tasks/memory_workloads.py)) |
| Snapshot check | At the midpoint, decoded state → JSON → restore; the restored tensor must equal the live one exactly |

A variant is one workload in one number format: 32 per run.

## Results

> **TL;DR:** Five runs, zero mismatches, zero drift, and every snapshot restored.

| Horizon per variant | Seed | Where | Actions checked | Mismatches | Max residue | Snapshots | Status | Record |
|---|---|---|---|---|---|---|---|---|
| 10^3 | 0 | local, Intel i7-1365U | 32,000 | 0 | 0 | 32/32 agree | passed | [69f9](e00-20261117T000402Z-ad888ad-69f9/metrics.json) |
| 10^3 | 1 | local, Intel i7-1365U | 32,000 | 0 | 0 | 32/32 agree | passed | [18bf](e00-20261117T000446Z-ad888ad-18bf/metrics.json) |
| 10^3 | 2 | local, Intel i7-1365U | 32,000 | 0 | 0 | 32/32 agree | passed | [7b73](e00-20261117T000500Z-ad888ad-7b73/metrics.json) |
| 10^5 | 0 | HF Jobs `cpu-upgrade`, AMD EPYC 7R13 | 3,200,000 | 0 | 0 | 32/32 agree | passed | [9581](e00-20261117T000412Z-ad888ad-9581/metrics.json) |
| 10^6 | 0 | HF Jobs `cpu-upgrade`, AMD EPYC 7R13 | 32,000,000 | 0 | 0 | 32/32 agree | passed | [7742](e00-20261117T000713Z-ad888ad-7742/metrics.json) |

Every variant in every run is classified `exact`: no first mismatch exists up to 10^6 actions, and no state entry ever left {0, 1}. Each action took one tick. The state tensor holds 128 entries, 512 bytes in float32, for 128 logical bits.

## Costs

> **TL;DR:** A minute of laptop time plus about two cents of cloud compute.

- Local 10^3 runs: under 25 s each.
- HF job [`6afb99e5…`](https://huggingface.co/jobs/jgalego/6afb99e5186b78cad7b6ceb9): 2,236 s running on `cpu-upgrade` at $0.03/h, about $0.02, with 8 worker processes.
- End-to-end time on HF was 0.3–0.7 ms per action, including oracle stepping and checks. This is not a backend latency measurement.

## Limits

> **TL;DR:** One layer type in one setup; GPUs, fast kernels, and learning are untested.

- **Not tested:** other capacities and word widths, multi-hot or learned keys, gates away from their endpoints, kernel-style normalization (eps inside the square root), chunked or fused kernels, GPUs, output normalization and gating of a full layer, the KDA, RWKV-7, SSM, and LSTM/GRU variants, and learned control.
- **One seed for the long sweeps:** the 10^5 and 10^6 sweeps used seed 0 only.
- **Not measured:** latency versus capacity, memory traffic.
- **Validation is host-side:** ordinary Python checks actions and tracks halting, not the tensor. The result covers storage and readout in the tensor.

## Artifacts

> **TL;DR:** Summaries live here; raw traces live in a private Hub dataset.

Run manifests (`run.json`, `metrics.json`) are in this directory. Raw traces (first 1,000 transitions per variant) and snapshots from the HF runs are in the private Hub dataset `jgalego/latent-machines-runs` at commit `4b00d50e`. The local runs' raw directories exist only on the machine that ran them, in the ignored `runs/` directory.

Reproduce a run with, for example:

```bash
git checkout ad888ad && uv sync --locked --extra torch-cpu
uv run --locked python -m latent_machines.experiments.e00 --horizon 100000 --jobs 8 --trace-limit 1000
```

## Decision

> **TL;DR:** This backend clears its checks; next, build E01 and test more E00 variants.

For this backend and configuration, the result meets the conformance and snapshot condition of [scope gate 1](../../EXPERIMENTS.md#gates-for-expanding-scope). Next steps:
- **Slice:** continue with step 5 (E01 machines).
- **E00 variants:** add them one at a time, starting with kernel-style key normalization. In an informal check during backend testing, it left reads correct in float32 while state drifted by about 5e-7.
