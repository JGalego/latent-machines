# Experiments

> **TL;DR:** Experiment runners compose machines, backends, and workloads, then record conformance results.

See the [experiment program](../../../EXPERIMENTS.md#experiments) and the [run-record contract](../../../EXPERIMENTS.md#run-records-and-results).

## E00: memory conformance

> **TL;DR:** Run reference and neural memory side by side, and stop at the first disagreement.

[e00.py](e00.py) checks the eager [delta-rule backend](../backends/README.md) against the [`memory/v1`](../machines/README.md#memoryv1) oracle on every [workload](../tasks/memory_workloads.py) and number format, one action at a time.

```mermaid
flowchart LR
    W["workload<br/>(seeded actions)"] --> O[oracle]
    W --> B[backend]
    O --> C{"same event, work,<br/>and decoded state?"}
    B --> C
    B -->|before decoding| R[residue]
    C -->|yes| N[next action]
    C -->|no| F["failure.json<br/>first mismatch"]
    N -.->|at snapshot_at| S["snapshot → JSON → restore<br/>must equal the live tensor"]
```

A variant is one workload in one number format. It is classified `exact` (no mismatch, zero residue), `conforms with residue` (no mismatch, but entries drifted from 0 or 1), or `mismatch` (stopped at the first wrong transition). The run passes only if no variant mismatches and every snapshot restores the live tensor exactly.

```bash
uv run --locked python -m latent_machines.experiments.e00                                    # 10^3 actions per variant
uv run --locked python -m latent_machines.experiments.e00 --horizon 1000000 --jobs 8 --trace-limit 1000
```

Each run writes `runs/e00-<time>-<revision>-<id>/` with `run.json`, `metrics.json`, `trace.jsonl`, `snapshots/`, and, when needed, `failure.json` and `source.patch`. Exit status is 0 on pass, 1 on a mismatch or snapshot disagreement, and 2 if the run failed.

## E01: exact streaming machines

> **TL;DR:** Compare compiled transducers and a delta-rule stack against their integer oracles after each action.

[e01.py](e01.py) uses the same [lockstep check](lockstep.py) as E00:

```mermaid
flowchart LR
    subgraph T["transducer/v1: parity, mod-5 counter, S3 tracker"]
        TS["random symbols<br/>(5% out of range)"] --> TO[oracle]
        TS --> TL[ReLU lookup circuit]
    end
    subgraph S["stack/v1"]
        SW["bracket parser · random push/pop ·<br/>fill-and-drain cycles"] --> SO[oracle]
        SW --> SD[delta-rule stack]
    end
    TO & TL --> C{"same event, work,<br/>decoded state?"}
    SO & SD --> C
```

Variants are named `backend/dtype/machine/workload`, for example `delta-stack/bfloat16/stack/brackets`. Metrics add circuit width, depth, and stored and nonzero weights.

```bash
uv run --locked python -m latent_machines.experiments.e01                                    # 10^3 actions per variant
uv run --locked python -m latent_machines.experiments.e01 --horizon 1000000 --jobs 8 --trace-limit 1000
```

## E06: spatial and dynamical machines

> **TL;DR:** Check cellular rules cell for cell, and measure linear filters against a tolerance, in separate runs.

[e06.py](e06.py) has two branches that write separate run records. The exact branch uses the [lockstep check](lockstep.py) on [`cellular/v1`](../machines/README.md#cellularv1); the numerical branch compares each output sample with the [`lds/v1`](../machines/README.md#ldsv1) reference.

```mermaid
flowchart LR
    subgraph X["--branch exact"]
        G["rule × workload<br/>(seeded grid, Advance/Set)"] --> O[oracle]
        G --> B["torch-int · conv-lookup<br/>(4 float formats)"]
        O & B --> C{"same grid, event, work?"}
    end
    subgraph N["--branch numerical"]
        U["system × input signal"] --> R[float64 reference]
        U --> M["direct · ssm-sequential · ssm-chunked<br/>(float64, float32, bfloat16)"]
        R & M --> T{"max error / rms ≤ declared tolerance?"}
    end
```

Exact variants are named `backend/dtype/rule/workload`; numerical ones `backend/dtype/system/signal`. Tolerances are fixed in the code and recorded in `run.json`: 1e-9 for float64, 1e-4 for float32, 1e-2 for bfloat16.

```bash
uv run --locked python -m latent_machines.experiments.e06                       # exact, 10^3 actions per variant
uv run --locked python -m latent_machines.experiments.e06 --branch numerical    # 10^4 samples per variant
```
