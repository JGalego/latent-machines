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

## E07: optimizer-defined memory

> **TL;DR:** Check optimizer-written memory against its own reference every step, and report where it stops behaving like RAM.

[e07.py](e07.py) runs each setting of [`optimizer-memory/v1`](../machines/README.md#optimizer-memoryv1) three ways over E00's workloads and an online regression task:

```mermaid
flowchart LR
    W["workload"] --> B["tensor backend"]
    W --> R["float64 reference"]
    W --> O["memory/v1 oracle"]
    B --> L{"one reference step from the<br/>backend's state agrees?"}
    R --> T["trajectory agreement<br/>(reported)"]
    B --> T
    B -->|decoded readout| C{"same words and events<br/>as memory/v1?"}
    O --> C
    L -->|pass condition| P[status]
    C -->|"finding; required only for sgd"| P
```

Settings change one feature at a time from plain SGD: learning rate 0.9 or 0.5, stale chunk-start gradients, momentum, decay, gradient normalization, and an MLP. A variant is one setting, workload, and dtype. Its `classification` is the local own-reference result; `trajectory` and `ram_conformance` record the other two comparisons, and `canonical_snapshot_resume` says whether restarting from the decoded `memory/v1` words alone reproduces the decoded behavior.

```bash
uv run --locked python -m latent_machines.experiments.e07                       # 10^3 actions, float32 and float64
uv run --locked python -m latent_machines.experiments.e07 --dtypes bfloat16
```
