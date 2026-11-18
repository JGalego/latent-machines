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

## E03: latent ticks and explicit events

> **TL;DR:** Four ways to return a machine's reply to a frozen model's controller, compared on copies that need the reply.

[e03.py](e03.py) runs record programs with copies ([task](../tasks/copies.py)): `Copy b to e.` is a read of `b`, then a write of the value read to `e`. Trained heads decide one tick at a time (`write`, `read`, or `end`; at most three machine actions per statement), and each tick's action and reply form a feedback code ([interface](../interfaces/ticks.py)). Arms differ only in where that code re-enters:

```mermaid
flowchart LR
    S["statement text"] --> LM["frozen LM"]
    LM -->|features| H["trained heads"]
    H -->|tick action| X["delta-rule memory<br/>+ shadow oracle"]
    X -->|"feedback code"| R{arm}
    R -->|"T: request and reply as text"| LM
    R -->|"L: one adaptor vector<br/>at the next position"| LM
    R -->|"H: embedded at the heads"| H
    R -->|"S: code to the heads, plus a<br/>constant adaptor vector to the LM"| H
    N["N: macro per statement;<br/>native loop runs the ticks"] -.-> X
```

Arm S is a control for L: the same trained vector slots, trained end to end through the frozen model, but carrying a constant instead of the feedback, which goes to the heads. If S matched L, L's gains would come from trained inputs steering the model rather than from feedback through it.

All arms execute on the delta-rule memory with a shadow oracle. For a sample of programs, each arm is snapshotted between the read and the write of a copy (machine state, controller history, pending work), restored through JSON, and must finish exactly as an uninterrupted run.

```bash
uv run --locked python -m latent_machines.experiments.e03 --model tiny-random   # smoke run
uv run --locked python -m latent_machines.experiments.e03 --device cuda         # recorded configuration
```
