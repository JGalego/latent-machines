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

[e01.py](e01.py) uses the [lockstep check](lockstep.py) that E00: memory uses:

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

## E04: dataflow and conditional units

> **TL;DR:** Evaluate the same integer programs on conventional, scalar-machine, and compiled tensor backends; check every evaluation and time each path.

[e04.py](e04.py) runs every [dataflow backend](../backends/README.md#dataflow-backends) on seeded [workloads](../tasks/dataflow_workloads.py): expression DAGs, a 64-element dot product, and gather/map/reduce/scatter. Each evaluation (one program, one input vector) is compared with the oracle: outputs or faulting node, work, and every evaluated node's value. Variants do not stop at the first mismatch; they count mismatching evaluations and keep the first in `failure.json`.

```mermaid
flowchart LR
    W["workload<br/>(programs × input vectors)"] --> O["dataflow/v1 oracle"]
    W --> B["backend × number format"]
    O & B --> C{"same event, work,<br/>node values?"}
    B -.->|middle level| S["decode → JSON → restore → encode<br/>resume must match the live run"]
    W --> T["timing: compile, cold call,<br/>batch-1 latency, batch throughput"]
```

bfloat16 and float16 variants are precision probes. They are reported like the others but excluded from acceptance, since they cannot represent every 16-bit word. Timing amortizes each compile against the Python-int baseline at batch 1 and the int64 baseline per batched evaluation.

```bash
uv run --locked python -m latent_machines.experiments.e04       # 20 programs × 50 inputs per workload
```

## E05: graph, semiring, and logic execution

> **TL;DR:** Check dense tensor graph kernels round by round against integer oracles, then time them against BFS and Dijkstra.

[e05.py](e05.py) runs each [fixed-point machine](../machines/README.md#reachv1-minplusv1-chain-datalogv1) problem to its terminal event with the oracle, then replays that many `Iterate` actions through the [lockstep check](lockstep.py) with the [dense tensor backend](../backends/README.md#semiring-tensor-kernels). Conventional variants (`python-int`) compare each oracle fixed point with BFS, Dijkstra, or semi-naive evaluation.

```mermaid
flowchart LR
    G["seeded problems<br/>(graphs, Datalog databases)"] --> O["oracle rounds<br/>to halt or fault"]
    O --> L["lockstep: tensor backend<br/>per round, per dtype"]
    O --> K["conventional algorithm<br/>same fixed point?"]
    G --> T["timing benchmark<br/>larger sparse graphs"]
```

Variants are named `backend/dtype/machine/workload`. A tensor variant counts checked rounds and the problems with a nonconforming round; one problem per variant is snapshotted mid-run. The benchmark checks every timed answer against BFS or Dijkstra.

```bash
uv run --locked python -m latent_machines.experiments.e05                     # 40 problems per variant
uv run --locked python -m latent_machines.experiments.e05 --horizon 200 --seed 1
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

## E07: optimizer-defined memory

> **TL;DR:** Check optimizer-written memory against its own reference every step, and report where it stops behaving like RAM.

[e07.py](e07.py) runs each setting of [`optimizer-memory/v1`](../machines/README.md#optimizer-memoryv1) three ways over the workloads of E00: memory and an online regression task:

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

## E08: search with an exact verifier

> **TL;DR:** Check the compiled verifier against the oracle, then compare search procedures at the same step budget with every claimed solution re-verified.

[e08.py](e08.py) has two parts. Verifier conformance compares the [tensor CNF verifier](../backends/README.md#tensor-cnf-verifier) with the [`cnf/v1`](../machines/README.md#cnfv1) oracle on every candidate of five [workloads](../tasks/sat_instances.py), in each lowering and dtype. Search runs the [procedures](../controllers/sat_search.py) on four 3-SAT instance classes:

```mermaid
flowchart LR
    I["3-SAT instance"] --> D["DPLL<br/>(complete)"]
    I --> W[WalkSAT]
    I --> R["random proposals"]
    I --> E["energy proposals<br/>(Langevin on a relaxation)"]
    R --> O["cnf/v1 oracle"]
    E --> O
    E --> T["tensor verifier<br/>float32, bfloat16"]
    D -->|"sat or unsat"| C{"oracle re-check;<br/>unsat by enumeration ≤ 20 vars"}
    W -->|sat| C
    O --> S["solved / unknown"]
    T --> S
```

The budget counts steps: a variable assignment for DPLL, a flip for WalkSAT, a verified candidate for proposers. Only DPLL can report `unsat`; the others report `unknown` when the budget runs out. The two energy arms see the same proposals, so their accept decisions must be identical and their difference is verification cost alone.

```bash
uv run --locked python -m latent_machines.experiments.e08          # default run, about 3 minutes on 2 threads
uv run --locked python -m latent_machines.experiments.e08 --instances 0 --horizon 10000   # verifier conformance only
```

Each variant records whether its formulas stay within the dtype's exact-integer range for clause counts (`within_declared_limits`). The run passes when every variant within range matches the oracle, every claimed solution is accepted by the oracle, the two energy arms agree, and every checked DPLL unsatisfiability claim is confirmed. Exit status is 0 on pass, 1 on a failed check, and 2 if the run failed.
