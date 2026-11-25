# Experiment program

> **TL;DR:** Validate executors before comparing learned interfaces; E00-E08 all have initial results.

Status, 2026-11-25: E00-E09 each have a first recorded run and report under [results/](results/), summarized in the experiment's **Results** line. These are initial results with one or a few seeds, not completed studies. [E09](#e09---state-in-the-machine-not-the-context) ran against pre-registered predictions: two held (one vacuously) and one failed. Its held-out follow-up met two of three and left the third inconclusive. [E10](#e10---a-machine-layer-inside-the-model) and [E11](#e11---learning-to-use-the-machine-from-outcomes) are pre-registered and not yet run, as are follow-ups to E00, E03, and E09 amended below. [RESEARCH.md](RESEARCH.md) records sources and evidence boundaries. [CONTRIBUTING.md](CONTRIBUTING.md) defines code style, tooling, integration, and verification standards.

## Questions and priorities

> **TL;DR:** Separate what runs, how it runs, how it is controlled, and what it is for; start with E00.

Keep these axes separate:

| Axis | Meaning | Example |
|---|---|---|
| Machine | Specified state transitions | Addressable memory, bounded stack, finite transducer |
| Backend | Implementation of those transitions | Conventional code, delta-rule tensors, structured indexed attention |
| Interface | Controller's observations and actions | Textual tool, typed internal actions, latent feedback |
| Task | Workload and evaluation distribution | Record updates, nested scopes, graph reachability |

Not every combination is valid. A backend must declare its supported semantics and limits and reject other configurations explicitly. An approximate optimizer or filter can participate only with its own stated numerical contract, not under an exact-RAM label.

Recommended order:

```text
E00 executor conformance
  -> E01 exact streaming machines
       -> E02 learned interface comparison
            -> E03 latent feedback
```

E04-E08 are independent follow-on branches after their prerequisites pass, not requirements for the first implementation. Report failed hypotheses before adding a larger model or compiler.

## Repository organization

> **TL;DR:** Where code lives: one folder per part, plus curated results and ignored raw runs.

Keep the current research documents at the root. Add an implementation directory only when it holds a working experiment. This is a target layout, not a list of existing files:

```text
README.md
RESEARCH.md
EXPERIMENTS.md
pyproject.toml                  # add with the first executable experiment
uv.lock
src/latent_machines/
  machines/                    # specifications and independent integer oracles
  backends/                    # native, tensor, then optimized implementations
  interfaces/                  # controller observations and action transport
  tasks/                       # generators, trace labels, splits, task metrics
  experiments/                 # small runnable E00, E01, ... entry points
tests/
  machines/                    # semantics, bounds, faults, snapshot/restore
  backends/                    # stepwise conformance and backend differences
  interfaces/                  # observation/action parity and feedback behavior
configs/                       # small versioned settings, when variation requires them
results/                       # curated reports and compact run manifests
runs/                          # ignored raw runs, traces, snapshots, local checkpoints
assets/
```

### Ownership boundaries

> **TL;DR:** Keep machine semantics, execution, observations, and task answers separate; use decoded state only for validation.

```mermaid
flowchart LR
    subgraph E["experiments/: compose parts and write run records"]
        T["tasks/<br/>inputs, ground truth"]
        C[controller]
        I["interfaces/<br/>observations, actions"]
        B["backends/<br/>native, tensor, optimized"]
        M["machines/<br/>specification, integer oracle"]
    end
    T -->|input| C
    C -->|answer| T
    C -->|action| I
    I -->|observation| C
    I -->|action| B
    B -->|event| I
    B -.->|"decoded state (conformance only)"| M
```

Solid edges are the execution path. The dotted edge exists only for validation: decoded backend state is compared against the oracle and never reaches the controller.

- `machines/` defines legal actions, canonical state, numerical domain, transition order, exits, capacity, and work accounting with no dependency on a model family or tensor backend.
- `backends/` owns state encoding, layer-specific lowering, kernels, and conversion to canonical state for conformance and snapshots. Decoding the full state for validation must not leak it to a controller.
- `interfaces/` owns what the controller can observe, how it requests actions, and how results return. It does not own task answers or executor arithmetic.
- `tasks/` owns input generation, ground truth, trace supervision, and split construction. It cannot change machine semantics for a backend.
- `experiments/` composes those parts and writes results. Do not duplicate machine or metric implementations in notebooks or one-off benchmark scripts.
- `tests/` contains bounded regression and conformance checks. Long precision sweeps and performance measurements are explicit experiments, not default unit tests.

Use Python and an integer reference implementation for the first harness, with PyTorch only when the tensor backend is added. Pin dependencies and the interpreter version then. A Python oracle is not an optimized native-performance baseline; introduce and label a suitable conventional implementation before claiming speed advantages.

Do not start with a general compiler, plugin registry, training framework, Rust runtime, or a full ISA. Two concrete machines and backends should show which abstractions to share. Keep differences in state semantics visible rather than hiding them behind one overly permissive `step`.

### First implementation slice

> **TL;DR:** All six steps are implemented and have recorded results, ending with the E02 learned-controller study; E03-E08 followed.

1. Add the uv-managed package and lockfile, focused tests, an integer memory oracle, pinned pre-commit hooks, and fast CPU CI.
2. Specify fixed capacity, word domain, reads/writes, invalid-address behavior, event ordering, and non-mutating faults.
3. Implement an explicit eager delta-rule tensor backend with the same observations and actions. Validate every step, not just the final answer.
4. Add an E00 entry point, a small default run, and the result records described below.
5. Add one compiled finite-state transducer and a bounded stack for E01.
6. Only then add a small LM, a frozen-base/learned-action-head baseline, and E02: learned interfaces. Record exactly what is trained; unfreeze more only as a separate ablation.

Keep CPU runs small by default. GPU sweeps, model downloads, paid jobs, and long training runs must be explicit and separately budgeted.

## Shared execution and evidence contracts

> **TL;DR:** Rules every machine, interface, and number format must follow for a result to count.

These are semantic requirements, not a frozen public Python API.

### Machine transitions

> **TL;DR:** Every machine states exactly what each action does, including errors and costs.

Specify `(state, action) -> (next_state, event, work)` with:

- A finite action set or typed argument schema and validation rules.
- Canonical state and a representation/version identifier.
- `continue`, `output`, `halt`, `fault`, `host_exit`, and `budget_exhausted` where applicable. A machine may support a smaller explicit set.
- The order of reads, computation, writes, and faults. Invalid actions cannot silently clamp or mutate state unless that behavior is explicitly the specification.
- Capacity and overflow behavior, integer widths or fixed-point scales, and the meaning of each work unit.
- Snapshot/restore at declared boundaries, including pending work if an operation spans ticks.

A tick is an execution step, not necessarily a logical operation or CPU instruction. Budget enforcement must say whether an operation can be interrupted, resumed, or rejected when insufficient budget remains.

Reference, dense tensor, and fused backends should produce the same canonical transitions within their shared contract. A native shortcut may implement equivalent semantics; it is not evidence that the actual emitted checkpoint has executed correctly. Test emitted layers separately when they exist.

### Controller and interface

> **TL;DR:** Match observations, action semantics, and resource budgets across interface variants.

Hold constant across interface variants:

- The information available, action semantics, value precision, capacity, stopping rules, and error messages.
- Task data, split definitions, base-model initialization, training-data budget, trainable-parameter accounting, and search/tuning budget.
- Logical machine-work limits; text-token and internal-tick limits must be recorded separately.

Interfaces can have different encodings, but compare them over the same underlying observations. State dumps, reference answers, teacher-forced actions, or privileged scratch variables cannot be supplied only to the internal interface.

Use oracle action replay to isolate executor cost, and free-running learned control to measure the full system. Label teacher forcing and oracle routing clearly; neither establishes autonomous reliability.

### Numerical conformance

> **TL;DR:** Check the actual numerical path and record residue before decoding can hide it.

Record weights, activations, accumulators, stored state, and index/key precision separately. Check:

- Gate endpoints, long idle retention, repeated overwrite, and untouched-slot preservation.
- Read-after-write ordering and key collisions.
- Query/key and output normalization, including one-, two-, four-, and eight-hot cases where the backend permits them.
- Reductions, cancellation, intermediate casts, overflow, and fixed-point scaling.
- Recurrent versus chunked kernels and eager versus optimized execution.
- Batch composition and supported devices.

The pass condition for an exact backend is zero canonical-state, event, and output mismatches on the declared suite. Numerical residue should also be recorded before canonicalization so thresholding cannot conceal a shrinking safety margin. Passing this finite suite is conformance evidence, not a universal proof.

An approximate backend must declare a reference, error metric, tolerance, horizon, and failure policy before running. Report it separately.

## Experiments

> **TL;DR:** E00 to E08, from basic memory checks to search with an exact checker.

### E00 - Memory and transition conformance

> **TL;DR:** Does neural-network memory behave exactly like real memory?

**Question:** which layer families implement the required storage semantics on the chosen execution path?

**Prerequisites:** none. Begin with the integer oracle and explicit GDN-style recurrence; a full pretrained model is unnecessary.

**Variants:** delta replacement; KDA channel gates; RWKV-7; Mamba-2/SSD-style scalar-decay SSM with and without explicit read-modify-write; Mamba-3 complex-valued state evolution as a separate SSM variant; LSTM/GRU gated vector state as a small register file; append-only history with dense lookup. Add Percepta-style indexed hard-max lookup as a distinct variant only when its key, query, tie, and update assumptions are specified. Report each SSM variant separately; a scalar-decay result does not characterize other SSMs.

**Workloads:** overwrite the same slot repeatedly, alternate addresses, preserve untouched slots, idle for long horizons, clear selected regions, hit capacity, and request invalid addresses. For histories, include duplicate keys, equal-score ties, and repeated latest-write queries.

**Acceptance:** zero mismatches against the matching reference semantics; snapshot/resume agrees with uninterrupted execution. Unsupported reset or overwrite behavior must be reported, not emulated invisibly or scored against the wrong contract.

**Measurements:** first mismatch, pre-canonicalization residue, state and index bytes, retained history, memory traffic estimates, ticks per action, and latency versus capacity and horizon. Compare the actions and work KDA needs to clear a selected region against GDN delta replacement; channel-wise retention is a hypothesized saving, not an assumed one. Start with small bounded tests; run named longer sweeps at 10^3, 10^5, and 10^6 actions only as resources permit.

**Discriminator:** a layer that cannot preserve unaddressed state or expose a value through its real normalization path is not a conforming RAM backend, even if a toy update equation works.

**Results:** [results/E00](results/E00/README.md). Delta replacement matched the oracle exactly up to 10^6 actions on CPU; the other variants are untested.

#### Amendment 2026-11-25: chunked and fused kernels, pre-registered

> **TL;DR:** The first run checked a recurrent PyTorch update on a CPU. This follow-up checks a chunked form of the same update, larger capacities, and the fused kernels a real Gated DeltaNet layer would use.

**Question:** does delta replacement stay exact when the recurrence is evaluated in chunks, at capacities well above the first run's, and in the fused kernels of [flash-linear-attention](https://github.com/fla-org/flash-linear-attention)?

**Design:** the E00 overwrite, alternate-address, untouched-slot, idle, and capacity workloads with one-hot keys and β = 1, at capacities 64, 512, and 4,096 and up to 10^5 actions. Variants:

- recurrent PyTorch update (the first run's backend), for reference;
- a pure-PyTorch chunked form of the delta rule (the WY representation, with chunk sizes 4, 16, and 64), in float64, float32, and bfloat16, on CPU;
- flash-linear-attention's `chunk_gated_delta_rule` and `fused_recurrent_gated_delta_rule` on an `l4x1` job, in float32 and bfloat16, if the package installs there. If it does not, the report says so and records the failure.

A read goes through the same normalization path as in the first run. Mismatches are counted against the integer oracle after decoding, and the pre-canonicalization residue is recorded.

**Predictions:**

1. The chunked PyTorch form has zero mismatches in float64 and float32 at every chunk size and capacity.
2. The chunked form in bfloat16 has at least one mismatch on some workload at 10^5 actions, because the chunk-level corrections accumulate rounding that one-hot overwrite does not tolerate. If it has none, the report says the bfloat16 chunked form is exact on this suite.
3. The fused kernels in bfloat16 have at least one mismatch on some workload.

**Outcomes that count against the thesis:** prediction 1 fails: a chunked evaluation of the delta rule does not preserve addressed state exactly, and a layer built on it needs a different update. Predictions 2 and 3 failing support exactness at lower precision, within this suite.

**Cost:** the CPU part runs locally. The kernel check is one `l4x1` job; cap 30 minutes, $0.40.

**Results:** not run.

### E01 - Exact streaming machines and transition algebras

> **TL;DR:** Can simple exact machines, like counters and parsers, run directly in layers?

**Question:** can useful specified algorithms run directly without a general-purpose CPU?

**Prerequisite:** E00: memory for each stateful backend used. A backend used here only for state tracking, not as addressed memory, needs the retention, idle, and snapshot checks from E00: memory for its declared state, not RAM conformance.

**Targets:** one-hot finite transducers, parity and modular counters, mutable-record assignment, permutation tracking, and one bounded stack parser. Test simple cyclic groups before larger permutation tasks; do not assume theory implies a particular low-precision encoding.

**Variants:** ordinary transition code; compiled lookup circuits with explicit feedback; direct signed, reflection, and rotation recurrences where representable (DeltaProduct, Complex KDA, RWKV-7, Mamba-3); and LSTM/GRU cells as compiled finite transducers. Build lookup circuits with piecewise-linear (ReLU/ReGLU) and smooth (SwiGLU) activations as separately labeled constructions. Compare constrained gate ranges with extended ones only as separately labeled architectures.

**Acceptance:** full state/output equivalence, explicit stack overflow and underflow, correct boundary behavior, and identical snapshot/resume. Compare the same input sequence and transition semantics across variants.

**Measurements:** circuit width/depth, nonzero and stored weights, state bytes, ticks, total work, and the smallest margin between a smooth or saturating activation and its intended discrete value. Test longer input sequences within fixed capacity separately from increasing stack depth or number of states.

**Discriminator:** establish whether a backend executes assignment, self-referential updates, or permutations directly, or pays for additional nonlinear feedback steps. Count a transition as supported only when it executes exactly at the tested precision and horizon, not because the architecture can represent it in theory.

**Results:** [results/E01](results/E01/README.md). ReLU lookup circuits for parity, a mod-5 counter, and S3 tracking, and a delta-rule stack, matched their oracles exactly up to 10^6 actions on CPU; the other variants are untested.

### E02 - Learned interface comparison

> **TL;DR:** Does a model do better with an internal machine than with a tool, and why?

**Question:** does an internal machine interface help a learned controller, and does compiling the machine add anything beyond native integration?

**Prerequisites:** E00: memory and E01: streaming machines. This is the first learned-controller study and is not a prerequisite for executor research.

Compare:

| Arm | Interface | Executor |
|---|---|---|
| A | No machine | Model only |
| B | Textual tool requests and replies | Conventional in-process implementation |
| C | Typed internal observations and actions | Conventional in-process implementation |
| D | Same internal interface as C | Conforming compiled tensor implementation |

Use the same small memory or bounded stack in B-D. Do not penalize B with unnecessary subprocess or network overhead. Separate any optional remote-tool comparison.

**Tasks:** nested scopes or record updates; supervised action traces first, then free-running control. Separate in-distribution evaluation, longer sequences at fixed capacity, unseen operation compositions, and explicit capacity changes.

**Validity constraints:** action masking or another mechanism that restricts learned actions to valid machine states is a separately labeled ablation. Apply it identically across B-D, using constrained decoding over the same action set for B. Report its effect on invalid-action rate, task accuracy, and training cost.

**Acceptance:** report executor equivalence independently from task and action accuracy. Invalid-action rate, recovery, halting, and budget exhaustion are outcomes, not examples to discard. Multiple training seeds and paired held-out task instances are required before comparative conclusions.

**Measurements:** task accuracy, action accuracy, executor mismatches, trainable parameters, training examples/steps, generated tokens, internal actions, machine work, wall time, and peak memory.

**Discriminator:** C versus B tests the interface; D versus C tests the implementation. A positive D-versus-A result alone does not identify either cause. A native-interface win is still useful.

**Results:** [results/E02](results/E02/README.md). With a frozen SmolLM2-135M and trained heads, C beat B on every split and seed, clearly beyond the training length and on held-out pairs; D matched C exactly. All arms degraded on long programs.

### E03 - Latent ticks and explicit events

> **TL;DR:** Compare non-text feedback with token feedback and a native loop at matched information and work budgets.

**Question:** does an explicit non-text feedback path improve cost or control without changing the available information or permitted computation?

**Prerequisites:** E01: streaming machines and E02: learned interfaces, including the native internal baseline.

**Variants:** token feedback, repeated latent transitions, and a conventional in-process loop over the same machine. Declare where late-computed control reenters the next tick, whether learned context is frozen, and which recurrent caches update.

**Acceptance:** bounded stepping, deterministic events, precise faults, and snapshot/resume during multi-tick actions. Hiding trace tokens without a new feedback path is not a latent-execution implementation.

**Measurements:** embedding/unembedding cost, controller and executor time, context/cache growth, internal state, ticks, and total logical work. Match observation and work budgets rather than declaring non-text computation free.

**Discriminator:** improvements must survive comparison with an efficient native loop and cannot come solely from providing hidden privileged state.

**Results:** [results/E03](results/E03/README.md). On copy programs that need a read then a write, latent feedback (one trained adaptor vector per tick through a frozen SmolLM2-135M) beat token feedback, head-only feedback, and a native loop by 0.08 to 0.49 task accuracy. A steering control with the same trained positions but no feedback in them matched it up to 32 statements. Latent feedback led only at 64 statements (+0.05), so the gain is mostly trained steering at unmatched model work. Executors had zero mismatches, and mid-copy snapshots resumed identically.

#### Amendment 2026-11-25: reply-only vectors, pre-registered

> **TL;DR:** A control the first run lacked: input vectors that carry the value a read returned, without the action that produced it.

**Question:** the first run's L vector carries the whole feedback code (operation, record, value, and event kind), and S carries none of it. Which part of the code do later ticks need? An arm V that carries only the reply separates feedback content from the trained positions at every length, and from the action echo.

**Design:** the same runner, task, model revision, seeds, splits, and hyperparameters as the first run, with one more arm:

- V, reply-only vectors: L's adaptor, but the operation, record, and value-written fields of the code are replaced by constants, so the vector depends only on the value the machine returned and the event kind.

V and S are run in one new job, on the same programs and seeds as the recorded run; S is rerun so that V is compared with an S from the same code. The recorded L, T, H, and N results are not rerun. If the rerun S differs from the recorded S by more than the recorded seed range, the report says so and does not compare V against L.

**Predictions:**

1. V's task accuracy exceeds S's by more than the conservative two-standard-error bound at 64 statements, in the mean over seeds. The reply content matters at the length where L led S.
2. V is within the bound of L at 64 statements. The action echo adds nothing beyond the reply.

**Outcomes that count against the reading:** prediction 1 fails: reply content does not help at 64 statements, so L's lead there is not explained by the value fed back. Prediction 2 fails with V behind L: the action echo carries information the reply does not.

**Cost:** two arms of the first run's five; cap 2 hours, $1.60.

**Results:** [results/E03](results/E03/README.md#e03-latent-feedback-reply-only-vectors). Run 2026-11-25 on an `l4x1` job, three seeds, with S rerun in the same job; the rerun S matched the recorded S exactly. Arm V, whose latent vector carries only the value a read returned, matched S at 64 statements (0.566 against 0.566, difference 0.000, bound 0.032), trailed it by 0.041 at 32 statements (bound 0.046), and was 0.049 below the recorded L at 64 statements (0.566 against 0.615, bound 0.032). Prediction 1 (V beats S at 64) was not met and prediction 2 (V within the bound of L at 64) was not met. The reply alone does not carry L's lead over S at 64 statements. Whether the action fields carry it, or the lead is within seed variation, is untested. The job cost about $1.32.

### E04 - Dataflow, specialization, and conditional units

> **TL;DR:** Is computing a whole pipeline at once better than one instruction at a time?

**Question:** is direct scheduling a better fit than scalar interpretation?

**Prerequisites:** E00: memory and E01: streaming machines. Replayed workloads need no learned controller.

**Targets:** an arithmetic expression DAG, a bounded integer dot product, and a small gather/map/reduce/scatter workload.

**Variants:** conventional algorithm; scalar-machine implementation; direct compiled graph; optional vector units, sparse expert dispatch, and program-specialized weights. Keep specialization and dispatch as separate ablations.

**Acceptance:** identical arithmetic domain, overflow rules, ties, outputs, and fault/work semantics. No expert may drop a required operation. Fused blocks need precise exits or explicit supported safepoints.

**Measurements:** construction and compilation time, checkpoint bytes including padding, runtime state, dispatch cost, work units, throughput, latency, and the invocation count needed to amortize specialization.

**Discriminator:** compare against optimized conventional arithmetic, not just a deliberately slow interpreter. QwenPU and Percepta specialization are precedents, not results from this repository.

**Results:** [results/E04](results/E04/README.md). In float32 and float64, level-by-level tensor graphs, program-specialized weights, and capacity-limited expert dispatch matched the `dataflow/v1` oracle exactly on CPU, faults included. Batched, they were no faster than the same schedule in int64 tensors, and slower than Python for single inputs. The bfloat16 and float16 probes mismatched with zero residue.

### E05 - Graph, semiring, and logic execution

> **TL;DR:** Can exact graph and logic engines replace step-by-step instructions?

**Question:** can a small exact relational/message-passing machine replace scalar instruction dispatch for suitable workloads?

**Prerequisites:** E00: memory plus a declared bounded Boolean or integer arithmetic contract.

**Targets:** Boolean reachability first; then bounded min-plus shortest paths; finally a small positive finite-domain Datalog program.

**Variants:** conventional sparse algorithms, direct tensor/sparse reductions, and scalar-machine execution where available. Keep learned GNNs or embedding-space reasoning separate from exact kernels.

**Acceptance:** all relation entries or distances match the oracle, including disconnected vertices and ties. Define infinity sentinels, negative-cycle policy if applicable, fixed-point stopping, and exhaustion limits.

**Measurements:** graph density, edges processed, iterations, tensor/index bytes, latency, and actual versus padded work.

**Discriminator:** learned approximate answers cannot be counted as exact entailment; dense tensor speedups must be compared at realistic sparse sizes.

**Results:** [results/E05](results/E05/README.md). Dense Boolean and min-plus tensor rounds matched reachability, shortest-path, and Datalog oracles round for round on CPU in float32, float64, and float16, and the Boolean machines also in bfloat16. bfloat16 shortest paths failed once distances passed 256. On sparse graphs, Python BFS and Dijkstra were 20 to 100 times faster than dense tensor rounds; scalar-machine execution is untested.

### E06 - Spatial and structured dynamical machines

> **TL;DR:** Compare local-rule and filter implementations with random-access emulation under their respective exact or approximate contracts.

**Question:** when does local or structured state evolution provide a better substrate than random-access emulation?

**Prerequisites:** an integer local-rule oracle for the exact branch, or a declared numerical error contract for the approximate branch.

**Exact branch:** a compiled cellular rule with specified boundaries and simultaneous updates. Compare conventional code, convolution plus lookup, and scalar-machine execution where available. Use double buffering or a proved equivalent schedule.

**Numerical branch:** a specified finite filter or linear dynamical system using a suitable SSM. Compare against conventional numerical code at matched precision and error tolerance; do not pool it with exact cellular results.

**Acceptance:** cell-for-cell equality for exact rules; predeclared horizon/error limits for numerical simulation. Fixed grids and finite filter state must be reported as bounded resources.

**Measurements:** cells or samples per second, generations, state bytes, communication, and boundary-update costs.

**Discriminator:** test the layer's natural operation, not just its ability to imitate RAM through extra feedback.

**Results:** [results/E06](results/E06/README.md). A 3×3 convolution plus a ReLU lookup matched three cellular rules cell for cell in float32, float64, bfloat16, and float16 over 720,000 CPU actions. Dense and diagonal-SSM filters met their declared tolerances in float64 and float32, while 17 of 18 bfloat16 variants exceeded theirs. Scalar-machine execution is untested.

### E07 - Optimization-defined state updates

> **TL;DR:** Which learning-style memory updates still behave like exact memory?

**Question:** which optimizer settings preserve useful specified memory or adaptation semantics?

**Prerequisite:** E00: memory reference traces; an independently specified optimizer for non-RAM variants.

**Variants:** sequential quadratic-loss updates, chunk-start/stale gradients, momentum, decay, normalization, and then nonlinear memory. Change one feature at a time.

**Workloads:** repeated writes to one address inside a chunk, alternating keys, idle retention, and a small online-regression task.

**Acceptance:** the sequential one-hot quadratic case must match replacement semantics when configured accordingly. Other optimizers must match their own update reference; a deviation from RAM is a finding, not necessarily an optimizer bug.

**Measurements:** state including optimizer buffers, update count, read/write interference, task loss, and work per sample.

**Discriminator:** distinguish an exact memory primitive from an adaptive approximate learner.

**Results:** [results/E07](results/E07/README.md). On CPU, sequential SGD with one-hot keys and learning rate 1 matched `memory/v1` exactly in float32 and float64. Each single change (learning rate, stale chunk gradients, momentum, decay, gradient normalization, an MLP) departed from RAM, while the tensor backend matched its own float64 reference at every step. bfloat16 failed that reference check, and its decay factor rounded to 1.

### E08 - Heuristic search with an exact verifier

> **TL;DR:** Do neural guesses plus an exact checker find more valid answers for the same budget?

**Question:** can neural or energy-based candidate generation improve verified solution yield at a fixed budget?

**Prerequisite:** an independently tested verifier and conventional search baseline; recurrent backends also need the relevant checks from E00: memory.

**Targets:** one small, explicitly specified constraint problem. Compare conventional search, heuristic proposals plus native verification, and the same proposals plus compiled verification.

**Acceptance:** no invalid candidate is accepted on the verifier conformance suite. Report rejection, timeout, and unknown separately from proved unsatisfiable; search failure is not an unsatisfiability proof.

**Measurements:** verified solutions per budget, proposal quality, verifier mismatches, search work, and total latency.

**Discriminator:** separate better proposals from faster verification, and do not infer efficient global optimization from associative-memory convergence.

**Results:** [results/E08](results/E08/README.md). A compiled CNF verifier matched its oracle on 60,000 candidates within its declared limits and accepted no invalid assignment; past bfloat16's limit, a folded lowering rejected 119 valid ones. At a fixed step budget, energy-based proposals solved 23 of 28 known-satisfiable random 3-SAT instances, against 0 for random proposals and 28 for WalkSAT. Only DPLL proved unsatisfiability.

### E09 - State in the machine, not the context

> **TL;DR:** Pre-registered: a controller that sees one statement at a time, with the machine holding all state, should beat full-context controllers on long programs and fall only as fast as its own errors compound.

**Status:** pre-registered 2026-11-25, before any code or run; first run 2026-11-25, held-out follow-up 2026-11-25. Changes after those runs go in a dated amendment below this section, not in an edit of it.

**Question:** E02 found that every controller loses accuracy on programs longer than its training programs, and that the executors are exact, so the loss is in the chosen actions. Is that a position effect of reading a long context, or the compounding of ordinary per-statement errors?

**Prerequisite:** E02's task, frozen SmolLM2-135M, heads, and test-program generator.

**Arms:** all arms use E02's task, model, head architecture, training data, three training seeds, and the same paired test programs. Splits are E02's 2-8, 32, and 64 statements plus a new 256-statement split. B and C are rerun in the same job, so every arm sees the new split.

| Arm | Context the model reads | Machine |
|---|---|---|
| B | Full program, with text requests and replies (E02 B) | Integer oracle |
| C | Full program (E02 C) | Integer oracle, typed interface |
| S | One statement per forward pass, fresh window each time | Integer oracle, typed interface; the machine is the only carrier of state |
| W | One statement per forward pass | None; the answer head sees only the current question |

W is a floor, not a competitor: with no memory it cannot answer about earlier writes. S trains on features computed from single-statement windows. If S works, repeat with the delta-rule executor as a conformance check, as E02 did for D.

**Compounding baseline** (fixed before the run, computed from the oracle alone): let ε be S's action error rate on the 2-8 split for a seed, and a its task accuracy on questions whose actions were all correct on that split. Replay each test program through the oracle machine, replacing each statement's action with probability ε by a uniformly random write target and value, or read target. The baseline accuracy at a length is a times the fraction of questions the replay answers correctly.

**Predictions:**

1. S's task accuracy at 64 and 256 statements exceeds C's by more than the conservative two-standard-error bound used in the E02 report.
2. S's task accuracy at every length is within 0.05 of the compounding baseline, in the mean over seeds.
3. W's task accuracy stays at or below 0.15 on every split, so the machine, not the window, carries the information.

**Sanity checks, not predictions:** S's action accuracy matches its 2-8 value at every length, since its inputs do not depend on position; a difference points to a bug. Executor mismatches are zero.

**Outcomes that count against the thesis:**

- Prediction 1 fails: S does not beat C beyond the bound at 64 statements. Then the context position is not what limits C, and the controller's own errors are the cause.
- Prediction 2 fails by falling more than 0.05 below the baseline: S's errors are correlated or state-dependent, which per-statement independence does not capture.
- Prediction 3 fails: the answer head recovers state from the question alone, so the task leaks and the comparison is not valid.

**Measurements:** task, action, and answer accuracy per arm and split; the compounding baseline per seed and length; maximum and total model positions per forward pass and per program; executor mismatches; GPU seconds and cost.

**Discriminator:** C versus S at the same task and heads isolates context handling from the machine. The baseline separates an effect of the window from errors that would compound in any controller. An S win says nothing about E03's question of whether feedback should re-enter the model.

**Cost:** E02's one job cost about $0.73 on an `l4x1`; expect the same order. Stay under the $5 limit in [AGENTS.md](AGENTS.md).

**Results:** [results/E09](results/E09/README.md). Run 2026-11-25 on an `l4x1` job, three seeds. S, reading one statement per forward pass, answered every question correctly at 2-8, 32, 64, and 256 statements, with zero executor mismatches for the delta-rule executor, while C and B fell to 0.134 and 0.111 at 256 statements. Prediction 1 held (S minus C +0.71 at 64, +0.87 at 256). Prediction 2 held vacuously, since S made no errors and the baseline is 1.000. Prediction 3 failed as written: W scored 0.501 on 2-8 statements and 0.205 on 32, equal to the share of questions about never-written records, so the 0.15 ceiling was set too low and the task does not leak. The held-out-pairs follow-up (see the amendment below) is also reported there.

#### Amendment 2026-11-25: first run

> **TL;DR:** Implementation choices made before the run that the pre-registration left open or stated differently.

Recorded after the run, with the code that made each choice committed before it (`c7d252a`):

- **Window:** S's and W's window is the preamble plus the statement, not the statement alone.
- **`a`:** computed as the answer head's accuracy given the gold observation on the 2-8 questions, rather than task accuracy on questions whose actions were all correct. With S's actions all correct the two coincide.
- **Delta-rule arm:** S+delta ran unconditionally instead of after S succeeded.
- **Splits:** the held-out-pairs split was dropped, and the 256-statement split has 100 programs (12,935 questions).
- **Prediction 3:** a ceiling of 0.15 ignored that a controller with no memory answers 0 for every unwritten record, and half the questions on 2-8 statement programs are of that kind. The result stands as not met; the reason is in the [report](results/E09/README.md).

#### Amendment 2026-11-25: held-out pairs, pre-registered

> **TL;DR:** A follow-up run on the eight statements never seen in training, to separate reading statements from looking them up; predictions fixed before the code or run.

**Question:** the first run's S saw all 128 statement texts in training, and a random 2-layer model reached 0.94 with the same pipeline. Does S read a statement it has not seen, and does pretraining contribute?

```mermaid
flowchart LR
    U["unseen statement<br/>Set g to 12."] --> W["window: preamble + statement"]
    W --> F["frozen model<br/>SmolLM2 or random"]
    F --> H["trained heads"]
    H --> A["action: write g 12?"]
```

**Design:** the same runner with `--held-out`. Training is unchanged: 2,000 programs that never contain the pairs (record g or h with value 12-15). Test splits are the 2-8 split and E02's held-out-pairs split (300 programs, each containing at least one such pair), with all five arms. Two extra measurements need no program: the action accuracy of S's heads on each of the 8 unseen statement texts, and on the 128 seen ones, from the window feature of the text alone. Runs: SmolLM2-135M on an `l4x1` job and the random 2-layer `tiny-random` model on CPU, three seeds each.

**Predictions:**

1. SmolLM2's S chooses the correct action for at least 0.90 of the 8 unseen statements, in the mean over seeds.
2. On the held-out-pairs split, S's task accuracy beats C's by more than the conservative two-standard-error bound, for SmolLM2.
3. SmolLM2's S beats the random model's S by more than 0.10 in unseen-statement action accuracy, in the mean over seeds.

**Sanity checks, not predictions:** S's action accuracy on the 128 seen statements is at least 0.99 for SmolLM2; executor mismatches are zero.

**Outcomes that count against the reading thesis:**

- Prediction 1 fails below 0.50: S memorizes strings, and the first run's result holds only for statements seen in training.
- Prediction 3 fails: pretraining adds nothing measurable to reading unseen statements here.
- Eight statements per seed is a small set; a result between 0.50 and 0.90 on prediction 1 is reported as inconclusive, not as support.

**Cost:** SmolLM2 evaluation is on 2-8 statement programs only, so the job should take well under an hour on an `l4x1`, under $1.

**Results:** [results/E09](results/E09/README.md#results-unseen-statements). Run 2026-11-25. S chose the right action for 7 of the 8 unseen statements in every seed (0.875) and scored 0.982 on the held-out-pairs split against 0.827 for C. Prediction 1 (at least 0.90) was not met and is inconclusive by its own rule, prediction 2 held (+0.155, bound 0.058), and prediction 3 held (0.875 against 0.250 for the random model). The job cost about $0.22.

#### Amendment 2026-11-25: more unseen statements, pre-registered

> **TL;DR:** The first follow-up tested one set of 8 unseen statements. This one tests 32, in four sets, and records which statements fail.

**Question:** the held-out result rests on 8 statements. Does S read unseen statements in general, or only that set?

**Design:** the held-out run repeated for four disjoint sets of unseen `Set` statements, one record pair times one value range each, together covering all eight records and all 16 values: records g and h with values 12-15 (the first follow-up's set); records a and b with 0-3; records c and d with 4-7; records e and f with 8-11. For each set, S trains on programs that never contain those statements and is tested on the 2-8 split and on a held-out-pairs split built from that set. The action accuracy of S's heads on each unseen statement text and each seen one is recorded per seed, with the statements that fail. Runs: SmolLM2-135M on an `l4x1` job, and the random 2-layer model on CPU, three seeds each.

**Predictions:**

1. SmolLM2's S chooses the correct action for at least 0.90 of the 32 unseen statements, in the mean over sets and seeds.
2. No single set falls below 0.50 in the mean over seeds.
3. SmolLM2's S beats the random model's S by more than 0.10 in unseen-statement action accuracy, in the mean over sets and seeds.

**Sanity checks, not predictions:** S's action accuracy on seen statements is at least 0.99 in every set.

**Outcomes that count against the reading thesis:** prediction 2 fails: some kind of statement is memorized, not read. A mean between 0.50 and 0.90 on prediction 1 is inconclusive. Prediction 3 failing means pretraining adds nothing measurable here.

**Cost:** four runs of the first follow-up's job (about $0.22 each); cap 45 minutes, $0.60.

**Results:** [results/E09](results/E09/README.md). Run 2026-11-25 on an `l4x1` job, three seeds, four disjoint sets of 8 unseen statements that together cover every record and value once. S chose the correct action for 0.833 of the 32 unseen statements (set means 0.875, 0.833, 0.750, 0.875) and for 1.000 of the seen ones. Prediction 1 (at least 0.90) was not met and is inconclusive by its own rule. Prediction 2 held (lowest set 0.750). Prediction 3 held (0.833 against 0.417 for the random model, which ranged from 0.125 to 0.750 by set). Seven statements failed in at least one seed and three in every seed. Every failure wrote a wrong value to the right record. Executor mismatches were zero. The job cost about $0.04.

### E10 - A machine layer inside the model

> **TL;DR:** Pre-registered: a delta-rule memory inside a layer of the frozen model, written and read by trained projections of the residual stream, should match an outside machine on copy programs while using one model pass per statement instead of two.

**Status:** pre-registered 2026-11-25, before any code or run. Changes after a run go in a dated amendment below this section, not in an edit of it.

**Question:** E02: learned interfaces and E03: latent feedback keep the machine outside the model and reach it through heads. Does a machine that lives in a layer, written and read by projections of the residual stream, match or beat the same machine reached through heads, and can that layer reject commands that are not valid?

**Why copy programs:** E09: state in the machine showed that a controller reading one statement at a time answers E02's record programs perfectly, so those programs cannot separate the arms. `Copy x to y.` ([task](src/latent_machines/tasks/copies.py)) needs a read whose value feeds a write. An outside controller does this in two ticks with feedback to its heads; a layer can do it in one pass. This also tests statements that depend on the machine's state, which E09 did not.

**Setup:** frozen SmolLM2-135M, copy programs from E03: latent feedback, and E09's windows: the model reads the preamble plus one statement per forward pass, so the machine is the only carrier of state. Splits: 2-8, 32, 64, and 256 statements, and E02's held-out-pairs split. Three training seeds, 2,000 training programs, 30 epochs, as in E02. Every arm drives the delta-rule backend with a shadow integer oracle.

```mermaid
flowchart LR
    subgraph O["O: machine outside"]
    O1["window"] --> O2["frozen model, all layers"] --> O3["heads"] -->|"read, then write"| O4["memory"]
    O4 -->|"value to heads"| O3
    end
    subgraph M["M: machine in layer k"]
    M1["window"] --> M2["layers 1..k"] --> M3["projections + memory<br/>copy in one pass"] --> M4["layers k+1..end"] --> M5["answer head"]
    M3 -->|"read added to residual"| M4
    end
```

**Arms:**

| Arm | Machine | How a read reaches the answer | Model passes per copy |
|---|---|---|---|
| O | Outside; heads read the last layer and issue ticks (E03's H on windows) | Typed value to the heads | 2 |
| M | Inside layer k = 15 of 30; trained projections of the residual stream give a macro (`set`, `get`, `copy`), the records, and the value; addressing is a straight-through argmax so it stays one-hot; the state is an 8×16 one-hot matrix updated by delta replacement | The read is projected to model width and added to the residual stream, and the answer head reads the last layer | 1 |
| Mh | As M, but the read goes to the answer head as an embedded value and does not enter the residual stream | Typed value to the head | 1 |

Trainable parameters are matched within 5% across arms by sizing the hidden layers of the heads and projections; the counts are recorded. M and Mh train end to end through the frozen layers above k. Supervision is gold macros and gold answers for every arm, so E11 stays the only experiment about learning from outcomes alone.

**Validity test:** a fourth run of M, `M-reject`, adds a `reject` macro. Training adds invalid statements: records `i` to `k` (`Set i to 3.`, `What is j?`, `Copy a to k.`) and values 16-31 (`Set a to 20.`), each labelled `reject`. Test statements use the unseen record `l` and values 32-47. An outside host check rejects every one of them by construction. The question is whether the layer learns to reject and whether that carries to invalid forms it has not seen.

**Predictions:**

1. M's task accuracy is at least O's minus 0.02, in the mean over seeds, on every length split (2-8, 32, 64, 256).
2. M and Mh differ by no more than the conservative two-standard-error bound used in E02 on every split. Re-entry through the residual stream does not beat typed feedback; E03 found the same for feedback through the input.
3. M's in-layer state, decoded after every statement, equals the oracle's state fed the same chosen actions, with zero mismatches over all splits.
4. `M-reject` rejects at least 0.95 of invalid statements of the trained forms and falsely rejects at most 0.01 of valid ones.
5. `M-reject` rejects less than 0.95 of the invalid statements with the unseen record and values, so learned validation does not replace the host check.

**Sanity checks, not predictions:** O's accuracy on the 2-8 split is at least 0.95, as H on copy programs is easy once windows remove the length effect; executor mismatches are zero for O and Mh.

**Outcomes that count against the thesis:**

- Prediction 1 fails: a machine in the layer is worse than the outside machine on the same budget, and the one-pass saving costs accuracy.
- Prediction 2 fails with M ahead: the residual path adds something typed feedback does not, and a control that holds the injection constant would be the next step. M behind Mh: re-entry hurts.
- Prediction 3 fails: the in-layer state does not implement the machine, whatever the accuracy.
- Prediction 5 fails with a rejection rate of 0.95 or more: learned validation generalizes on this test, and the host check could be replaced here.

**Measurements:** task, action, and answer accuracy per arm and split; model positions per program and per statement; trainable parameters; in-layer state mismatches; rejection and false-rejection rates; GPU seconds and cost.

**Discriminator:** O versus Mh versus M separates where the machine lives (outside or inside the model) from how the read returns (typed or through the residual). The result does not show a gain from in-layer memory unless M beats O and Mh beyond the bound.

**Cost:** one `l4x1` job; the upper layers run for every question, so expect about an hour. Cap: 90 minutes, $1.20.

**Results:** [results/E10](results/E10/README.md). Run 2026-11-25 on an `l4x1` job, three seeds. A delta-rule memory inside layer 15 of a frozen SmolLM2, written and read by trained projections of the residual stream, matched an integer oracle after every statement and answered every copy-program question correctly on 2-8, 32, 64, and 256 statements and on held-out pairs. The same memory with the read going to the answer head did the same. The outside machine matched it on every length split and trailed by 0.026 on held-out pairs, within the bound. All five predictions held, but the accuracy tests were at ceiling, so this shows parity, not a benefit of the in-layer memory. A reject macro rejected every trained invalid form and 0.70 of unseen ones, so learned validation does not replace the host check. The job cost about $0.17.

#### Amendment 2026-11-25: first run

> **TL;DR:** Where the first run differed from the text above.

- **Model passes:** the arm table says O makes two model passes per copy. O reuses the statement's feature for both ticks, as E03's arm H does, so model positions are equal across arms. The saving is in controller decisions: O makes three per copy (read, write, end) and M makes one.
- **Prediction 2** is evaluated on all five splits, including held-out pairs.
- **Prediction 5** is evaluated on the pooled unseen texts, with the two kinds (record `l`, values 32-47) reported separately.

### E11 - Learning to use the machine from outcomes

> **TL;DR:** Pre-registered: train heads from answer correctness alone, with gradients passing through a differentiable copy of the machine, and compare with imitation and with reinforcement learning over the real machine.

**Status:** pre-registered 2026-11-25, before any code or run. Changes after a run go in a dated amendment below this section, not in an edit of it.

**Question:** E02: learned interfaces and E03: latent feedback train heads on gold actions, so the controller imitates a known procedure. Can a controller learn to write and read when the only training signal is whether its final answers are right? And does it need gradients through the machine, or does reinforcement learning over the real machine do as well?

**Setup:** E02's record programs, frozen SmolLM2-135M, E09's windows so that the machine is the only carrier of state, E02's heads, three seeds, 2,000 training programs, 30 epochs, Adam at 1e-3. Test splits: 2-8, 32, and 64 statements and the held-out-pairs split. The features of the 136 distinct statement windows are computed once, so training runs on a CPU.

```mermaid
flowchart LR
    S["statement window<br/>feature"] --> H["action head"]
    H -->|"I: gold action loss"| X1(("imitation"))
    H -->|"O: straight-through one-hots"| D["differentiable<br/>8×16 memory"]
    D --> A["answer head"] -->|"answer loss at questions"| H
    H -->|"R: sampled action"| M["integer oracle<br/>(external tool)"]
    M -->|"value"| A2["answer head"] -->|"reward 1 if right"| H
```

**Arms:**

| Arm | Training signal | Machine during training |
|---|---|---|
| I | Cross-entropy on the gold action of every statement; the answer head trains on the gold value read | Integer oracle (this is E09's S) |
| O | Cross-entropy on the answers at questions only; no gold actions and no gold values | Differentiable memory: Gumbel-softmax straight-through one-hots (temperature 1) for the operation, record, and value; state is an 8×16 matrix of one-hot values, initially value 0; a write sets `M ← M + w·y_record ⊗ (y_value − M)`; a read outputs `y_record · M`, and a write outputs the one-hot of 0, as the oracle's typed observation does |
| R | REINFORCE: reward 1 for a correct answer, the batch mean as baseline; the answer head trains with cross-entropy on its own answers | Integer oracle driven by sampled actions, as an external tool |
| W | None for actions; the answer head sees only the question window | None (E09's W) |

Only the answer at each question is supervised in O and R. The loss knows which statements are questions, since only they have answers; it never sees which action the statement needs. After training, all arms run free-running on the delta-rule backend with a shadow oracle, with greedy actions.

A consistent relabeling of records or values works as well as the gold one, because the machine stores whatever code it is given and the answer head decodes it. O and R can therefore score well with low agreement to gold records and values. The operation is fixed by the machine, so it cannot be relabeled.

**Predictions:**

1. O's task accuracy at 2-8 statements is at least 0.90, in the mean over seeds.
2. O's operation accuracy is at least 0.95: it writes at set statements and reads at questions.
3. O's task accuracy at 2-8 statements exceeds R's by more than the conservative two-standard-error bound used in E02.

**Reported, not predicted:** agreement of O's and R's record and value choices with gold; task accuracy at 32 and 64 statements and on the held-out pairs; training curves.

**Sanity checks, not predictions:** I reaches at least 0.99 task accuracy at 2-8, as E09's S did; W stays at or below 0.55; executor mismatches are zero.

**Outcomes that count against the thesis:**

- Prediction 1 fails below 0.50: this recipe does not discover how to use the machine from outcomes in this setting. That is a result about discoverability, not about whether it is possible.
- Prediction 2 fails: the answer head is solving the task without the machine, and the arm is not using it.
- Prediction 3 fails: REINFORCE over the real machine does as well, so gradients through the machine are not needed, and an external tool can be trained from outcomes too.

**Measurements:** task, action, operation, and answer accuracy per arm and split; agreement with gold records and values; the training steps to reach 0.90 where reached; executor mismatches.

**Discriminator:** O against W and R. Beating W shows the machine is used; beating R shows the benefit comes from gradients through it rather than from outcome-only training.

**Cost:** local CPU, no GPU job. The SmolLM2 weights are in the local Hugging Face cache, so nothing is downloaded.

**Results:** not run.

## Research coverage

> **TL;DR:** Every idea in RESEARCH.md points to the experiment that tests it, or says why not yet.

Every claim, proposal, and open question in [RESEARCH.md](RESEARCH.md) maps to an experiment, a [scope gate](#gates-for-expanding-scope) that defers it, or context only. When RESEARCH.md gains one, add a row here in the same change.

Layer families appear under the experiments that test their natural primitive, not in one shared experiment. Testing every family against RAM conformance alone would score filters, optimizers, and spatial operators against the wrong contract.

| RESEARCH.md item | Coverage |
|---|---|
| Terms, findings, closest systems, foundations, adjacent work | Context, not reproduced here. Specific limits feed experiments: Qwenputer kernel residues → numerical conformance; Percepta indexed lookup → E00; Percepta specialization and QwenPU → E04; Coconut → E03 |
| QwenFX: a compiled CPU driving its own LM | Context; the machine-to-model call stays tool-shaped, the interface E02 arms B-D and E03 compare |
| chipformer: a compiled CHIP-8 transformer | E00 append-only history and Percepta-style indexed lookup; E01 exact ReGLU lookup circuits |
| Weight-Space Programming: LLM-authored weight edits | Context: edits judged by task metrics, not specified transitions |
| Parity tracked in GDN state with beta up to 2 | E01 parity with signed transitions; E03 latent computation, not yet tested on parity |
| Reduction-order device fingerprints | Numerical conformance: supported devices and deterministic options; E00 GPU variants |
| Delta-rule memory and its optimization interpretation | E00 delta replacement; E07 sequential one-hot quadratic acceptance |
| DeltaNet / Gated DeltaNet | E00 reference tensor backend; E01 |
| Kimi Delta Attention | E00, including clearing cost versus GDN |
| RWKV-7 | E00 storage; E01 state tracking |
| DeltaProduct / Complex KDA | E01 signed, reflection, and rotation transitions |
| Mamba-2 / SSD | E00 with and without read-modify-write; E06 numerical branch |
| Mamba-3 | E00 separate SSM variant; E01 state tracking |
| LSTM / GRU | E00 register file; E01 finite transducers |
| Full attention | E00 append-only history; Percepta indexed variant once specified |
| MLP / ReGLU / SwiGLU | E01 piecewise-linear versus smooth lookup circuits; E04 arithmetic |
| Mixture of experts | E04 dispatch ablation |
| Convolution | E06 exact branch |
| TTT / Titans | E07 |
| Frozen MLP "key-value memory" | Context: not writable runtime RAM, so not an E00 variant |
| Transition algebra | E01 |
| Streaming transducers, stacks, and parsers | E01 has initial CPU results for ReLU transducers and a delta-rule stack; other backend comparisons remain proposed |
| Dataflow, vector, and conditional functional units | E04, with initial CPU results |
| Graph, semiring, and finite-domain logic | E05, with initial CPU results |
| Cellular automata, systolic arrays, and signal processing | E06 cellular rules and filters, with initial CPU results; systolic arrays have no separate target yet |
| Optimizers as machines | E07, with initial CPU results |
| Energy-based search with an exact verifier | E08, with initial CPU results |
| Latent execution feedback contract | E03, with initial GPU results; a steering control explains most of the latent arm's gain |
| Compiler target and proof-oriented compilation | Deferred: [scope gates](#gates-for-expanding-scope) 4 (shared compiler) and 5 (exactness guarantee) |
| Numerical and resource limits | Numerical conformance; E00 measurements; correct requests → E02 executor versus action accuracy |
| First controller experiment | E02 |

| Open question | Coverage |
|---|---|
| 1. Typed actions without text serialization | E02 arms C and D, free-running |
| 2. Benefit over external tools | E02 C versus B |
| 3. Compiled versus native implementation | E02 D versus C |
| 4. Constraining learned writes to valid states | E02 validity-constraint ablation |
| 5. Semantics-preserving dtypes, kernels, and devices | E00; numerical conformance |
| 6. Joint checkpoint contents | E03 snapshot during multi-tick actions; run-record snapshot artifacts; E07 shows canonical memory snapshots omit optimizer buffers |
| 7. Transition algebra fit | E00, E01, E06 |
| 8. Alternatives to scalar instruction execution | E01, E04, E05, E06; E04 compares with a scalar machine, E05 and E06 only with conventional algorithms so far |
| 9. Compiler-generated numerical obligations | Deferred: [scope gate](#gates-for-expanding-scope) 5 (exactness guarantee) |
| 10. Indexed lookup, specialization, or feedback versus placement | E00 indexed variant; E04 specialization; E03 feedback; E02 D versus C |

## Run records and results

> **TL;DR:** Record each run's source, configuration, environment, outcomes, and failure evidence.

Each runner must write a unique directory under ignored `runs/` containing:

| File | Required content |
|---|---|
| `run.json` | Schema version, experiment ID, exact command, source revision and dirty-tree fingerprint, resolved configuration, seeds, input/split digests, model/tokenizer/adaptor identifiers and hashes when present |
| `metrics.json` | Declared metrics with units, denominators, run status, failures, budgets, and correctness/approximation classification |
| `trace.jsonl` | Per-transition evidence for bounded validation runs; long sweeps may retain failures plus declared sampling and aggregate checks |
| `failure.json` | When a failure occurs: reproducible input, action index, expected/actual state difference and event, relevant numerical metadata |
| Snapshot artifacts | When tested: versioned machine state, controller state, RNG state, pending work, and backend/environment metadata needed to resume |

Environment metadata includes device, library/runtime versions, kernel implementation, compiler settings, deterministic/reduction options, batch shape, and each relevant dtype. Distinguish cold initialization/compilation from steady-state timing and end-to-end time; synchronize asynchronous device work before measuring. Record actual peak memory separately from logical state size.

Write a terminal status even when a run fails; do not emit success-shaped partial results. A replay verifies the declared observations, not the identity or trustworthiness of whoever ran it.

Commit curated summaries and small manifests to `results/`, including failed hypotheses and links to larger artifacts where appropriate. Keep raw traces, downloaded models, generated weights, caches, and local checkpoints out of Git. Do not add a hosted tracking service until there is a concrete need.

Each curated report should state the hypothesis, evidence class, matched controls, source/configuration identifiers, correctness outcomes, uncertainty across task and training seeds where relevant, costs, and the decision made. Report first-mismatch horizons and capacity limits, not just average task accuracy.

## Gates for expanding scope

> **TL;DR:** Require specific evidence before adding learned control, performance claims, model families, compilers, guarantees, or runtime integration.

1. **Before learned control:** the executor passes bounded conformance and snapshot tests.
2. **Before speed claims:** correctness passes on the exact timed path, and the conventional baseline and budgets are credible.
3. **Before adding another model family:** a concrete capability gap or hypothesis justifies it.
4. **Before a shared compiler:** multiple working machines/backends expose a useful common lowering. Preserve an independent oracle.
5. **Before an exactness guarantee:** distinguish finite testing from proof; define the checker, floating-point assumptions, and trusted runtime. Compiler-generated range/representation obligations are a later research target, not certificates that already exist.
6. **Before Shem integration:** identify a workload that benefits. Adapt execution/snapshot APIs without forcing graph or spatial state into RV32 registers, or expanding this repository into another container runtime.

The executor milestone requires a conforming memory backend, an exact streaming machine, and reproducible evidence. E02: learned interfaces then tests interface benefits separately from implementation benefits, whether the result is positive or negative.
