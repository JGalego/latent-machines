# Research

> **TL;DR:** The reviewed systems show compiled execution or learned memory. None establishes the proposed interface comparison.

Reviewed: 2026-11-17. Primary-source literature and implementation review; experiments have not been reproduced here. Benchmark results below are attributed reports, not independent validation.

## Terms

> **TL;DR:** Exact execution, learned control, persistence, weight-space programming, and internal interfaces describe different properties.

- **Exact execution:** transitions match a specified machine, within an explicit representation, precision, and resource budget. Deterministic output alone is insufficient.
- **Learned control:** a network chooses operations, arguments, or addresses. Exact execution does not make those choices correct.
- **Persistent state:** memory survives successive computation steps. Saving a session is a separate capability.
- **Weight-space programming:** programs can be compiled into fixed weights. This differs from using changing fast weights as learned memory; neither should be confused with ordinary fine-tuning. [Takehi et al.](https://rikiyat.github.io/blog/weight-space-programming/) use the same name for LLMs editing weights directly; here it means compiled programs.
- **Internal interface:** machine results influence model computation without requiring a textual request and response. Being inside the checkpoint does not necessarily provide this interface.

## Findings

> **TL;DR:** The reviewed sources do not establish a learned internal machine interface evaluated against matched tool and native baselines.

Learned controllers with explicit state predate modern LLMs. Compiled neural programs have also been combined with pretrained language models.

The less-established combination is a pretrained LM with a learned internal interface to an exact, mutable machine, evaluated against both external tools and native in-process implementations. The sources below establish parts of it. None is a mature general solution. The combination is a research hypothesis, not a novelty claim.

## Closest systems

> **TL;DR:** These systems show compiled execution, learned control, or integration with language models. Each has different limits.

| Work | Mechanism | Relevance and limit |
|---|---|---|
| [Qwenputer](https://divagr.com/blog/qwenputer), Divyansh Agrawal, 2026 | Compiled CPU layers added to Qwen3.5; registers and RAM in Gated DeltaNet state. | A pretrained LM and a weight-space machine in one system. Interaction remains tool-shaped, through text and execution tokens. |
| [Percepta transformer-vm](https://www.percepta.ai/blog/constructing-llm-computer), 2026 | Analytically compiled WebAssembly interpreter using hard-max attention and an append-only execution trace. | A standalone weight-space executor. The released construction is not a learned LM controller with mutable recurrent RAM. |
| [Neural Comprehension](https://arxiv.org/abs/2304.01665), Yixuan Weng et al., 2023 / ICLR 2024 | Compiled rule modules run alongside an LM and gate output-token selection. | Direct precedent for an LM using compiled neural computation. No general persistent read/write machine interface. |
| [Tracr-Injection](https://arxiv.org/abs/2505.10719), Tomas Vergara-Browne and Alvaro Soto, 2025 | Trains a GPT-2 residual subspace against a compiled program's activations. | Direct precedent for algorithm representations inside a pretrained LM. Distillation, not exact live execution. |
| [Neural Programmer-Interpreters](https://arxiv.org/abs/1511.06279), Scott Reed and Nando de Freitas, 2015 / 2016 | A recurrent controller selects subprograms and arguments, with scratchpad and pointer operations. | Strong precedent for learned control over explicit machine state. Not a pretrained language model. |
| [Neural Harvard Computer](https://arxiv.org/abs/2105.07957), Daniel Tanneberg et al., Nature Machine Intelligence 2020 | Separates control from data, using hard-address memory and an ALU with hand-designed or learned operations. | Particularly close to learned control over discrete state. Evidence is on algorithmic tasks, not general pretrained LMs. |
| [chipformer](https://github.com/divagr18/chipformer), Divyansh Agrawal, 2026 | A compiled fp64 transformer that runs CHIP-8: each predicted token is one memory write, read back by hard-max attention over the write log. | A complete compiled machine whose traces the author reports as bit-identical to a reference emulator. Memory is an append-only log, as in Percepta, and no language model controls it. |
| [Differentiable Forth](https://arxiv.org/abs/1605.06640), Matko Bosnjak et al., 2016 / ICML 2017 | A neural stack machine executes program sketches with learned slots, including coupling to an LSTM language encoder. | A concrete program/neural hybrid. Continuous execution relaxations must not be presented as an exact discrete interpreter. |

### Qwenputer: internal execution, tool-shaped interaction

> **TL;DR:** Qwenputer adds a compiled CPU to a language model; interaction still uses tool-call text and execution tokens.

The original Qwen language weights are unchanged. Added compiled layers implement the CPU. The LM emits a tool-call format; the circuit receives those tokens and executes the request.

The RV32 driver keeps CPU trace tokens out of Qwen's language layers and caches. This is not a learned latent read/write channel between the LM residual stream and arbitrary machine state. The RV32 variant also needs its driver, not a plain `generate()` call.

The author reports official ISA tests, long free-running executions, and arithmetic results roughly comparable to a Python-tool baseline. A [third-party reproduction report](https://github.com/divagr18/qwenputer/issues/3) identifies dtype- and kernel-sensitive behavior. Neither finite testing nor compilation alone establishes exactness on every backend.

The filesystem variant saves selected state through host-managed files. Cross-session persistence is explicit. Storing the CPU in weights does not provide it.

[QwenFX](https://github.com/divagr18/QwenFX) runs Vercel's fx coding agent, compiled as an RV32 binary, on Qwenputer's CPU, calling the same Qwen3.5-4B for every model step. The author reports 15 of 15 small coding tasks passing hidden tests, six pull requests opened by the agent, and no wrong chip token against a reference machine. Control runs in the opposite direction from this project's question: the compiled machine drives the language model, which it reaches through a tool-shaped call.

Sources: [code](https://github.com/divagr18/qwenputer), [design and results](https://divagr.com/blog/qwenputer), [model](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer), [QwenFX code](https://github.com/divagr18/QwenFX).

### Percepta: a different memory architecture

> **TL;DR:** Percepta represents machine state through attention lookups over an append-only execution history.

Percepta stores machine state as an append-only history read by attention lookups. Qwenputer uses recurrent RAM. The two designs trade off memory growth and execution cost differently.

The team describes a subset of WebAssembly, substantial overhead relative to conventional execution, and integration with learned models as future work. The released executor is not a trained hybrid.

The overview and repository describe a specialized decoding path: two-dimensional hard-max lookup heads permit incremental convex-hull indexing, with reported logarithmic insert/query cost instead of a dense scan over history. This does not establish an equivalent exact acceleration for arbitrary softmax attention. Hard-max ties, finite-temperature softmax, index maintenance, and numerical behavior need separate checks. The append-only representation still grows with the trace.

The repository also implements program specialization, described as a first Futamura projection: bake one program into weights and remove its bytecode prefix. This is a separate experimental axis from replacing the execution backend or changing the controller interface.

Sources: [construction](https://www.percepta.ai/blog/constructing-llm-computer), [overview](https://www.percepta.ai/blog/can-llms-be-computers), [code](https://github.com/Percepta-Core/transformer-vm).

### Neural Comprehension: compiled modules already influence decoding

> **TL;DR:** Neural Comprehension uses compiled rule modules to gate a language model's output-token selection.

Compiled neural networks encode symbolic rules using constructed attention weights. A gate combines their one-hot outputs with the LM's output distribution during generation.

This is a counterexample to any broad claim that combining LMs and compiled neural programs is new. The integration is at the output level, over context. It is not a demonstrated general RAM that the LM reads and updates through latent actions.

The paper reports symbolic-task length generalization and comparisons with tool-based arithmetic. Those results do not isolate every benefit of internal placement from routing, representation, or implementation.

Sources: [paper, section 4.1](https://arxiv.org/html/2304.01665v3#S4.SS1), [code](https://github.com/WENGSYX/Neural-Comprehension).

### Tracr-Injection: algorithm supervision is not exact execution

> **TL;DR:** Teaching a model to imitate a program is not the same as running the program.

A Tracr-compiled model supplies training targets. GPT-2 is trained so a linear projection of its residual stream resembles the compiled model's variables. The compiled model does not execute at inference.

Reported out-of-distribution results improve on some Dyck-language cases but remain weak on several counting and addition cases. The method gives no guarantee that the compiled algorithm's semantics are preserved.

The repository also labels an earlier live compiled-model coupling attempt as failed. That prototype explores learned attention between model and compiled activations. It is not evidence of a working exact read/write interface.

Sources: [paper](https://arxiv.org/abs/2505.10719), [code and prototype](https://github.com/tvergara/tracr-injection). Check upstream licensing before reusing code.

## Foundations

> **TL;DR:** Earlier work establishes program compilation and learned memory under different representation and precision assumptions.

| Work | Contribution | Boundary |
|---|---|---|
| [Tracr](https://arxiv.org/abs/2301.05062), DeepMind, 2023; [code](https://github.com/google-deepmind/tracr) | Compiles RASP sequence programs into transformer weights without gradient descent. | A bounded sequence computation, not persistent decoding state. Categorical and numerical representations require different precision analysis. |
| [ALTA](https://arxiv.org/abs/2410.18077), DeepMind, 2024 / 2025; [code](https://github.com/google-deepmind/alta) | Compiles programs, including looped Universal Transformers, into weights. | Categorical state is one-hot; numerical compilation is approximate. No general saved-session runtime. |
| [Looped Transformers as Programmable Computers](https://arxiv.org/abs/2301.13196), 2023 | Constructs a looped transformer computer with read/write state in its sequence. | Includes approximation and accumulated-error analysis; not an unconditional exact-machine result. |
| [Neural Random-Access Machines](https://arxiv.org/abs/1511.06392), Kurach, Andrychowicz, and Sutskever, 2015 / 2016 | Learns pointer manipulation over variable-sized memory from input/output examples. | Algorithm-learning experiments, with differentiable representations; not a modern pretrained-LM integration. |
| [Neural Turing Machines](https://arxiv.org/abs/1410.5401), 2014, and [Differentiable Neural Computer](https://github.com/google-deepmind/dnc), 2016 | Learned controllers read and write memory every recurrent step. | Soft addressing and learned behavior are not exact discrete-machine semantics. |
| [Differentiable Forth code](https://github.com/uclnlp/d4) | A released implementation of program sketches, stacks, heap, and learned operations. | Historical research code; the repository explicitly says it is not maintained. |

The Neural Harvard Computer is relevant to experiment design. Its controller and memory-access interfaces operate on a control stream separate from data manipulation. It uses hard memory decisions and evolutionary training. Its reported generalization to much larger task configurations is empirical. It does not prove that every learned controller implements the intended algorithm.

## Adjacent work

> **TL;DR:** Related memory and interface ideas that stop short of an exact machine.

These are interface or memory precedents. None establishes an exact persistent machine inside a pretrained LM.

| Work | Relevant mechanism | What it does not establish |
|---|---|---|
| [Fast Weight Programmers](https://arxiv.org/abs/2102.11174), 2021 | Connects linear attention with controllers that program memory using outer-product and delta-rule updates. | Exact addressing and arbitrary discrete computation in an ordinary trained model. |
| [TTT layers](https://arxiv.org/abs/2407.04620), 2024, and [Titans](https://arxiv.org/abs/2501.00663), 2024 / 2025 | Hidden memory is itself a model updated during inference. | A specified RAM/ISA contract; learned memory updates are not equivalent to exact state transitions. |
| [Larimar](https://arxiv.org/abs/2403.11901), IBM, 2024; [code](https://github.com/IBM/larimar) | Reads and writes episodic latent memory that conditions an LM decoder. | Byte-exact mutable memory or an instruction executor. |
| [Differentiable Cache Augmentation](https://arxiv.org/abs/2412.17747), Liu et al., 2024 | A trained coprocessor augments a frozen decoder's KV cache with latent embeddings. | An exact symbolic machine; the augmentation can run offline and asynchronously. |
| [Coconut](https://arxiv.org/abs/2412.06769), 2024 / 2025 | Feeds hidden states back as input embeddings instead of decoding reasoning tokens. | Typed machine actions or exact persistent storage. |
| [MemGen](https://arxiv.org/abs/2509.24704), Zhang, Fu, and Yan, 2025 / ICLR 2026; [code](https://github.com/bingreeky/MemGen) | A learned trigger invokes a weaver that inserts latent memory sequences into reasoning. | An addressable store with specified discrete transitions. The repository documents incomplete releases and evaluation-format corrections. |
| [Recursive SALM](https://arxiv.org/abs/2611.16375), Omer Haq, November 2026 | Persistent learned state adapts a frozen LM through prompts, features, or LoRA. | Exact memory or general algorithmic execution. The initial experiments use small GPT-2 and short synthetic interactions. |
| [ToolkenGPT](https://arxiv.org/abs/2305.11554), 2023 | Learned tool-token embeddings select tools from a frozen LM. | A fully internal interface: arguments are still generated and tools execute externally. |
| [Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/), Takehi, Teoh, Zhang, and Khattab, MIT CSAIL, November 2026 | A frontier LLM reads and edits raw weights: it names what networks were trained on, writes small game-playing networks from scratch, and raises Qwen3.5-0.8B on MATH-500 from 7.9% to 48.1% by editing 0.016% of its weights. | Specified machine semantics: edits are judged by task metrics, not checked transitions. The same approach did not improve Gemma 3 270M on math. |
| [HOLA](https://arxiv.org/abs/2607.02303), Wanyun Cui, 2026 | Supplements delta-rule memory with a bounded cache of retained KV pairs. | A programmable exact machine; preserving vectors is different from exposing arbitrary reads, writes, and instructions. |

## Expanding the programmable substrate

> **TL;DR:** Test which layer families implement the required transitions; a scalar CPU is not the only target.

The hypothesis is that different layer families suit different specified state transitions. Test that fit rather than requiring every backend to emulate a CPU.

Consider these properties independently:

| Ingredient | Examples | Question |
|---|---|---|
| Persistent state | Recurrent matrices, vectors, token histories, spatial grids, fast weights | What can be retained, addressed, replaced, or reset? |
| Computation | MLP circuits, multiplicative gates, sparse experts, convolution, attention | Which operations fit within one transition? |
| Time and feedback | Runtime steps, autoregressive tokens, repeated latent layers | How do computed values affect the next transition? |

An existing demonstration, a theorem under explicit assumptions, and an untested proposed backend are different kinds of evidence. None alone demonstrates that an unchanged pretrained model already implements an exact machine.

### Delta-rule memory and its optimization interpretation

> **TL;DR:** A unit-step quadratic-loss update with a one-hot key replaces one addressed memory slot.

Writing the state as a matrix whose columns are slots:

```text
decayed = alpha * state
next = decayed + beta * (value - decayed @ key) @ key.T
read = next @ query
```

With alpha = 1, beta = 1, and key equal to a basis vector, this replaces one slot. A basis-vector query reads one slot. Transposing the representation gives the row-oriented formulation used in other sources.

The same update follows from a gradient step on:

```text
loss(W; key, value) = 0.5 * ||W @ key - value||^2
W_next = W - eta * (W @ key - value) @ key.T
```

At eta = 1 with a one-hot key, this is addressed replacement. This connects exact delta memory with fast-weight programming and test-time learning, but only in this restricted update regime.

Sources: [Gated Delta Networks](https://arxiv.org/abs/2412.06464), [Fast Weight Programmers](https://arxiv.org/abs/2102.11174), [TTT](https://arxiv.org/abs/2407.04620).

### Candidate layer families

> **TL;DR:** Each candidate needs conformance checks for its proposed operation and execution path.

These are capabilities to test, not a compatibility list. [Research coverage](EXPERIMENTS.md#research-coverage) records where each is tested.

| Family | Natural primitive | Boundary and proposed use |
|---|---|---|
| DeltaNet / Gated DeltaNet | Addressed overwrite and associative read | A demonstrated compiled-memory substrate. Gate endpoints, orthogonal addressing, normalization, and kernels remain part of the contract. |
| [Kimi Delta Attention](https://arxiv.org/abs/2510.26692) | Delta updates with channel-wise retention | Selective retention may reduce clearing or masking work; exact gate endpoints must be verified. |
| [RWKV-7](https://arxiv.org/abs/2503.14456) | Diagonal-plus-rank-one state evolution | More flexible state transformations, with theoretical state-tracking results. Its deployed gates and output normalization constrain a compiled implementation. |
| [DeltaProduct](https://arxiv.org/abs/2502.10297) / [Complex KDA](https://arxiv.org/abs/2609.24797) | Products of reflections, signed gates, rotations | Richer transition algebras. Theoretical representability does not establish exact floating-point execution. Complex KDA releases code and research models. |
| [Mamba-2 / SSD](https://arxiv.org/abs/2405.21060) | Scalar-decay matrix-state recurrence and structured scans | Uniform decay plus an additive write does not directly erase just one slot. Compare explicit read-modify-write against delta replacement; also test native filtering workloads. |
| [Mamba-3](https://arxiv.org/abs/2603.15569) | Complex-valued state evolution and richer state tracking | Do not generalize Mamba-2 limitations to every SSM. Richer tracking still does not by itself establish addressed RAM. |
| LSTM / GRU | Nonlinear recurrent control and gated vector state | Candidate bounded registers or transducers. Dense recurrent feedback and numerical encoding need analysis; this is not a claim about a preserved pretrained LM. |
| Full attention | Selection, aggregation, append-only memory | A latest-write log can emulate mutable state. Dense softmax scans and Percepta-style structured indexed lookup have different costs and semantics. |
| MLP / ReGLU / SwiGLU | Arithmetic circuits, Boolean logic, ROM, transition tables | No persistent memory by itself. Smooth activations require discrete-domain numerical constructions rather than assuming exact threshold behavior. |
| Mixture of experts | Conditional dispatch to compiled functional units | A proposed sparse arithmetic or microcode bank. Soft routing weights, dropped tokens, ties, capacity, and dispatch cost must preserve semantics. |
| Convolution | Local communication, finite delay lines, spatial stencils | Natural for spatial computation; ongoing execution needs recurrence or repeated depth. A constrained deployed SSM is not an arbitrary shift-register implementation. |
| TTT / Titans | Optimization-defined memory updates | Candidate online-regression or adaptation machines. Nonlinear memory, normalization, momentum, and stale chunk gradients may break exact last-writer-wins behavior. |

Supporting sources: [GLU variants](https://arxiv.org/abs/2002.05202), [sparsely gated MoE](https://arxiv.org/abs/1701.06538), [Switch Transformers](https://arxiv.org/abs/2101.03961), [Titans](https://arxiv.org/abs/2501.00663).

An ordinary frozen MLP's "key-value memory" is not writable runtime RAM. [Geva et al.](https://aclanthology.org/2021.emnlp-main.446/) describe an interpretation of learned feed-forward weights. That differs from updating recurrent state or fast weights during execution.

### Transition algebra matters more than architecture names

> **TL;DR:** What a layer can compute depends on which state changes its math allows, not on its name.

For a recurrence `h_next = A(input) @ h + b(input)`, the allowed transition matrices matter: scalar, diagonal, permutation, diagonal-plus-low-rank, products of reflections, or unrestricted dense transformations.

For a unit key, `I - beta * key @ key.T` removes a component at beta = 1 and reflects it at beta = 2. Negative or signed transitions enable state-tracking behaviors that positive transitions cannot directly express in some linear-recurrence regimes.

Sources:

- [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537).
- [DeltaProduct](https://arxiv.org/abs/2502.10297).
- [Complex KDA](https://arxiv.org/abs/2609.24797), September 2026.
- [A Fine-Grained Hierarchy and Benchmark for State Tracking in Language Models](https://arxiv.org/abs/2611.03094), November 2026 preprint. It separates immutable assignment, mutable assignment, self-referential updates, and general finite-state tracking, and establishes an aperiodic state-tracking result for positive-eigenvalue GDN.

A trained-model report is consistent with this. [Hybrid models can hide reasoning in their RNN state](https://divagr.com/blog/hybrid-models-can-hide-reasoning-in-their-rnn-state), Divyansh Agrawal, October 2026 ([code](https://github.com/divagr18/hidden-reasoning)), reports that under answer-only fine-tuning on parity, OLMo-Hybrid, whose GDN allows beta up to 2, learned to track parity without chain-of-thought and generalized to twice the training length. Capping beta at 1 removed that generalization, and Qwen3.5, which caps beta at 1, fit only the trained lengths. If this holds, recurrent state can carry serial computation that a chain-of-thought monitor cannot see.

These are results under particular architectural, precision, and computation assumptions. They do not say a CPU built from nonlinear circuits plus repeated feedback cannot compute parity. Conversely, an ideal real-valued rotation need not remain exact after a million floating-point updates.

Proposed compiler direction: use assignment-oriented memory for writes, permutation or signed transitions for appropriate state tracking, local operators for spatial communication, and finite lookup circuits for arbitrary bounded transitions. Only introduce shared lowering machinery after concrete backends justify it.

## Computation models beyond a scalar CPU

> **TL;DR:** Proposed alternatives include direct state transitions, dataflow, graph operations, and local grid updates.

### Streaming transducers, stacks, and parsers

> **TL;DR:** Finite transducers and bounded stacks can track state and nesting without instruction fetch.

A deterministic finite-state transition can be `state_next = P_symbol @ state`, with a one-hot state and compiled transition matrices. Add a bounded stack and explicit overflow behavior for nested scopes, expression evaluation, or parsing.

This avoids instruction fetch and a general-purpose decoder for tasks that only require a small transition system. Stack or Forth-like ISAs can simplify implementation, but changing from RISC-V to WebAssembly alone does not establish a fundamentally different computational model.

Evidence includes the automata, compiled-program, and differentiable-stack foundations above. [E01: streaming machines](results/E01/README.md) has local CPU conformance results for ReLU transducers and a delta-rule stack; comparisons across the other candidate backends remain proposed.

### Dataflow, vector, and conditional functional-unit machines

> **TL;DR:** Schedule arithmetic graphs or vector operations directly rather than interpreting scalar instructions.

Compile `gather -> map -> reduce -> scatter` or a scheduled arithmetic graph instead of serial scalar instructions. Potential workloads include bit-parallel arithmetic, expression evaluation, fixed-point inference, and specialized hot paths.

This direction has a precedent: [QwenPU](https://github.com/divagr18/QwenPU) adds compiled vector operations and dedicated DeltaNet memories. The author reports reducing work for one generated token of the nested stories260K model from 2,430,000 trace tokens in ordinary C, to 225,000 with dot-product extensions, to 53,000 with its vector unit. These are related-system implementation reports, not a speedup over native model inference or a matched backend-only experiment.

Proposals: generalize to a small dataflow IR; compare dense predication with expert dispatch; specialize hot programs or basic blocks. Compare interpreter and specialized variants separately, account for construction time, and preserve precise faults and work budgets.

### Graph, semiring, and finite-domain logic machines

> **TL;DR:** Graph problems and logic rules become repeated combine-and-reduce steps.

Many dynamic programs consist of repeated local composition and reduction. Shortest-path relaxation uses min-plus; reachability uses Boolean AND/OR. A machine can expose edge-local computation, neighborhood reduction, and bounded iteration rather than scalar loads and branches.

[Graph Neural Networks are Dynamic Programmers](https://papers.nips.cc/paper_files/paper/2022/hash/8248b1ded388fcdbbd121bcdfea3068c-Abstract-Conference.html) develops the algorithmic alignment connection; it is not itself proof that an arbitrary trained GNN executes an exact graph algorithm.

Positive, finite-domain Datalog suggests another target: relations as Boolean tensors, joins as suitable contractions, existential aggregation as OR, and recursion as least-fixed-point iteration. [TensorLog](https://arxiv.org/abs/1605.06523) is probabilistic differentiable reasoning. [Implementing Tensor Logic](https://arxiv.org/abs/2601.17188), January 2026, includes a recursive-Datalog/tensor-contraction demonstration as well as separate embedding-space experiments. Do not conflate soft inference with exact Boolean entailment.

Proposals here require bounded arithmetic, explicit infinity sentinels, deterministic ties where relevant, and specified nonconvergence or resource-limit behavior.

### Cellular automata, systolic arrays, and signal processing

> **TL;DR:** Local operators and lookup tables can implement specified grid updates, with explicit boundaries and buffering.

For a binary three-cell neighborhood, a local operator can form `left + 2 * center + 4 * right`; a compiled eight-entry table supplies the next cell. Repeated convolution and local lookup then implement a cellular rule.

Sequence positions or a grid can be cells, channels can be registers, and repeated execution can represent time. Simultaneous updates need preserved old state, usually double buffering; boundary conditions are part of the specification.

Sources: [Cellular automata as convolutional neural networks](https://arxiv.org/abs/1809.02942), [Neural GPUs Learn Algorithms](https://arxiv.org/abs/1511.08228). Learned algorithmic behavior in Neural GPU differs from compiling an exact local rule. Universality results requiring an unbounded lattice do not make a fixed grid unbounded-memory.

Related but distinct: structured SSMs may be better used as compiled filters and numerical dynamical systems than forced to emulate random-access RAM. Approximate real-valued simulation needs an explicit error contract, separate from bit-exact cellular or integer tasks.

### Optimizers as machines

> **TL;DR:** Some memory updates are learning steps, and they are exact only in special cases.

The quadratic-loss derivation above makes delta memory a restricted optimizer. Broader TTT/Titans-style updates suggest online regression, adaptive filtering, or task-local fitting as computation models.

The first discriminating check is repeated writes to the same address within an update chunk. Sequential quadratic updates can replace the slot; stale gradients can produce a different result. Momentum and forgetting are additional state, not harmless implementation details.

This is a proposal for specified optimization machines, not a claim that arbitrary learned memory has RAM semantics.

### Energy-based search with an exact verifier

> **TL;DR:** Let a network guess answers, then let an exact checker accept or reject each one.

[Modern Hopfield networks](https://arxiv.org/abs/2008.02217) connect attention with associative energy-based dynamics. This motivates candidate retrieval or constraint-search systems, not a generic efficient global optimizer.

One hybrid is heuristic search followed by a fixed exact verifier. Measure search quality and verification correctness independently. Rejecting invalid candidates does not prove that failure to find a candidate means none exists. Compare equal search budgets against conventional algorithms.

## Latent execution and compilation

> **TL;DR:** Non-text execution needs a feedback contract; compilation needs explicit representation and numerical obligations.

### Replace execution tokens with an explicit feedback contract

> **TL;DR:** Define how machine outputs reach the next computation step without text serialization.

An experimental latent transition can be specified as:

```text
(hidden_next, machine_next, event) = step(hidden, machine)
event = continue | output | halt | fault | host_exit
```

The runtime repeatedly executes within a work budget. This separates the clock, internal payload transport, and audit trace from the text vocabulary. It might avoid embedding/unembedding overhead and language-cache pollution, but does not remove sequential dependencies.

Hiding trace tokens is insufficient: late-layer computations need a defined route back to the next tick's inputs or state. Specify which layer state updates, which learned context freezes, and which outputs are visible to the controller. [Coconut](https://arxiv.org/abs/2412.06769) is a feedback-interface precedent, not an exact-machine implementation.

Compare token feedback, repeated latent computation, and an ordinary native in-process loop at matched information and work budgets. Host initialization, stopping, and I/O remain part of the system even if execution is internal.

### A compiler target, not a claim of free computation

> **TL;DR:** A shared compiler remains proposed; its numerical checks and runtime costs need separate validation.

RASP/Tracr, ALTA, looped transformers, Qwenputer, and Percepta provide precedents for program representation, recurrence, memory, scheduling, and optimized execution. A general compiler combining their strengths remains a research direction.

Separate fixed weights as code/ROM, recurrent state as writable storage, KV history as a log, and runtime-updated fast weights. A capability description should eventually record value domains, state-update semantics, communication, numerical obligations, work units, and snapshot behavior.

Proposed proof-oriented compilation would emit bounds on intermediate values and reductions, representability obligations at casts, addressing invariants, and checks against reference transitions. Testing supports a claim only for the tested cases; a compiler certificate would require a separately defined and validated checker and floating-point model.

## Numerical and resource limits

> **TL;DR:** Exactness depends on rounding, representation, kernels, and resource limits throughout the execution path.

- **Retention:** intended alpha = 1 must not quietly become `1 - epsilon`; idle state then decays as `(1 - epsilon)^steps`.
- **Integer representation:** all consecutive integers through 2^8 in BF16, 2^11 in FP16, 2^24 in FP32, and 2^53 in FP64 are representable. Intermediates, reductions, and casts must also stay within the required domain.
- **Normalization:** correct storage does not guarantee a magnitude-preserving read path. Orthogonal keys and multi-hot normalization require explicit analysis.
- **Kernel equivalence:** Qwenputer's [technical notes](https://github.com/divagr18/qwenputer/blob/main/experimental/riscv/docs/rv32.md) describe residues from an eight-row read in a chunked kernel and a resulting redesign. A graph evaluator alone can miss backend errors.
- **State dtype:** [Lost in Rounding](https://arxiv.org/abs/2611.12127), November 2026 preprint, reports behavior changes from recurrent-state rounding in trained hybrid reasoning models. This is not a compiled-CPU experiment. It supports recording state precision in the run specification.
- **Reduction order:** [Telling identical GPUs apart](https://divagr.com/blog/telling-identical-gpus-apart), Divyansh Agrawal, October 2026 ([code](https://github.com/divagr18/gpu-instance-fingerprint)), reports that atomic-add ordering differs enough between identical GPUs to identify the chip (99.1% among 19 A40s), and that a default `torch.index_add_` exposes it; PyTorch's deterministic mode removes that signal. GPU conformance runs must record deterministic settings and the exact device.
- **Finite memory:** fixed-width, fixed-precision recurrent state has finitely many configurations. Turing-completeness claims must state where growing memory, precision, or context enters.
- **Access cost:** logical one-hot random access can still touch an entire dense state. Fused gather, dense matrix operations, and structured indexed attention have different physical costs.
- **History cost:** an indexed lookup need not scan every past entry, but retaining a log or index still consumes memory. Account for retained entries, index storage, and maintenance.
- **Correct requests:** an exact executor can correctly run a mistaken program or argument choice. Controller correctness, program correctness, and execution correctness are separate.

## Open questions

> **TL;DR:** These questions remain open in the reviewed sources and map to experiments or explicit deferrals.

1. Can a pretrained LM learn reliable typed actions over exact state without serializing every action into text?
2. Does that interface improve generalization or cost over a well-engineered external tool?
3. Does a compiled tensor implementation offer anything beyond an equivalent native module, such as backend portability?
4. Can learned writes be constrained to valid machine states without making training impractical?
5. Which dtypes, kernels, batch sizes, and devices preserve the machine's semantics?
6. What must a joint checkpoint contain to restore both learned context and exact machine state?
7. Which transition algebras make a layer family a good target for a particular computation, rather than a poor RAM emulator?
8. When do direct transducers, dataflow, graph, or spatial machines beat scalar instruction execution?
9. Can compiler-generated numerical obligations and independent checks turn tested conformance into stronger evidence?
10. How much benefit comes from indexed lookup, program specialization, or latent feedback rather than weight-space placement itself?

The reviewed sources do not settle this combination. Absence from this review is not evidence of novelty. [Research coverage](EXPERIMENTS.md#research-coverage) maps each question to an experiment or a scope gate that defers it.

## First controller experiment

> **TL;DR:** The E02: learned interfaces experiment will compare internal and textual interfaces, then compiled and native execution under the same internal interface.

[EXPERIMENTS.md](EXPERIMENTS.md) specifies the research program and repository organization. Start with executor conformance and a streaming machine (E00: memory and E01: streaming machines), then this controller study (E02: learned interfaces). Do not build a complete operating system. Keep transition rules fixed. Train action heads with supervised traces before attempting harder credit assignment.

Compare:

1. The model without a machine.
2. The model with textual external tools.
3. The model with an internal native module.
4. The same internal interface with a compiled tensor module.

Use state-tracking tasks such as nested scopes or record updates. Train on short sequences; test longer sequences and unseen operation combinations within the same memory capacity.

Measure task accuracy separately from executor state equivalence. Account for generated tokens, internal actions, machine instructions, wall-clock latency, and memory. Match available information and resource budgets; do not make subprocess overhead an artificial disadvantage for the tool baseline.

A positive result must distinguish the benefit of the interface from the benefit of compiling it into weights. If native integration wins, that result still counts.

A runtime such as [Shem](https://github.com/JGalego/shem) can supply execution and snapshots; verifying an internal controller and its state transitions still requires dedicated traces.
