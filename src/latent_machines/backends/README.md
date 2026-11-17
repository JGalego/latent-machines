# Backends

> **TL;DR:** Ways to run a machine, each checked step by step against its reference.

Implementations of machine semantics, validated step by step against the integer oracles in [machines/](../machines/). See [ownership boundaries](../../../EXPERIMENTS.md#ownership-boundaries).

## Delta-rule memory

> **TL;DR:** The delta-rule backend stores memory in a state matrix, with one column per slot and one row per bit.

[delta.py](delta.py) stores [`memory/v1`](../machines/README.md#memoryv1) in a Gated DeltaNet state matrix `S` of shape `(word_bits, capacity)`: one column per slot, one row per bit. The module docstring is the authoritative description.

```mermaid
flowchart LR
    A[action] --> V{"host checks:<br/>halted, address, value"}
    V -->|invalid| F["fault event<br/>S untouched"]
    V -->|valid| L["lower to k, v, α, β, q"]
    L --> U["S ← α S (I − β k kᵀ) + β v kᵀ"]
    U -->|Read| O["o = S q<br/>threshold bits at 0.5"]
    O --> E1[output event]
    U -->|"Write, Idle, Halt"| E2[continue or halt event]
    U -.->|"decode and residue<br/>(conformance only)"| C["canonical state<br/>compared with oracle"]
```

Every valid action runs one recurrence step with gates at their endpoints:

| Action | Key `k` | Value `v` | `α` | `β` | Query `q` |
|---|---|---|---|---|---|
| `Write(a, x)` | one-hot `a` | bits of `x` | 1 | 1 | none |
| `Read(a)` | zero | zero | 1 | 0 | one-hot `a` |
| `Idle`, `Halt` | zero | zero | 1 | 0 | none |

Conformance requires zero residue: every entry of `S` is exactly 0 or 1, so thresholding never hides drift. Normalization variants, non-endpoint gates, multi-hot keys, and chunked or fused kernels are separate backends to measure, not options of this one.

## ReLU lookup circuit

> **TL;DR:** A two-layer ReLU network computes the next state and output from a compiled transition table.

[lookup.py](lookup.py) compiles a [`transducer/v1`](../machines/transducer.py) table into a ReLU network with explicit feedback. Hidden unit `(i, a)` fires only when the state is `i` and the symbol is `a`; its outgoing weights select the next state and the output.

```mermaid
flowchart LR
    S["state s<br/>(one-hot)"] --> C["[s; x]"]
    X["symbol x<br/>(one-hot)"] --> C
    C --> Z["z = relu(W_in [s; x] − 1)<br/>one unit per (state, symbol)"]
    Z --> N["s′ = W_next z"]
    Z --> Y["y = W_out z → output event"]
    N -.->|feedback| S
```

Every intermediate is 0, 1, or 2, so the circuit is exact in float32, float64, bfloat16, and float16. Width is `states × symbols`, depth is 2, and each unit has four nonzero weights. Smooth activations such as SwiGLU are a separate construction, not an option of this one.

## Delta-rule stack

> **TL;DR:** The stack combines delta-rule memory with a one-hot depth pointer moved by shift matrices.

[delta_stack.py](delta_stack.py) stores [`stack/v1`](../machines/stack.py) in a delta-rule matrix, one column per slot, plus a one-hot depth pointer moved by shift matrices. Overflow and underflow come from the pointer's last and first entries, read from the tensor.

```mermaid
flowchart LR
    P["Push(v)"] --> F{"pointer at top?"}
    F -->|yes| O[overflow fault]
    F -->|no| W["write bits(v) at slot = pointer<br/>(delta rule, β = 1)"] --> U["pointer ← U · pointer"]
    Q["Pop()"] --> E{"pointer at bottom?"}
    E -->|yes| N[underflow fault]
    E -->|no| D["pointer ← Uᵀ · pointer"] --> R["read slot → output"] --> C["clear slot<br/>(write zeros, β = 1)"]
```

Clearing on pop keeps freed slots at zero, so the live tensors always equal a freshly encoded snapshot. Tests require this at every step.

## Semiring tensor kernels

> **TL;DR:** Graph rounds become dense matrix products: Boolean products clamped to 1, or min-plus broadcasts.

[semiring.py](semiring.py) lowers each round of the [E05 machines](../machines/README.md#reachv1-minplusv1-chain-datalogv1) to dense `n × n` tensors; [graph_native.py](graph_native.py) holds the conventional sparse algorithms (BFS, Dijkstra, semi-naive Datalog) that compute the same fixed points directly.

```mermaid
flowchart LR
    R["relation R (0/1)"] --> P["R · A"] --> U["clamp(R + R·A, max=1)"]
    D["distances D (+inf = unreachable)"] --> M["min_k D[:, k] + W[k, :]"] --> N["min(D, ·)"]
    X["IDB relations"] --> J["chain body: product of 0/1 matrices<br/>(transposed for inverted atoms), clamped"] --> V["union into head"]
    U & N & V --> E{"equal to previous?"}
    E -->|yes| H[halt]
    E -->|no| C[continue]
```

No format-specific range check is made: integers beyond a format's exact range (256 in bfloat16, 2048 in float16) round, and the decoded state then disagrees with the oracle.
