# Machines

> **TL;DR:** Independent integer oracles define the state transitions that backends must match.

Specifications and independent integer oracles. A machine here depends on no model family or tensor backend; backends are validated against it. See [ownership boundaries](../../../EXPERIMENTS.md#ownership-boundaries).

## `memory/v1`

> **TL;DR:** Fixed-capacity memory supports reads, writes, idle, and halt; invalid addresses or values fault without changing state.

Fixed-capacity random-access memory of unsigned words, the reference for [E00](../../../EXPERIMENTS.md#e00---memory-and-transition-conformance). The docstring in [memory.py](memory.py) is the authoritative specification; this diagram shows the order of checks in one step.

```mermaid
flowchart TD
    S["step(state, action)"] --> H{halted?}
    H -->|yes| FH["fault: halted"]
    H -->|no| K{action}
    K -->|Read or Write| A{"0 ≤ address < capacity?"}
    A -->|no| FA["fault: address_out_of_range"]
    A -->|yes, Read| O["output: stored word"]
    A -->|yes, Write| V{"0 ≤ value < 2^word_bits?"}
    V -->|no| FV["fault: value_out_of_range"]
    V -->|yes| W["store value; continue"]
    K -->|Idle| C[continue]
    K -->|Halt| HL["set halted; halt"]
```

Every outcome costs one work unit. Faults leave the state unchanged, and an action outside the schema raises `TypeError` before any check.

## `transducer/v1`

> **TL;DR:** A machine that reads symbols one at a time and looks up its next state and output in a table.

Deterministic finite-state transducer, the reference for [E01](../../../EXPERIMENTS.md#e01---exact-streaming-machines-and-transition-algebras). [transducer.py](transducer.py) is the authoritative specification and builds the E01 targets: running parity, modular counters, and S_n permutation tracking.

```mermaid
flowchart LR
    F["Feed(symbol)"] --> R{"0 ≤ symbol < alphabet?"}
    R -->|no| X["fault: symbol_out_of_range<br/>state unchanged"]
    R -->|yes| T["state ← next_state[state][symbol]"] --> O["output: output[state][symbol]"]
```

Snapshots carry a digest of both tables, so a snapshot cannot be restored into a different transducer.

## `stack/v1`

> **TL;DR:** A bounded last-in, first-out stack faults on overflow or underflow without changing state.

Bounded stack of unsigned words, the reference for E01's stack parser. [stack.py](stack.py) is the authoritative specification.

```mermaid
flowchart LR
    P["Push(v)"] --> V{"value in range?"}
    V -->|no| FV["fault: value_out_of_range"]
    V -->|yes| C{"full?"}
    C -->|yes| FO["fault: overflow"]
    C -->|no| OK1["push; continue"]
    Q["Pop()"] --> E{"empty?"}
    E -->|yes| FU["fault: underflow"]
    E -->|no| OK2["pop; output top word"]
```

Every fault leaves the stack unchanged, and every action costs one work unit.

## `optimizer-memory/v1`

> **TL;DR:** A memory written by optimizer steps: one gradient step per action, with the optimizer's settings as the specification.

An associative map `f(k)` updated by one optimizer step per action on `½‖f(k) − y‖²`, the reference for [E07](../../../EXPERIMENTS.md#e07---optimization-defined-state-updates). [optimizer_memory.py](optimizer_memory.py) is the authoritative specification and a plain-Python float64 reference. It fixes what the optimizer does; whether that behaves like `memory/v1` is what E07 measures.

```mermaid
flowchart LR
    A["memory/v1 action"] --> V{valid?}
    V -->|no| F["fault; nothing changes"]
    V -->|Write a v| G["g = −∇ at θ<br/>(at chunk start θ₀ if chunk > 1)"]
    V -->|Read, Idle, Halt| Z["g = 0"]
    G --> N["g ← g / max(‖g‖, ε) if normalize"]
    Z --> N
    N --> M["m ← momentum · m + g"]
    M --> U["θ ← (1 − decay) θ + lr · m"]
    U -->|Read a| O["output f(e_a), threshold 0.5"]
```

The model is linear, `f(k) = W k`, or a one-hidden-layer ReLU MLP. With a linear model, one-hot keys, `lr = 1`, and every other feature off, a write is delta replacement. Reads, idles, and halts are optimizer steps without a sample, so decay and momentum still act on them.
