# Machines

> **TL;DR:** Plain-code reference machines that every neural version must match.

Specifications and independent integer oracles. A machine here depends on no model family or tensor backend; backends are validated against it. See [ownership boundaries](../../../EXPERIMENTS.md#ownership-boundaries).

## `memory/v1`

> **TL;DR:** A tiny memory that reads, writes, idles, and halts; bad requests fail without damage.

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
