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

## `reach/v1`, `minplus/v1`, `chain-datalog/v1`

> **TL;DR:** Graph and logic problems run as fixed-point machines: one synchronous round per step until nothing changes.

Three fixed-point machines for [E05](../../../EXPERIMENTS.md#e05---graph-semiring-and-logic-execution), specified in [graph.py](graph.py). Each machine holds one validated problem; its state is a relation or distance table that changes monotonically, one synchronous round per `Iterate` action:

- `reach/v1`: reflexive transitive closure of a directed graph; a round extends every known path by one edge.
- `minplus/v1`: all-pairs shortest paths with integer weights `0..max_weight`; `None` is the infinity sentinel, negative weights are rejected when the problem is built, and any stored distance above `max_distance` faults rather than saturating.
- `chain-datalog/v1`: positive Datalog over binary relations, with chain rules `h(x0, xk) :- r1(x0, x1), ..., rk(x(k-1), xk)` and optional inverted atoms, evaluated naively.

```mermaid
flowchart TD
    S["step(state, Iterate)"] --> H{halted?}
    H -->|yes| FH["fault: halted"]
    H -->|no| B{"rounds = max_rounds?"}
    B -->|yes| FB["fault: budget_exhausted<br/>last state returned, not a fixed point"]
    B -->|no| R["compute next state from previous state only"]
    R --> D{"minplus: any finite entry<br/>> max_distance?"}
    D -->|yes| FD["fault: distance_bound_exceeded"]
    D -->|no| C{"next = previous?"}
    C -->|yes| HL["halt: fixed point"]
    C -->|no| CO["continue"]
```

Faults leave the state unchanged and every action costs one work unit. The default `max_rounds` always suffices for a valid problem: `n` for the graph machines and `n² · |IDB| + 1` for Datalog.
