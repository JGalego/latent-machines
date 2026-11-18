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

## `dataflow/v1`

> **TL;DR:** A DAG of integer operations evaluated level by level; the first out-of-range node faults, and nothing wraps.

Signed integer dataflow graphs, the reference for [E04](../../../EXPERIMENTS.md#e04---dataflow-specialization-and-conditional-units). Nodes are `input`, `const`, `add`, `sub`, `mul`, `max`, and n-ary `sum`; arguments name earlier nodes. The docstring in [dataflow.py](dataflow.py) is the authoritative specification.

```mermaid
flowchart TD
    S["step(program, inputs, state)"] --> F{finished?}
    F -->|yes| FF["fault: finished<br/>(no work)"]
    F -->|no| L["evaluate the next level<br/>in node-index order"]
    L --> R{"node value within<br/>width-bit signed range?"}
    R -->|no| FO["fault: overflow, naming the node<br/>later nodes not evaluated"]
    R -->|yes, more nodes| L
    R -->|yes, level done| E{last level?}
    E -->|no| C[continue]
    E -->|yes| O["output: values of the output nodes"]
```

A tick is one level; levels are the safepoints where snapshots are taken. Work is one unit per node evaluated, the faulting node included. A `sum` is checked on its final value only.

## `cnf/v1`

> **TL;DR:** A stateless checker accepts an assignment only when every clause has a true literal; malformed assignments fault.

Assignment verifier for CNF formulas, the reference for [E08](../../../EXPERIMENTS.md#e08---heuristic-search-with-an-exact-verifier). The docstring in [cnf.py](cnf.py) is the authoritative specification.

```mermaid
flowchart TD
    C["check(formula, assignment)"] --> T{"every value an int?"}
    T -->|no| E[TypeError]
    T -->|yes| L{"length = num_vars?"}
    L -->|no| FL["fault: wrong_length"]
    L -->|yes| V{"every value 0 or 1?"}
    V -->|no| FV["fault: value_out_of_range"]
    V -->|yes| S["evaluate every clause"]
    S --> A{"all satisfied?"}
    A -->|yes| ACC["output 1: accept"]
    A -->|no| REJ["output 0: reject,<br/>unsatisfied count, first index"]
```

A valid check costs one work unit per clause; a fault costs none. Accepting is the only claim the verifier makes: a rejection says nothing about whether another assignment satisfies the formula.

## `cellular/v1`

> **TL;DR:** A grid of live and dead cells updates all at once from neighbor counts, with dead or wrapping edges.

Binary outer-totalistic cellular automaton, the exact-branch reference for [E06](../../../EXPERIMENTS.md#e06---spatial-and-structured-dynamical-machines). [cellular.py](cellular.py) is the authoritative specification. A rule such as `B3/S23` lists the neighbor counts that give birth to a dead cell and keep a live one alive.

```mermaid
flowchart LR
    A["Advance()"] --> R["read the whole current grid<br/>(boundary: zero or torus)"]
    R --> N["count the 8 neighbors of every cell"]
    N --> U["next = count ∈ birth (dead) or survival (live)"]
    U --> W["replace the grid at once; generation + 1"]
    W --> O["output: population"]
    S["Set(row, col, v)"] --> C{"cell in grid?"}
    C -->|no| FC["fault: cell_out_of_range"]
    C -->|yes| V{"v ∈ {0, 1}?"}
    V -->|no| FV["fault: value_out_of_range"]
    V -->|yes| OK["overwrite one cell; continue"]
```

`Advance` costs one work unit per cell; `Set` costs one, faults included. Snapshots carry a digest of the rule, size, boundary, and starting grid.

## `lds/v1`

> **TL;DR:** A linear filter with a float64 reference, measured under a tolerance rather than checked for exactness.

Single-input, single-output linear dynamical system `y = C x + D u`, `x ← A x + B u`, the reference for E06's numerical branch. [lds.py](lds.py) evaluates it in IEEE double precision and builds the E06 systems: a lightly damped oscillator (pole radius 0.999), a second-order Butterworth low-pass, and a random stable 8-dimensional system. It is a declared numerical reference, not an exact oracle.
