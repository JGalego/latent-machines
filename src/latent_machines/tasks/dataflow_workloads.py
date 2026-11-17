"""Seeded ``dataflow/v1`` programs and inputs for E04.

Three workloads, each a program generator plus an input generator:

- ``expression-dag``: random arithmetic DAGs over 8 inputs and two small
  constants, with ``add``, ``sub``, ``mul``, and ``max`` nodes whose
  arguments favor recent nodes, so products chain and sometimes overflow.
- ``dot-product``: a bounded integer dot product of 64 inputs with 64
  constant weights: one ``mul`` per element and one ``sum``.
- ``gather-map-reduce-scatter``: gather 24 of 32 inputs (repeats allowed),
  map each through ``x * c + d``, reduce contiguous segments with ``sum``,
  combine two segment pairs with ``max`` and ``sub``, and scatter the
  results to output slots in a random order.

Inputs are mostly moderate. About one vector in ten is drawn from a wide
range that tends to overflow, and about one in fifty carries an element
outside the 16-bit word range, so input overflow faults are exercised too.
"""

import random

from latent_machines.machines.dataflow import Node, Program

WORKLOADS = ("expression-dag", "dot-product", "gather-map-reduce-scatter")
WIDTH = 16
_LIMIT = 1 << (WIDTH - 1)


def program(workload: str, rng: random.Random) -> Program:
    if workload == "expression-dag":
        return _expression_dag(rng)
    if workload == "dot-product":
        return _dot_product(rng)
    if workload == "gather-map-reduce-scatter":
        return _gather_map_reduce_scatter(rng)
    raise ValueError(f"unknown workload {workload!r}; choose from {WORKLOADS}")


def inputs(workload: str, program: Program, rng: random.Random) -> list[int]:
    """One input vector for ``program``."""
    moderate = {"expression-dag": 15, "dot-product": 31}.get(workload, 500)
    draw = rng.random()
    bound = moderate if draw < 0.9 else _LIMIT - 1
    values = [rng.randint(-bound, bound) for _ in range(program.inputs)]
    if draw > 0.98:
        values[rng.randrange(program.inputs)] = rng.choice((_LIMIT, -_LIMIT - 1))
    return values


def _expression_dag(rng: random.Random, n_inputs: int = 8, ops: int = 30) -> Program:
    nodes = [Node("input", value=i) for i in range(n_inputs)]
    nodes += [Node("const", value=rng.randint(-9, 9)) for _ in range(2)]
    for _ in range(ops):
        op = rng.choices(("add", "sub", "mul", "max"), (0.3, 0.25, 0.25, 0.2))[0]
        recent = max(0, len(nodes) - 12)
        args = (rng.randrange(recent, len(nodes)), rng.randrange(len(nodes)))
        nodes.append(Node(op, args))
    last = len(nodes) - 1
    outputs = (last - 2, last - 1, last, rng.randrange(n_inputs + 2, last))
    return Program(tuple(nodes), outputs, n_inputs)


def _dot_product(rng: random.Random, length: int = 64) -> Program:
    nodes = [Node("input", value=i) for i in range(length)]
    nodes += [Node("const", value=rng.randint(-63, 63)) for _ in range(length)]
    nodes += [Node("mul", (i, length + i)) for i in range(length)]
    nodes.append(Node("sum", tuple(range(2 * length, 3 * length))))
    return Program(tuple(nodes), (len(nodes) - 1,), length)


def _gather_map_reduce_scatter(
    rng: random.Random, n_inputs: int = 32, gathered: int = 24, segments: int = 6
) -> Program:
    nodes = [Node("input", value=rng.randrange(n_inputs)) for _ in range(gathered)]
    mapped = []
    for i in range(gathered):
        scale = len(nodes)
        nodes.append(Node("const", value=rng.randint(-7, 7)))
        shift = len(nodes)
        nodes.append(Node("const", value=rng.randint(-50, 50)))
        nodes.append(Node("mul", (i, scale)))
        nodes.append(Node("add", (len(nodes) - 1, shift)))
        mapped.append(len(nodes) - 1)
    cuts = sorted(rng.sample(range(1, gathered), segments - 1))
    sums = []
    for start, stop in zip([0, *cuts], [*cuts, gathered], strict=True):
        nodes.append(Node("sum", tuple(mapped[start:stop])))
        sums.append(len(nodes) - 1)
    nodes.append(Node("max", (sums[0], sums[1])))
    nodes.append(Node("sub", (sums[2], sums[3])))
    results = [*sums, len(nodes) - 2, len(nodes) - 1]
    rng.shuffle(results)
    return Program(tuple(nodes), tuple(results), n_inputs)
