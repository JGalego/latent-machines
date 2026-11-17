"""Dataflow backends against the ``dataflow/v1`` oracle."""

import random

import pytest

torch = pytest.importorskip("torch")

from latent_machines.backends.dataflow import (  # noqa: E402
    CompiledGraph,
    Outcome,
    ScalarDelta,
    python_int,
)
from latent_machines.machines.dataflow import Dataflow, Program  # noqa: E402
from latent_machines.tasks import dataflow_workloads as workloads  # noqa: E402

LOWERINGS = [
    ("gather", None),
    ("weights", None),
    ("dispatch", 2),
    ("grouped", None),
]


def cases(count: int = 2, vectors: int = 12) -> list[tuple[Program, list[list[int]]]]:
    rng = random.Random(0)
    return [
        (
            program := workloads.program(workload, rng),
            [workloads.inputs(workload, program, rng) for _ in range(vectors)],
        )
        for workload in workloads.WORKLOADS
        for _ in range(count)
    ]


def expected(program: Program, x: list[int]) -> tuple:
    event, state, work = Dataflow().run(program, x)
    return event, state.values, work


def observed(outcome: Outcome) -> tuple:
    return outcome.event, outcome.values, outcome.work


def test_python_int_matches_the_oracle() -> None:
    for program, inputs in cases():
        for x in inputs:
            assert observed(python_int(program, x, 16)) == expected(program, x)


def test_scalar_delta_matches_the_oracle_and_resumes_from_a_snapshot() -> None:
    for program, inputs in cases(count=1, vectors=1):
        for x in inputs:
            outcome, snapshot = ScalarDelta().run(program, x, snapshot_at=3)
            assert observed(outcome) == expected(program, x)
            assert outcome.residue == 0.0
            assert snapshot in ("agrees", "not reached")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize(("lowering", "capacity"), LOWERINGS)
def test_compiled_graph_is_exact_in_wide_formats(
    dtype: torch.dtype, lowering: str, capacity: int | None
) -> None:
    if lowering == "weights" and dtype == torch.int64:
        pytest.skip("dense weights are a floating-point lowering here")
    faults = 0
    for program, inputs in cases():
        graph = CompiledGraph(program, 16, dtype, lowering, capacity)
        state = graph.run(inputs)
        for column, x in enumerate(inputs):
            outcome = graph.outcome(state, column)
            assert observed(outcome) == expected(program, x)
            assert outcome.residue == 0.0
            faults += outcome.event.kind.value == "fault"
    assert faults > 0  # the workloads exercise overflow


def test_batch_composition_does_not_change_results() -> None:
    program, inputs = cases(count=1)[0]
    graph = CompiledGraph(program, 16, torch.float32, "gather")
    batched = graph.run(inputs)
    for column, x in enumerate(inputs):
        alone = graph.outcome(graph.run([x]), 0)
        assert observed(alone) == observed(graph.outcome(batched, column))


def test_bfloat16_cannot_hold_every_word() -> None:
    """A precision probe: 16-bit words do not fit an 8-bit significand."""
    program, inputs = next(c for c in cases() if c[0].inputs == 64)  # dot product
    graph = CompiledGraph(program, 16, torch.bfloat16, "gather")
    state = graph.run(inputs)
    wrong = sum(
        observed(graph.outcome(state, c)) != expected(program, x)
        for c, x in enumerate(inputs)
    )
    assert wrong > 0


def test_dispatch_pads_but_never_drops_nodes() -> None:
    program, _ = cases(count=1)[0]
    graph = CompiledGraph(program, 16, torch.float32, "dispatch", capacity=3)
    grouped = CompiledGraph(program, 16, torch.float32, "grouped")
    assert graph.padded_slots() > 0 and grouped.padded_slots() == 0
    for level, nodes in zip(graph.levels, graph.level_nodes[1:], strict=True):
        routed = sorted(
            int(r) for rounds in level.experts.values() for m, _, _ in rounds for r in m
        )
        assert routed == list(range(len(nodes)))


def test_encode_inverts_decode_at_a_level_boundary() -> None:
    program, inputs = cases(count=1)[0]
    graph = CompiledGraph(program, 16, torch.float32, "weights")
    x = torch.tensor([inputs[0]], dtype=torch.float32).T
    state = graph.initial_state(1)
    state = graph.step(state, x)
    canonical, residue = graph.decode(state, 0)
    if canonical.finished:
        pytest.skip("first vector faults at level 0")
    restored = graph.encode(canonical)
    assert residue == 0.0
    assert torch.equal(restored.values, state.values)
    assert torch.equal(restored.work, state.work)


def test_unknown_lowering_and_missing_capacity_are_rejected() -> None:
    program, _ = cases(count=1)[0]
    with pytest.raises(ValueError):
        CompiledGraph(program, lowering="sparse")
    with pytest.raises(ValueError):
        CompiledGraph(program, lowering="dispatch")
