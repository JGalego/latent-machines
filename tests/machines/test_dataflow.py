"""The ``dataflow/v1`` oracle: levels, precise overflow faults, snapshots."""

import json

import pytest

from latent_machines.machines.dataflow import (
    Dataflow,
    DataflowEvent,
    FaultReason,
    Node,
    Program,
)
from latent_machines.machines.events import EventKind

# (x0 + x1) * 3, max(x0, x1) - 2, outputs both
PROGRAM = Program(
    (
        Node("input", value=0),
        Node("input", value=1),
        Node("const", value=3),
        Node("add", (0, 1)),
        Node("max", (0, 1)),
        Node("const", value=2),
        Node("mul", (3, 2)),
        Node("sub", (4, 5)),
    ),
    outputs=(6, 7),
    inputs=2,
)


def test_levels_order_by_depth_then_index() -> None:
    assert Dataflow().levels(PROGRAM) == ((0, 1, 2, 5), (3, 4), (6, 7))


def test_run_outputs_values_and_work() -> None:
    event, state, work = Dataflow().run(PROGRAM, [4, -1])
    assert event == DataflowEvent(EventKind.OUTPUT, outputs=(9, 2))
    assert work == len(PROGRAM.nodes)
    assert state.finished and state.values == (4, -1, 3, 3, 4, 2, 9, 2)


def test_overflow_faults_at_first_node_in_evaluation_order() -> None:
    machine = Dataflow(width=8)  # -128..127
    # x0 + x1 = 120 is fine; 120 * 3 overflows at node 6, before node 7.
    event, state, work = machine.run(PROGRAM, [60, 60])
    assert event == DataflowEvent(EventKind.FAULT, node=6, reason=FaultReason.OVERFLOW)
    assert work == 7
    assert state.values[6] is None and state.values[7] is None
    # x0 + x1 = 200 overflows at node 3, before node 4 in level 1.
    event, state, work = machine.run(PROGRAM, [100, 100])
    assert (event.node, work) == (3, 5)
    assert state.values == (100, 100, 3, None, None, 2, None, None)


def test_input_out_of_range_is_an_overflow_fault() -> None:
    event, _, work = Dataflow(width=8).run(PROGRAM, [1, 128])
    assert (event.kind, event.node, work) == (EventKind.FAULT, 1, 2)


def test_stepping_after_finish_faults_without_work() -> None:
    machine = Dataflow()
    _, state, _ = machine.run(PROGRAM, [1, 2])
    transition = machine.step(PROGRAM, [1, 2], state)
    assert transition.event.reason is FaultReason.FINISHED
    assert (transition.work, transition.next_state) == (0, state)


def test_sum_checks_only_its_final_value() -> None:
    program = Program(
        (
            Node("input", value=0),
            Node("input", value=1),
            Node("sum", (0, 0, 1)),
        ),
        outputs=(2,),
        inputs=2,
    )
    event, _, _ = Dataflow(width=8).run(program, [100, -100])
    assert event.outputs == (100,)  # partial sum 200 is not checked


@pytest.mark.parametrize(
    "nodes",
    [
        (Node("add", (0, 0)),),  # argument not earlier
        (Node("input", value=5),),  # no such input
        (Node("const", value=1 << 15),),  # constant out of range
        (Node("input", value=0), Node("mul", (0,))),  # arity
        (Node("input", value=0), Node("sum", ())),
        (Node("nop"),),
    ],
)
def test_malformed_programs_are_rejected(nodes: tuple[Node, ...]) -> None:
    with pytest.raises(ValueError):
        Dataflow().initial_state(Program(nodes, (0,), 1))


def test_wrong_inputs_are_outside_the_schema() -> None:
    machine = Dataflow()
    state = machine.initial_state(PROGRAM)
    with pytest.raises(ValueError):
        machine.step(PROGRAM, [1], state)
    with pytest.raises(TypeError):
        machine.step(PROGRAM, [1, True], state)


def test_snapshot_round_trip_at_every_safepoint() -> None:
    machine = Dataflow()
    state = machine.initial_state(PROGRAM)
    while not state.finished:
        saved = json.loads(json.dumps(machine.snapshot(state)))
        assert machine.restore(saved, PROGRAM) == state
        state = machine.step(PROGRAM, [4, -1], state).next_state


def test_restore_rejects_values_ahead_of_their_level() -> None:
    machine = Dataflow()
    saved = machine.snapshot(machine.initial_state(PROGRAM))
    saved["values"][3] = 7
    with pytest.raises(ValueError):
        machine.restore(saved, PROGRAM)
