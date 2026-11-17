"""reach/v1, minplus/v1, and chain-datalog/v1 oracles."""

import json

import pytest

from latent_machines.machines.events import EventKind
from latent_machines.machines.graph import (
    Atom,
    ChainDatalog,
    GraphFault,
    Iterate,
    Reachability,
    Rule,
    ShortestPaths,
    compose_chain,
)


def run(machine):
    state, events = machine.initial_state(), []
    while True:
        t = machine.step(state, Iterate())
        assert t.work == 1
        state = t.next_state
        events.append(t.event)
        if t.event.kind is not EventKind.CONTINUE:
            return state, events


def test_reachability_closes_a_chain_in_n_rounds_and_halts() -> None:
    machine = Reachability(4, [(0, 1), (1, 2), (2, 3)])
    state, events = run(machine)
    assert [e.kind for e in events] == [EventKind.CONTINUE] * 3 + [EventKind.HALT]
    assert state.relation == {(u, v) for u in range(4) for v in range(u, 4)}
    assert state.rounds == 4 == machine.max_rounds and state.halted
    after = machine.step(state, Iterate())
    assert after.event.reason is GraphFault.HALTED and after.next_state == state


def test_reachability_keeps_isolated_vertices_reflexive() -> None:
    state, _ = run(Reachability(3, [(0, 0)]))
    assert state.relation == {(0, 0), (1, 1), (2, 2)}


def test_budget_exhaustion_returns_the_last_state_unchanged() -> None:
    machine = Reachability(4, [(0, 1), (1, 2), (2, 3)])
    machine.max_rounds = 2
    state = machine.initial_state()
    for _ in range(2):
        state = machine.step(state, Iterate()).next_state
    t = machine.step(state, Iterate())
    assert t.event.kind is EventKind.FAULT
    assert t.event.reason is GraphFault.BUDGET_EXHAUSTED
    assert t.next_state == state and not state.halted


def test_shortest_paths_use_smallest_parallel_weight_and_infinity() -> None:
    machine = ShortestPaths(4, [(0, 1, 5), (0, 1, 2), (1, 2, 0), (0, 2, 3)], 9, 20)
    state, events = run(machine)
    assert events[-1].kind is EventKind.HALT
    assert state.distances[0] == (0, 2, 2, None)
    assert state.distances[3] == (None, None, None, 0)


def test_shortest_paths_fault_when_a_stored_distance_exceeds_the_bound() -> None:
    machine = ShortestPaths(3, [(0, 1, 6), (1, 2, 6)], 9, 10)
    state, events = run(machine)
    assert events[-1].reason is GraphFault.DISTANCE_BOUND_EXCEEDED
    assert state.distances[0] == (0, 6, None)  # the faulting round left no trace


@pytest.mark.parametrize("edge", [(0, 1, -1), (0, 1, 10), (0, 3, 1), (0, 1, 1.0)])
def test_invalid_edges_are_rejected_at_construction(edge) -> None:
    with pytest.raises(ValueError):
        ShortestPaths(3, [edge], 9, 100)


def test_same_generation_program() -> None:
    rules = [
        Rule("anc", (Atom("par"),)),
        Rule("anc", (Atom("anc"), Atom("par"))),
        Rule("sg", (Atom("par"), Atom("par", inverted=True))),
        Rule("sg", (Atom("par"), Atom("sg"), Atom("par", inverted=True))),
    ]
    # 0 is the root; 1 and 2 are its children; 3 is 1's child, 4 is 2's child.
    machine = ChainDatalog(5, rules, {"par": [(1, 0), (2, 0), (3, 1), (4, 2)]})
    state, events = run(machine)
    assert events[-1].kind is EventKind.HALT
    anc = machine.relation(state, "anc")
    sg = machine.relation(state, "sg")
    assert anc == {(1, 0), (2, 0), (3, 1), (4, 2), (3, 0), (4, 0)}
    assert (3, 4) in sg and (1, 2) in sg and (0, 0) not in sg and (1, 3) not in sg


def test_datalog_rejects_rules_over_unknown_or_extensional_heads() -> None:
    with pytest.raises(ValueError):
        ChainDatalog(2, [Rule("par", (Atom("par"),))], {"par": []})
    with pytest.raises(ValueError):
        ChainDatalog(2, [Rule("p", (Atom("q"),))], {"par": []})


def test_compose_chain() -> None:
    assert compose_chain([frozenset({(0, 1)}), frozenset({(1, 2), (1, 3)})]) == {
        (0, 2),
        (0, 3),
    }


@pytest.mark.parametrize(
    "machine",
    [
        Reachability(3, [(0, 1), (1, 2)]),
        ShortestPaths(3, [(0, 1, 2), (1, 2, 3)], 9, 100),
        ChainDatalog(3, [Rule("p", (Atom("e"), Atom("e")))], {"e": [(0, 1), (1, 2)]}),
    ],
)
def test_snapshot_round_trips_through_json(machine) -> None:
    state = machine.step(machine.initial_state(), Iterate()).next_state
    restored = machine.restore(json.loads(json.dumps(machine.snapshot(state))))
    assert restored == state


def test_snapshot_for_another_problem_is_rejected() -> None:
    a, b = Reachability(3, [(0, 1)]), Reachability(3, [(1, 2)])
    with pytest.raises(ValueError):
        b.restore(a.snapshot(a.initial_state()))
