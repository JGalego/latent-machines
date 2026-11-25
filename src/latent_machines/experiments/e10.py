"""E10: a machine layer inside the model.

A frozen model reads the preamble plus one statement per forward pass, so the
machine is the only carrier of state. Programs are E03's copy programs:
``Copy x to y.`` needs a read whose value feeds a write. Four arms differ in
where the memory lives and how a read returns:

- O: the memory is outside. Trained heads on the last layer issue ticks
  (``write``, ``read``, ``end``) and receive each tick's reply as a typed
  feedback code (E03's arm H on windows).
- M: the memory is inside layer ``k``. Trained projections of the residual
  stream choose a macro, records, and a value; the delta-rule memory
  executes it inside the pass, a copy included; the read is projected to the
  model's width and added to the residual stream, and an answer head reads
  the last layer.
- Mh: as M, but the read goes to the answer head as an embedded value and
  does not enter the residual stream.
- M-reject: M with a ``reject`` macro, trained on invalid statements too.

Every arm trains on gold actions and gold answers, so supervision is the
same. The pre-registered predictions are in EXPERIMENTS.md; this runner
evaluates them as written. A shadow integer oracle checks the delta-rule
backend for every arm and the in-layer memory after every statement.

    uv run --locked python -m latent_machines.experiments.e10 --model tiny-random
    uv run --locked python -m latent_machines.experiments.e10 --device cuda

Exit status: 0 when the run completes with no mismatch, 1 if the executor or
the in-layer memory disagrees with its oracle, 2 if the run failed.
"""

import argparse
import json
import math
import random
import resource
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from latent_machines.backends.delta import DeltaMemory
from latent_machines.controllers import layer_machine as lmach
from latent_machines.controllers.frozen_lm import FrozenLM
from latent_machines.controllers.heads import TickHead, parameter_count
from latent_machines.experiments import e02, lockstep, records
from latent_machines.interfaces import ticks
from latent_machines.machines.events import Event
from latent_machines.machines.memory import Action, Read, Write
from latent_machines.tasks import copies as task
from latent_machines.tasks import records as record_task

EXPERIMENT = "E10"
MODULE = "latent_machines.experiments.e10"
ARMS = ("O", "M", "Mh", "M-reject")
LENGTH_SPLITS = (
    "test_in_distribution",
    "test_length_32",
    "test_length_64",
    "test_length_256",
)
TEST_SPLITS = (*LENGTH_SPLITS, "test_held_out_pairs")
HELD_OUT = record_task.HELD_OUT
VALUES = lmach.VALUES
TICK_SIZES = (len(ticks.OPERATIONS), len(task.NAMES), VALUES)
NON_INFERIORITY = 0.02  # prediction 1: M may trail O by at most this
REJECT_MIN = 0.95  # predictions 4 and 5
FALSE_REJECT_MAX = 0.01  # prediction 4
O_SANITY = 0.95
PARAMETER_TOLERANCE = 0.05
INVALID_SHARE = 0.15  # invalid rows added to M-reject's training, per valid row
HELD_OUT_SHARE = 0.2  # of the trained-form invalid texts, kept for testing
INVALID_TRAIN_RECORDS = "ijk"
INVALID_UNSEEN_RECORD = "l"
TRACE_PROGRAMS = 3


@dataclass(frozen=True)
class Config:
    model: str
    revision: str | None
    seeds: tuple[int, ...]
    data_seed: int
    train: int
    test: int
    long_test: int
    epochs: int
    hidden: int
    learning_rate: float
    batch_size: int
    layer: int


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bound(questions: int) -> float:
    """E02's conservative two-standard-error bound for a difference of two
    proportions over ``questions`` questions."""
    return 2 * math.sqrt(0.5 / questions)


# ---------- programs ----------


def avoid_held_out(program: task.Program, rng: random.Random) -> task.Program:
    """Redraw the value of every ``set`` that uses a held-out pair."""
    out = []
    for s in program.statements:
        if s.kind == "set" and (s.record, s.value) in HELD_OUT:
            value = rng.randrange(VALUES)
            while (s.record, value) in HELD_OUT:
                value = rng.randrange(VALUES)
            s = task.Statement("set", s.record, value=value)
        out.append(s)
    return task.Program(tuple(out))


def uses_held_out(program: task.Program) -> bool:
    return any(
        s.kind == "set" and (s.record, s.value) in HELD_OUT for s in program.statements
    )


def draw(name: str, count: int, low: int, high: int, seed: int, held: str) -> list:
    """``count`` programs; ``held`` is ``exclude`` or ``include`` for the pairs."""
    split = task.Split(name, count, low, high)
    rng = random.Random(f"{seed}:copies:{name}")
    fix = random.Random(f"{seed}:copies:{name}:redraw")
    out: list[task.Program] = []
    while len(out) < count:
        program = task.random_program(rng, split)
        if held == "include":
            if uses_held_out(program):
                out.append(program)
        else:
            out.append(avoid_held_out(program, fix))
    return out


def splits(config: Config) -> dict[str, tuple[dict[str, Any], list[task.Program]]]:
    spec = [
        ("train", config.train, 2, 8, "exclude"),
        ("test_in_distribution", config.test, 2, 8, "exclude"),
        ("test_length_32", config.long_test, 32, 32, "exclude"),
        ("test_length_64", config.long_test, 64, 64, "exclude"),
        ("test_length_256", config.long_test, 256, 256, "exclude"),
        ("test_held_out_pairs", config.test, 2, 8, "include"),
    ]
    return {
        name: (
            {"count": n, "min": lo, "max": hi, "held_out_pairs": held},
            draw(name, n, lo, hi, config.data_seed, held),
        )
        for name, n, lo, hi, held in spec
    }


def valid_texts() -> list[str]:
    """Every statement text a copy program can contain."""
    names = task.NAMES
    texts = [f" Set {n} to {v}." for n in names for v in range(VALUES)]
    texts += [f" Copy {a} to {b}." for a in names for b in names if a != b]
    texts += [f" What is {n}?" for n in names]
    return texts


def held_out_texts() -> list[str]:
    return [f" Set {task.NAMES[r]} to {v}." for r, v in sorted(HELD_OUT)]


def invalid_sets(seed: int) -> dict[str, list[str]]:
    """Invalid statements: trained forms (split for testing) and unseen forms."""
    names = task.NAMES
    pool = []
    for r in INVALID_TRAIN_RECORDS:
        pool += [f" Set {r} to {v}." for v in range(VALUES)]
        pool.append(f" What is {r}?")
        for n in names:
            pool += [f" Copy {n} to {r}.", f" Copy {r} to {n}."]
    pool += [f" Set {n} to {v}." for n in names for v in range(16, 32)]
    random.Random(f"invalid:{seed}").shuffle(pool)
    held = round(len(pool) * HELD_OUT_SHARE)
    r = INVALID_UNSEEN_RECORD
    record = [f" Set {r} to {v}." for v in range(VALUES)] + [f" What is {r}?"]
    for n in names:
        record += [f" Copy {n} to {r}.", f" Copy {r} to {n}."]
    values = [f" Set {n} to {v}." for n in names for v in range(32, 48)]
    return {
        "trained_form_train": pool[held:],
        "trained_form_test": pool[:held],
        "unseen_record": record,
        "unseen_values": values,
    }


# ---------- gold control ----------


@dataclass
class Gold:
    actions: list[list[Action]]
    events: list[list[Event]]
    codes: list[list[ticks.FeedbackCode]]


def gold(program: task.Program) -> Gold:
    machine = task.machine()
    state = machine.initial_state()
    actions = program.gold_actions()
    events, codes = [], []
    for statement_actions in actions:
        evs, cs = [], []
        for action in statement_actions:
            transition = machine.step(state, action)
            state = transition.next_state
            evs.append(transition.event)
            cs.append(ticks.feedback(action, transition.event))
        events.append(evs)
        codes.append(cs)
    return Gold(actions, events, codes)


def macro_target(statement: task.Statement) -> tuple[int, int, int, int]:
    """Gold (macro, record, source, value); ``-1`` where a field is unused."""
    if statement.kind == "set":
        assert statement.value is not None
        return lmach.SET, statement.record, -1, statement.value
    if statement.kind == "copy":
        assert statement.source is not None
        return lmach.COPY, statement.record, statement.source, -1
    return lmach.GET, statement.record, -1, -1


class Executor:
    """The delta-rule memory, checked against a shadow oracle at every action."""

    def __init__(self) -> None:
        self.machine = task.machine()
        self.delta = DeltaMemory(self.machine)
        self.oracle = self.machine.initial_state()
        self.state = self.delta.encode(self.oracle)
        self.mismatches = self.work = 0

    def execute(self, action: Action) -> Event:
        expected = self.machine.step(self.oracle, action)
        actual = self.delta.step(self.state, action)
        self.oracle, self.state = expected.next_state, actual.next_state
        if (
            actual.event != expected.event
            or self.delta.decode(self.state) != self.oracle
        ):
            self.mismatches += 1
        self.work += actual.work
        return actual.event


# ---------- training data ----------


@dataclass
class Data:
    texts: list[str]
    index: dict[str, int]
    mid: torch.Tensor  # (texts, d) residual stream after layer k
    top: torch.Tensor  # (texts, d) last layer
    # macro rows: one per statement of every training program
    row_text: torch.Tensor
    row_target: torch.Tensor  # (rows, 4)
    # get rows: one per training question
    get_text: torch.Tensor
    get_record: torch.Tensor
    get_state: torch.Tensor  # gold memory before the question, (n, 16, 8)
    get_answer: torch.Tensor
    # tick rows for arm O (E03's arm H)
    tick_text: torch.Tensor
    tick_codes: torch.Tensor
    tick_targets: torch.Tensor
    answer_text: torch.Tensor
    answer_codes: torch.Tensor


def build_data(
    cache: lmach.WindowCache,
    train: list[task.Program],
    extra_texts: list[str],
) -> Data:
    golds = [gold(p) for p in train]
    texts = sorted({*valid_texts(), *extra_texts})
    cache.add(texts)
    index = {t: i for i, t in enumerate(texts)}
    row_text, row_target = [], []
    get_text, get_record, get_state, get_answer = [], [], [], []
    tick_text, tick_codes, tick_targets = [], [], []
    answer_text, answer_codes = [], []
    for program, g in zip(train, golds, strict=True):
        state = lmach.initial_state()
        answers = program.answers()
        for i, s in enumerate(program.statements):
            target = macro_target(s)
            row_text.append(index[s.text])
            row_target.append(target)
            if s.kind == "get":
                get_text.append(index[s.text])
                get_record.append(s.record)
                get_state.append(state[0].clone())
                get_answer.append(answers[i])
                answer_text.append(index[s.text])
                answer_codes.append(g.codes[i][-1].indices())
            macro, record, source, value = (torch.tensor([max(x, 0)]) for x in target)
            macro = torch.tensor([target[0]])
            state, _ = lmach.apply(state, macro, record, source, value)
            n = len(g.actions[i]) + 1
            tick_text += [index[s.text]] * n
            tick_codes += [ticks.START.indices()] + [c.indices() for c in g.codes[i]]
            tick_targets += ticks.targets(g.actions[i])
    return Data(
        texts,
        index,
        cache.mid_features(texts),
        cache.top_features(texts),
        torch.tensor(row_text),
        torch.tensor(row_target),
        torch.tensor(get_text),
        torch.tensor(get_record),
        torch.stack(get_state),
        torch.tensor(get_answer),
        torch.tensor(tick_text),
        torch.tensor(tick_codes),
        torch.tensor(tick_targets),
        torch.tensor(answer_text),
        torch.tensor(answer_codes),
    )


# ---------- arm O ----------


@dataclass
class OutsideController:
    decide: TickHead
    answer: TickHead
    steps: int = 0


def tick_loss(logits: tuple[torch.Tensor, ...], targets: torch.Tensor) -> torch.Tensor:
    op = targets[:, 0]
    loss = F.cross_entropy(logits[0], op)
    for column, mask in (
        (1, op != ticks.OPERATIONS.index("end")),
        (2, op == ticks.OPERATIONS.index("write")),
    ):
        if mask.any():
            loss = loss + F.cross_entropy(logits[column][mask], targets[mask, column])
    return loss


def train_outside(data: Data, config: Config, seed: int) -> OutsideController:
    torch.manual_seed(seed)
    d = data.top.shape[1]
    c = OutsideController(
        TickHead(d, config.hidden, TICK_SIZES, ticks.CODE_SIZES),
        TickHead(d, config.hidden, (VALUES,), ticks.CODE_SIZES),
    )
    modules = [c.decide, c.answer]
    optimizer = torch.optim.Adam(
        [p for m in modules for p in m.parameters()], lr=config.learning_rate
    )
    answers = data.get_answer
    n_d, n_a = len(data.tick_targets), len(answers)
    for _ in range(config.epochs):
        order_d, order_a = torch.randperm(n_d), torch.randperm(n_a)
        batches = max(1, -(-max(n_d, n_a) // config.batch_size))
        for b in range(batches):
            idx = order_d[b::batches]
            loss = tick_loss(
                c.decide(data.top[data.tick_text[idx]], data.tick_codes[idx]),
                data.tick_targets[idx],
            )
            idx = order_a[b::batches]
            if len(idx):
                (logits,) = c.answer(
                    data.top[data.answer_text[idx]], data.answer_codes[idx]
                )
                loss = loss + F.cross_entropy(logits, answers[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            c.steps += 1
    for m in modules:
        m.eval()
    return c


# ---------- arms M, Mh, M-reject ----------


@dataclass
class LayerController:
    machine: lmach.LayerMachine
    answer: lmach.TopAnswer | lmach.ReadAnswer
    steps: int = 0


def macro_loss(logits: tuple[torch.Tensor, ...], target: torch.Tensor) -> torch.Tensor:
    macro, record, source, value = logits
    loss = F.cross_entropy(macro, target[:, 0])
    for column, lg, mask in (
        (1, record, target[:, 0] != lmach.REJECT),
        (2, source, target[:, 0] == lmach.COPY),
        (3, value, target[:, 0] == lmach.SET),
    ):
        if mask.any():
            loss = loss + F.cross_entropy(lg[mask], target[mask, column])
    return loss


def train_layer(
    arm: str,
    data: Data,
    cache: lmach.WindowCache,
    lm: FrozenLM,
    config: Config,
    seed: int,
    invalid: list[str],
) -> LayerController:
    torch.manual_seed(seed)
    d = data.mid.shape[1]
    residual = arm in ("M", "M-reject")
    machine = lmach.LayerMachine(d, config.hidden, arm == "M-reject", residual)
    if machine.residual is not None:
        torch.nn.init.normal_(machine.residual.weight, std=0.5 * data.mid.std().item())
    answer = (
        lmach.TopAnswer(d, config.hidden)
        if residual
        else lmach.ReadAnswer(d, config.hidden)
    )
    optimizer = torch.optim.Adam(
        [p for m in (machine, answer) for p in m.parameters()],
        lr=config.learning_rate,
    )
    row_text, row_target = data.row_text, data.row_target
    if arm == "M-reject":
        rng = random.Random(f"reject-rows:{seed}")
        n = round(INVALID_SHARE * len(row_text))
        picked = [data.index[rng.choice(invalid)] for _ in range(n)]
        row_text = torch.cat([row_text, torch.tensor(picked)])
        row_target = torch.cat(
            [row_target, torch.tensor([[lmach.REJECT, -1, -1, -1]] * n)]
        )
    ids = None
    if residual:
        per_record = [
            lmach.window_ids(lm, record_task.PREAMBLE, f" What is {n}?")
            for n in task.NAMES
        ]
        if len({len(x) for x in per_record}) != 1:
            raise RuntimeError("question windows differ in length")
        ids = torch.tensor(per_record)
    n_rows, n_gets = len(row_text), len(data.get_answer)
    controller = LayerController(machine, answer)
    for _ in range(config.epochs):
        order_r, order_g = torch.randperm(n_rows), torch.randperm(n_gets)
        batches = max(1, -(-max(n_rows, n_gets) // config.batch_size))
        for b in range(batches):
            idx = order_r[b::batches]
            loss = macro_loss(machine(data.mid[row_text[idx]]), row_target[idx])
            g = order_g[b::batches]
            if len(g):
                text = data.get_text[g]
                record_logits = machine(data.mid[text])[1]
                y = lmach.st_onehot(record_logits)
                read = torch.einsum("bvr,br->bv", data.get_state[g], y)
                if residual:
                    assert ids is not None and machine.residual is not None
                    top = lmach.inject_forward(
                        lm,
                        config.layer,
                        ids[data.get_record[g]],
                        machine.residual(read),
                    )
                    logits = answer(top)
                else:
                    logits = answer(data.top[text], read)
                loss = loss + F.cross_entropy(logits, data.get_answer[g])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            controller.steps += 1
    machine.eval()
    answer.eval()
    return controller


# ---------- evaluation ----------


class OutsideRunner:
    def __init__(self, c: OutsideController, data: Data) -> None:
        self.c, self.data = c, data
        self.decisions: dict[tuple[str, tuple[int, ...]], tuple[int, int, int]] = {}
        self.answers: dict[tuple[str, tuple[int, ...]], int] = {}

    @torch.no_grad()
    def decide(self, text: str, code: ticks.FeedbackCode) -> Action | None:
        key = (text, code.indices())
        if key not in self.decisions:
            f = self.data.top[self.data.index[text]][None]
            logits = self.c.decide(f, torch.tensor([code.indices()]))
            self.decisions[key] = tuple(x.argmax().item() for x in logits)  # type: ignore[assignment]
        return ticks.to_action(*self.decisions[key])

    @torch.no_grad()
    def answer(self, text: str, code: ticks.FeedbackCode) -> int:
        key = (text, code.indices())
        if key not in self.answers:
            f = self.data.top[self.data.index[text]][None]
            (logits,) = self.c.answer(f, torch.tensor([code.indices()]))
            self.answers[key] = logits.argmax().item()
        return self.answers[key]


def evaluate_outside(
    runner: OutsideRunner,
    cache: lmach.WindowCache,
    programs: list[task.Program],
    trace: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    questions = correct = statements = right = decisions = ticks_done = 0
    mismatches = work = positions = exhausted = 0
    longest = 0
    started = time.perf_counter()
    for p_index, program in enumerate(programs):
        g, answers, executor = gold(program), program.answers(), Executor()
        steps = []
        for i, s in enumerate(program.statements):
            code, acts = ticks.START, []
            while True:
                if len(acts) == ticks.TICK_BUDGET:
                    exhausted += 1
                    break
                decisions += 1
                action = runner.decide(s.text, code)
                if action is None:
                    break
                event = executor.execute(action)
                acts.append(action)
                code = ticks.feedback(action, event)
            statements += 1
            right += acts == g.actions[i]
            ticks_done += len(acts)
            positions += cache.length[s.text]
            longest = max(longest, cache.length[s.text])
            answer = runner.answer(s.text, code) if s.kind == "get" else None
            if s.kind == "get":
                questions += 1
                correct += answer == answers[i]
            if trace is not None and p_index < TRACE_PROGRAMS:
                steps.append(
                    {
                        "statement": s.text.strip(),
                        "actions": [lockstep.action_json(a) for a in acts],
                        "answer": answer,
                        "gold": answers[i],
                    }
                )
        mismatches += executor.mismatches
        work += executor.work
        if trace is not None and p_index < TRACE_PROGRAMS:
            trace.append({"arm": "O", "program": p_index, "steps": steps})
    return {
        "questions": questions,
        "task_accuracy": correct / questions,
        "statements": statements,
        "action_accuracy": right / statements,
        "executor_mismatches": mismatches,
        "layer_mismatches": None,
        "false_rejections": None,
        "decisions": decisions,
        "machine_actions": ticks_done,
        "budget_exhausted": exhausted,
        "machine_work": work,
        "model_positions": positions,
        "max_positions": longest,
        "eval_seconds": round(time.perf_counter() - started, 3),
    }


class LayerRunner:
    """Free-running execution of the in-layer machine from cached features."""

    def __init__(
        self,
        arm: str,
        c: LayerController,
        data: Data,
        cache: lmach.WindowCache,
        lm: FrozenLM,
        layer: int,
    ) -> None:
        self.arm, self.c, self.data, self.cache = arm, c, data, cache
        self.lm, self.layer = lm, layer
        with torch.no_grad():
            choices = torch.stack(c.machine.choose(data.mid), dim=1).tolist()
        self.table = {t: tuple(choices[i]) for t, i in data.index.items()}
        self.memo: dict[tuple[str, int], int] = {}

    @torch.no_grad()
    def answer(self, text: str, read: int | None) -> int:
        key = (text, -1 if read is None else read)
        if key not in self.memo:
            vec = torch.zeros(1, VALUES)
            if read is not None:
                vec[0, read] = 1.0
            if self.arm == "Mh":
                logits = self.c.answer(self.data.top[self.data.index[text]][None], vec)
            else:
                assert self.c.machine.residual is not None
                ids = torch.tensor(
                    lmach.window_ids(self.lm, record_task.PREAMBLE, text)
                )[None]
                top = lmach.inject_forward(
                    self.lm, self.layer, ids, self.c.machine.residual(vec)
                )
                logits = self.c.answer(top)
            self.memo[key] = logits.argmax().item()
        return self.memo[key]


def macro_actions(
    macro: int, record: int, source: int, value: int, stored: list[int]
) -> list[Action]:
    """The machine actions a macro stands for; ``stored`` is the memory before."""
    if macro == lmach.SET:
        return [Write(record, value)]
    if macro == lmach.GET:
        return [Read(record)]
    if macro == lmach.COPY:
        return [Read(source), Write(record, stored[source])]
    return []


@torch.no_grad()
def evaluate_layer(
    runner: LayerRunner,
    programs: list[task.Program],
    trace: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    cache = runner.cache
    questions = correct = statements = right = rejected = 0
    executor_mismatches = layer_mismatches = work = positions = ticks_done = 0
    longest = 0
    started = time.perf_counter()
    for p_index, program in enumerate(programs):
        g, answers, executor = gold(program), program.answers(), Executor()
        state = lmach.initial_state()
        steps = []
        for i, s in enumerate(program.statements):
            macro, record, source, value = runner.table[s.text]
            stored = lmach.decode(state)[0]
            acts = macro_actions(macro, record, source, value, stored)
            events = [executor.execute(a) for a in acts]
            state, seen = lmach.apply(
                state,
                torch.tensor([macro]),
                torch.tensor([record]),
                torch.tensor([source]),
                torch.tensor([value]),
            )
            if lmach.decode(state)[0] != list(executor.oracle.words):
                layer_mismatches += 1
            if macro == lmach.GET and seen.argmax().item() != events[0].value:
                layer_mismatches += 1
            statements += 1
            right += acts == g.actions[i]
            rejected += macro == lmach.REJECT
            ticks_done += len(acts)
            positions += cache.length[s.text]
            longest = max(longest, cache.length[s.text])
            answer = None
            if s.kind == "get":
                read = seen.argmax().item() if macro == lmach.GET else None
                answer = runner.answer(s.text, read)
                questions += 1
                correct += answer == answers[i]
            if trace is not None and p_index < TRACE_PROGRAMS:
                steps.append(
                    {
                        "statement": s.text.strip(),
                        "macro": lmach.MACROS[macro],
                        "actions": [lockstep.action_json(a) for a in acts],
                        "answer": answer,
                        "gold": answers[i],
                    }
                )
        executor_mismatches += executor.mismatches
        work += executor.work
        if trace is not None and p_index < TRACE_PROGRAMS:
            trace.append({"arm": runner.arm, "program": p_index, "steps": steps})
    return {
        "questions": questions,
        "task_accuracy": correct / questions,
        "statements": statements,
        "action_accuracy": right / statements,
        "executor_mismatches": executor_mismatches,
        "layer_mismatches": layer_mismatches,
        "false_rejections": rejected,
        "decisions": statements,
        "machine_actions": ticks_done,
        "budget_exhausted": 0,
        "machine_work": work,
        "model_positions": positions,
        "max_positions": longest,
        "eval_seconds": round(time.perf_counter() - started, 3),
    }


def validity(runner: LayerRunner, groups: dict[str, list[str]]) -> dict[str, float]:
    """Share of each group of statements that the layer rejects."""

    def rate(texts: list[str]) -> float:
        return mean([runner.table[t][0] == lmach.REJECT for t in texts])

    result = {name: rate(texts) for name, texts in groups.items()}
    unseen = [*groups["unseen_record"], *groups["unseen_values"]]
    result["unseen_pooled"] = rate(unseen)
    result["false_rejection_valid"] = rate(valid_texts())
    result["false_rejection_unseen_valid"] = rate(held_out_texts())
    return result


# ---------- predictions ----------


def evaluate_predictions(
    rows: list[dict[str, Any]],
    validities: list[dict[str, float]],
    trained: dict[str, Any],
) -> dict[str, Any]:
    def by(arm: str, split: str, key: str) -> dict[int, Any]:
        return {
            r["seed"]: r[key] for r in rows if r["arm"] == arm and r["split"] == split
        }

    seeds = sorted(by("O", TEST_SPLITS[0], "task_accuracy"))

    def diff(a: str, b: str, split: str) -> float:
        x, y = by(a, split, "task_accuracy"), by(b, split, "task_accuracy")
        return mean([x[s] - y[s] for s in seeds])

    one, two = [], []
    for split in TEST_SPLITS:
        questions = by("M", split, "questions")[seeds[0]]
        if split in LENGTH_SPLITS:
            d = diff("M", "O", split)
            one.append(
                {
                    "split": split,
                    "mean_m_minus_o": d,
                    "minimum": -NON_INFERIORITY,
                    "met": d >= -NON_INFERIORITY,
                }
            )
        d = diff("M", "Mh", split)
        two.append(
            {
                "split": split,
                "mean_m_minus_mh": d,
                "bound": bound(questions),
                "met": abs(d) <= bound(questions),
            }
        )
    layer_mismatches = sum(r["layer_mismatches"] or 0 for r in rows if r["arm"] == "M")

    def v(key: str) -> float:
        return mean([x[key] for x in validities])

    trained_test, false_rate = v("trained_form_test"), v("false_rejection_valid")
    unseen = v("unseen_pooled")
    o_short = mean(list(by("O", TEST_SPLITS[0], "task_accuracy").values()))
    exec_o = sum(r["executor_mismatches"] for r in rows if r["arm"] == "O")
    exec_mh = sum(r["executor_mismatches"] for r in rows if r["arm"] == "Mh")
    counts = [trained[a]["trainable_parameters"] for a in ("O", "M", "Mh")]
    return {
        "1_m_not_worse_than_o": {"results": one, "met": all(r["met"] for r in one)},
        "2_m_and_mh_within_bound": {
            "results": two,
            "met": all(r["met"] for r in two),
        },
        "3_in_layer_state_matches_oracle": {
            "layer_mismatches": layer_mismatches,
            "met": layer_mismatches == 0,
        },
        "4_reject_learned_for_trained_forms": {
            "trained_form_test_rejection": trained_test,
            "minimum": REJECT_MIN,
            "false_rejection_valid": false_rate,
            "maximum": FALSE_REJECT_MAX,
            "met": trained_test >= REJECT_MIN and false_rate <= FALSE_REJECT_MAX,
        },
        "5_reject_does_not_generalize": {
            "unseen_pooled_rejection": unseen,
            "unseen_record": v("unseen_record"),
            "unseen_values": v("unseen_values"),
            "threshold": REJECT_MIN,
            "met": unseen < REJECT_MIN,
        },
        "sanity": {
            "o_task_accuracy_2_8": o_short,
            "o_minimum": O_SANITY,
            "executor_mismatches_o": exec_o,
            "executor_mismatches_mh": exec_mh,
            "parameter_spread": (max(counts) - min(counts)) / max(counts),
            "parameter_tolerance": PARAMETER_TOLERANCE,
            "ok": o_short >= O_SANITY
            and exec_o == 0
            and exec_mh == 0
            and (max(counts) - min(counts)) / max(counts) <= PARAMETER_TOLERANCE,
        },
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["arm"], row["split"]), []).append(row)
    summary = []
    for (arm, split), group in sorted(groups.items()):
        entry: dict[str, Any] = {"arm": arm, "split": split, "seeds": len(group)}
        for key in ("task_accuracy", "action_accuracy"):
            values = [r[key] for r in group]
            entry[key] = {
                "mean": mean(values),
                "min": min(values),
                "max": max(values),
            }
        summary.append(entry)
    return summary


# ---------- run ----------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="E10: a delta-rule machine inside a layer of a frozen model.",
    )
    parser.add_argument("--model", default=e02.MODEL)
    parser.add_argument("--revision", default=e02.REVISION)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=2000, help="training programs")
    parser.add_argument("--test", type=int, default=300, help="2-8 statement programs")
    parser.add_argument(
        "--long-test", type=int, default=100, help="programs per long split"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="decoder layer k that holds the machine (0: 15 of 30, 1 for tiny-random)",
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="device for the frozen model")
    parser.add_argument("--out", default="runs")
    args = parser.parse_args(argv)
    if min(args.train, args.test, args.long_test, args.epochs) < 1:
        parser.error("sizes and epochs must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    if args.threads:
        torch.set_num_threads(args.threads)
    tiny = args.model == "tiny-random"
    config = Config(
        model=args.model,
        revision=None if tiny else args.revision,
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        data_seed=args.data_seed,
        train=args.train,
        test=args.test,
        long_test=args.long_test,
        epochs=args.epochs,
        hidden=args.hidden,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        layer=args.layer or (1 if tiny else 15),
    )
    run_dir, source, root = lockstep.start(args.out, "e10")
    data_splits = splits(config)
    header = lockstep.run_header(
        experiment=EXPERIMENT,
        module=MODULE,
        argv=argv,
        run_dir=run_dir,
        source=source,
        root=root,
        config=asdict(config),
        device=args.device,
    )
    if args.device.startswith("cuda"):
        header["environment"] |= {
            "gpu": torch.cuda.get_device_name(args.device),
            "cuda": torch.version.cuda,
        }
    rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    trained: dict[str, Any] = {}
    validities: list[dict[str, Any]] = []
    predictions = None
    error = None
    lm = None
    layers = None
    try:
        lm = FrozenLM.load(config.model, config.revision, args.device)
        layers = len(lm.model.layers)
        if not 1 <= config.layer <= layers:
            raise SystemExit(f"--layer must be in 1..{layers}")
        cache = lmach.WindowCache(lm, config.layer, record_task.PREAMBLE)
        groups = invalid_sets(config.data_seed)
        extra = [t for texts in groups.values() for t in texts]
        for _, programs in data_splits.values():
            extra += [s.text for p in programs for s in p.statements]
        data = build_data(cache, data_splits["train"][1], extra)
        for seed in config.seeds:
            started = time.perf_counter()
            runners: dict[str, Any] = {}
            outside = train_outside(data, config, seed)
            runners["O"] = OutsideRunner(outside, data)
            trained["O"] = {
                "trainable_parameters": parameter_count(outside.decide, outside.answer),
                "training_steps": outside.steps,
            }
            for arm in ("M", "Mh", "M-reject"):
                c = train_layer(
                    arm,
                    data,
                    cache,
                    lm,
                    config,
                    seed,
                    groups["trained_form_train"],
                )
                runners[arm] = LayerRunner(arm, c, data, cache, lm, config.layer)
                trained[arm] = {
                    "trainable_parameters": parameter_count(c.machine, c.answer),
                    "training_steps": c.steps,
                }
            validities.append(
                {"seed": seed, **validity(runners["M-reject"], groups)}
                | {"train_seconds": round(time.perf_counter() - started, 1)}
            )
            print(
                f"seed {seed} reject "
                + " ".join(
                    f"{k}={v:.3f}" for k, v in validities[-1].items() if k != "seed"
                ),
                flush=True,
            )
            for arm in ARMS:
                for name in TEST_SPLITS:
                    local: list[dict[str, Any]] = []
                    first = seed == config.seeds[0]
                    programs = data_splits[name][1]
                    if arm == "O":
                        result = evaluate_outside(
                            runners[arm], cache, programs, local if first else None
                        )
                    else:
                        result = evaluate_layer(
                            runners[arm], programs, local if first else None
                        )
                    if name == TEST_SPLITS[0]:
                        trace += local
                    rows.append({"arm": arm, "split": name, "seed": seed, **result})
                    print(
                        f"seed {seed} arm {arm:<8} {name:<21} "
                        f"task {result['task_accuracy']:.3f} "
                        f"actions {result['action_accuracy']:.3f}",
                        flush=True,
                    )
        predictions = evaluate_predictions(rows, validities, trained)
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)

    mismatches = sum(
        (r["executor_mismatches"] or 0) + (r["layer_mismatches"] or 0) for r in rows
    )
    status = "error" if error else ("completed" if mismatches == 0 else "mismatch")
    records.write_json(
        run_dir / "run.json",
        header
        | {
            "seeds": {"heads": list(config.seeds), "data": config.data_seed},
            "model": {
                "name": config.model,
                "revision": config.revision,
                "parameters": lm.parameters if lm else None,
                "hidden_size": lm.hidden_size if lm else None,
                "layers": layers,
                "machine_layer": config.layer,
                "frozen": True,
            },
            "task": {
                "machine": "memory/v1, capacity 8, 4-bit words",
                "programs": "E03 copy programs: set 0.4, copy 0.3, question 0.3",
                "held_out_pairs": sorted(HELD_OUT),
                "splits": {
                    name: meta | {"sha256": e02.digest(programs)}
                    for name, (meta, programs) in data_splits.items()
                },
                "invalid_statements": {
                    name: len(texts) for name, texts in invalid_sets(0).items()
                },
            },
            "arms": {
                "O": "memory outside; tick heads on the last layer with typed "
                "feedback (E03 arm H on windows)",
                "M": "delta-rule memory in layer k; the read is added to the "
                "residual stream and the answer head reads the last layer",
                "Mh": "as M; the read goes to the answer head as an embedded value",
                "M-reject": "M with a reject macro, trained on invalid statements",
            },
            "windows": "the preamble followed by one statement; features are cached "
            "by statement text because they do not depend on position",
            "trained": trained,
        },
    )
    with (run_dir / "trace.jsonl").open("w") as handle:
        for line in trace:
            handle.write(json.dumps(line) + "\n")
    records.write_json(
        run_dir / "metrics.json",
        {
            "schema_version": records.SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "status": status,
            "finished_at": records.now(),
            "acceptance": (
                "pre-registered predictions in EXPERIMENTS.md evaluated as written; "
                "zero mismatches for the delta-rule executor and the in-layer memory"
            ),
            "executor_mismatches": sum(r["executor_mismatches"] or 0 for r in rows),
            "layer_mismatches": sum(r["layer_mismatches"] or 0 for r in rows),
            "peak_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "summary": summarize(rows),
            "runs": rows,
            "validity": validities,
            "predictions": predictions,
            "not_measured": [
                "layers other than k, or several machine layers",
                "gated or non-one-hot addressing",
                "other model sizes, fine-tuned or unfrozen baselines",
                "model passes for O: it reuses statement features as E03's H does",
            ],
            "error": error,
        },
    )
    print(f"status: {status}\nrecord: {run_dir}")
    return {"completed": 0, "mismatch": 1}.get(status, 2)


if __name__ == "__main__":
    sys.exit(main())
