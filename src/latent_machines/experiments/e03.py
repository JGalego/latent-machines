"""E03: latent ticks and explicit events on record programs with copies.

``Copy x to y`` takes two machine ticks, a read and then a write of the value
read, so the read's reply must reach the controller's next decision. A
frozen language model reads each program and trained heads choose one tick
at a time (``write``, ``read``, or ``end``). Four arms share the task, the
model, the executor, and the tick budget, and differ only in how a tick's
action and reply (its feedback code) reach the next decision:

- T, token feedback: the request and reply are appended as text, as in E02's
  arm B, and the model reads them.
- L, latent feedback: the code is embedded by a trained adaptor into one input
  vector at the model's next position. It re-enters every layer of the next
  tick and stays in the key-value cache for later statements.
- H, head feedback: the code is embedded and given to the trained heads
  only; the model never sees it. This extends E02's typed arm C to ticks.
- N, native loop: the heads issue one macro per statement (``write``,
  ``read``, or ``copy``) and a conventional in-process loop runs its ticks,
  passing the value read to the write itself.

Every arm executes on E00's delta-rule memory with a shadow oracle that
counts executor mismatches. Evaluation is free-running. For a sample of
programs, each arm is also snapshotted between the two ticks of a copy:
machine, controller history, and pending work go through JSON, a new
controller replays the history, and the rest of the run must match an
uninterrupted run exactly.

    uv run --locked python -m latent_machines.experiments.e03
    uv run --locked python -m latent_machines.experiments.e03 --model tiny-random

Exit status: 0 when the run completes with no executor mismatch and every
snapshot resumes identically, 1 otherwise, 2 if the run itself failed.
"""

import argparse
import hashlib
import json
import resource
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from latent_machines.backends.delta import DeltaMemory
from latent_machines.controllers.frozen_lm import FrozenLM
from latent_machines.controllers.heads import (
    FeedbackEmbedding,
    TickHead,
    parameter_count,
)
from latent_machines.experiments import lockstep, records
from latent_machines.interfaces import memory as text
from latent_machines.interfaces import ticks
from latent_machines.machines.events import Event, EventKind
from latent_machines.machines.memory import Action, Read, Write
from latent_machines.tasks import copies as task

EXPERIMENT = "E03"
MODULE = "latent_machines.experiments.e03"
MODEL = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
ARMS = ("T", "L", "H", "N")
ARM_NAMES = {
    "T": "token feedback",
    "L": "latent feedback",
    "H": "head feedback",
    "N": "native loop",
}
VALUES = task.MAX_VALUE + 1
TICK_SIZES = (len(ticks.OPERATIONS), len(task.NAMES), VALUES)
MACRO_SIZES = (len(ticks.MACROS), len(task.NAMES), len(task.NAMES), VALUES)
TRACE_PROGRAMS = 5


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
    latent_batch: int
    snapshot_programs: int
    tick_budget: int = ticks.TICK_BUDGET


def splits(config: Config) -> list[task.Split]:
    return [
        task.Split("train", config.train, 2, 8),
        task.Split("test_in_distribution", config.test, 2, 8),
        task.Split("test_length_32", config.long_test, 32, 32),
        task.Split("test_length_64", config.long_test, 64, 64),
    ]


def digest(programs: list[task.Program]) -> str:
    joined = "\n".join("".join(s.text for s in p.statements) for p in programs)
    return hashlib.sha256(joined.encode()).hexdigest()


@dataclass
class Gold:
    """A program's gold actions, events, and feedback codes per statement."""

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


# ---------- training data ----------


@dataclass
class Cached:
    """Training rows from features computed once, for arms T, H, and N."""

    decision_features: torch.Tensor
    decision_codes: torch.Tensor | None
    decision_targets: torch.Tensor  # (rows, len(sizes))
    answer_features: torch.Tensor
    answer_codes: torch.Tensor | None
    answers: torch.Tensor


def plain_features(lm: FrozenLM, program: task.Program) -> torch.Tensor:
    """Feature at the end of each statement, with no feedback in the context."""
    chunks = [task.PREAMBLE, *(s.text for s in program.statements)]
    return lm.chunk_features(chunks)[1:]


def transcript(program: task.Program, g: Gold) -> list[str]:
    """Arm T's context under gold control: statements, requests, and replies."""
    chunks = [task.PREAMBLE]
    for statement, actions, events in zip(
        program.statements, g.actions, g.events, strict=True
    ):
        chunks.append(statement.text)
        for action, event in zip(actions, events, strict=True):
            chunks += [text.text_request(action), text.text_reply(event)]
    return chunks


def cached_examples(
    arm: str, lm: FrozenLM, programs: list[task.Program], golds: list[Gold]
) -> Cached:
    d_rows, d_codes, d_targets, a_rows, a_codes, answers = [], [], [], [], [], []
    for program, g in zip(programs, golds, strict=True):
        if arm == "T":
            features = lm.chunk_features(transcript(program, g))
            cursor = 1
        else:
            plain = plain_features(lm, program)
        for i, statement in enumerate(program.statements):
            actions, codes = g.actions[i], g.codes[i]
            if arm == "T":
                # Statement chunk, then a request and a reply chunk per action.
                at = [cursor + 2 * j for j in range(len(actions) + 1)]
                cursor += 1 + 2 * len(actions)
                d_rows += [features[k] for k in at]
                d_targets += ticks.targets(actions)
                if statement.kind == "get":
                    a_rows.append(features[at[-1]])
            elif arm == "H":
                d_rows += [plain[i]] * (len(actions) + 1)
                d_codes += [ticks.START.indices()] + [c.indices() for c in codes]
                d_targets += ticks.targets(actions)
                if statement.kind == "get":
                    a_rows.append(plain[i])
                    a_codes.append(codes[-1].indices())
            else:  # N
                macro = ticks.macro_target(actions)
                d_rows.append(plain[i])
                d_targets.append(
                    (macro.operation, macro.record, macro.source, macro.value)
                )
                if statement.kind == "get":
                    a_rows.append(plain[i])
                    a_codes.append(codes[-1].indices())
            if statement.kind == "get":
                answers.append(g.events[i][-1].value)
    return Cached(
        torch.stack(d_rows),
        torch.tensor(d_codes) if d_codes else None,
        torch.tensor(d_targets),
        torch.stack(a_rows),
        torch.tensor(a_codes) if a_codes else None,
        torch.tensor(answers),
    )


@dataclass
class LatentProgram:
    """Arm L's sequence under gold control, with feedback slots left empty."""

    base: torch.Tensor  # (positions, d) token embeddings, zeros at code slots
    code_positions: torch.Tensor
    codes: torch.Tensor  # (slots, 4)
    decision_positions: torch.Tensor
    decision_targets: torch.Tensor
    answer_positions: torch.Tensor
    answers: torch.Tensor


@torch.no_grad()
def latent_program(lm: FrozenLM, program: task.Program, g: Gold) -> LatentProgram:
    parts: list[torch.Tensor] = []
    length = 0
    code_pos, codes, d_pos, d_targets, a_pos, answers = [], [], [], [], [], []

    def add_tokens(chunk: str) -> int:
        nonlocal length
        (ids,) = lm.tokens([chunk])
        parts.append(lm.token_embeddings(ids))
        length += len(ids)
        return length - 1

    def add_slot(code: ticks.FeedbackCode) -> int:
        nonlocal length
        parts.append(torch.zeros(1, lm.hidden_size, device=lm.device))
        code_pos.append(length)
        codes.append(code.indices())
        length += 1
        return length - 1

    add_tokens(task.PREAMBLE)
    for i, statement in enumerate(program.statements):
        at = [add_tokens(statement.text)]
        at += [add_slot(code) for code in g.codes[i]]
        d_pos += at
        d_targets += ticks.targets(g.actions[i])
        if statement.kind == "get":
            a_pos.append(at[-1])
            answers.append(g.events[i][-1].value)
    return LatentProgram(
        torch.cat(parts),
        torch.tensor(code_pos, dtype=torch.long, device=lm.device),
        torch.tensor(codes, dtype=torch.long, device=lm.device),
        torch.tensor(d_pos, dtype=torch.long, device=lm.device),
        torch.tensor(d_targets, dtype=torch.long, device=lm.device),
        torch.tensor(a_pos, dtype=torch.long, device=lm.device),
        torch.tensor(answers, dtype=torch.long, device=lm.device),
    )


# ---------- heads and losses ----------


@dataclass
class Controller:
    arm: str
    decide: TickHead
    answer: TickHead
    adaptor: FeedbackEmbedding | None = None
    steps: int = 0
    seconds: float = 0.0

    def modules(self) -> list[torch.nn.Module]:
        return [m for m in (self.decide, self.answer, self.adaptor) if m is not None]


def new_controller(arm: str, lm: FrozenLM, config: Config) -> Controller:
    d, hidden = lm.hidden_size, config.hidden
    codes = ticks.CODE_SIZES if arm in ("H", "N") else None
    sizes = MACRO_SIZES if arm == "N" else TICK_SIZES
    decide = TickHead(d, hidden, sizes, ticks.CODE_SIZES if arm == "H" else None)
    answer = TickHead(d, hidden, (VALUES,), codes)
    adaptor = None
    if arm == "L":
        std = lm.model.get_input_embeddings().weight.std().item()
        adaptor = FeedbackEmbedding(ticks.CODE_SIZES, d, std=std / 2).to(lm.device)
    return Controller(arm, decide, answer, adaptor)


def decision_loss(
    logits: tuple[torch.Tensor, ...], targets: torch.Tensor, arm: str
) -> torch.Tensor:
    op = targets[:, 0]
    loss = F.cross_entropy(logits[0], op)
    if arm == "N":
        masks = {
            1: torch.ones_like(op, dtype=torch.bool),
            2: op == ticks.MACROS.index("copy"),
            3: op == ticks.MACROS.index("write"),
        }
    else:
        masks = {
            1: op != ticks.OPERATIONS.index("end"),
            2: op == ticks.OPERATIONS.index("write"),
        }
    for column, mask in masks.items():
        if mask.any():
            loss = loss + F.cross_entropy(logits[column][mask], targets[mask, column])
    return loss


def train_cached(c: Controller, data: Cached, config: Config) -> None:
    params = [p for m in c.modules() for p in m.parameters()]
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    n_d, n_a = len(data.decision_targets), len(data.answers)
    started = time.perf_counter()
    for _ in range(config.epochs):
        order_d, order_a = torch.randperm(n_d), torch.randperm(n_a)
        batches = max(1, -(-max(n_d, n_a) // config.batch_size))
        for b in range(batches):
            idx = order_d[b::batches]
            codes = (
                data.decision_codes[idx] if data.decision_codes is not None else None
            )
            loss = decision_loss(
                c.decide(data.decision_features[idx], codes),
                data.decision_targets[idx],
                c.arm,
            )
            idx = order_a[b::batches]
            if len(idx):
                codes = (
                    data.answer_codes[idx] if data.answer_codes is not None else None
                )
                (logits,) = c.answer(data.answer_features[idx], codes)
                loss = loss + F.cross_entropy(logits, data.answers[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            c.steps += 1
    c.seconds = time.perf_counter() - started


def train_latent(
    c: Controller, lm: FrozenLM, data: list[LatentProgram], config: Config
) -> None:
    """Train the adaptor and heads through the frozen model, end to end."""
    assert c.adaptor is not None
    c.decide.to(lm.device)
    c.answer.to(lm.device)
    params = [p for m in c.modules() for p in m.parameters()]
    optimizer = torch.optim.Adam(params, lr=config.learning_rate)
    started = time.perf_counter()
    for _ in range(config.epochs):
        order = torch.randperm(len(data)).tolist()
        for b in range(0, len(order), config.latent_batch):
            batch = [data[i] for i in order[b : b + config.latent_batch]]
            width = max(len(p.base) for p in batch)
            embeds, mask = [], []
            for p in batch:
                filled = p.base.index_put((p.code_positions,), c.adaptor(p.codes))
                pad = width - len(p.base)
                embeds.append(F.pad(filled, (0, 0, 0, pad)))
                mask.append(
                    torch.tensor([1] * len(p.base) + [0] * pad, device=lm.device)
                )
            hidden = lm.model(
                inputs_embeds=torch.stack(embeds), attention_mask=torch.stack(mask)
            ).last_hidden_state
            rows = torch.cat(
                [hidden[k, p.decision_positions] for k, p in enumerate(batch)]
            )
            targets = torch.cat([p.decision_targets for p in batch])
            loss = decision_loss(c.decide(rows), targets, "L")
            answer_rows = torch.cat(
                [hidden[k, p.answer_positions] for k, p in enumerate(batch)]
            )
            if len(answer_rows):
                answers = torch.cat([p.answers for p in batch])
                (logits,) = c.answer(answer_rows)
                loss = loss + F.cross_entropy(logits, answers)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            c.steps += 1
    c.seconds = time.perf_counter() - started
    for m in c.modules():
        m.to("cpu")


# ---------- executor ----------


class Executor:
    """The delta-rule memory, checked against a shadow oracle at every tick."""

    def __init__(self, snapshot: dict[str, Any] | None = None) -> None:
        self.machine = task.machine()
        self.delta = DeltaMemory(self.machine)
        self.oracle = (
            self.machine.restore(snapshot) if snapshot else self.machine.initial_state()
        )
        self.state = self.delta.encode(self.oracle)
        self.mismatches = self.work = self.faults = 0
        self.seconds = 0.0  # delta-rule step only, without the shadow oracle

    def execute(self, action: Action) -> Event:
        expected = self.machine.step(self.oracle, action)
        started = time.perf_counter()
        actual = self.delta.step(self.state, action)
        self.seconds += time.perf_counter() - started
        self.oracle, self.state = expected.next_state, actual.next_state
        if (
            actual.event != expected.event
            or self.delta.decode(self.state) != self.oracle
        ):
            self.mismatches += 1
        self.work += actual.work
        self.faults += actual.event.kind is EventKind.FAULT
        return actual.event

    def snapshot(self) -> dict[str, Any]:
        return self.machine.snapshot(self.delta.decode(self.state))


# ---------- controllers at evaluation ----------


def argmaxes(logits: tuple[torch.Tensor, ...]) -> list[int]:
    return [int(x.argmax()) for x in logits]


class TickController:
    """Arms T, L, and H: one decision per tick from the current features."""

    def __init__(self, c: Controller, lm: FrozenLM, plain: torch.Tensor) -> None:
        self.c, self.lm, self.plain = c, lm, plain
        self.session = lm.session() if c.arm in ("T", "L") else None
        self.history: list[list[Any]] = []
        self.features = torch.zeros(lm.hidden_size)
        self.code = ticks.START
        self.decision_features: list[torch.Tensor] = []

    def _text(self, chunk: str) -> None:
        assert self.session is not None
        self.features = self.session.feed(chunk)
        self.history.append(["text", chunk])

    def start(self) -> None:
        if self.session is not None:
            self._text(task.PREAMBLE)

    def read_statement(self, index: int, statement: task.Statement) -> None:
        self.code = ticks.START
        if self.session is not None:
            self._text(statement.text)
        else:
            self.features = self.plain[index]
            self.history.append(["statement", index])

    def _codes(self) -> torch.Tensor | None:
        return torch.tensor(self.code.indices()) if self.c.arm == "H" else None

    def decide(self) -> Action | None:
        self.decision_features.append(self.features)
        op, record, value = argmaxes(self.c.decide(self.features, self._codes()))
        return ticks.to_action(op, record, value)

    def observe(self, action: Action, event: Event) -> None:
        self.code = ticks.feedback(action, event)
        if self.c.arm == "T":
            self._text(text.text_request(action))
            self._text(text.text_reply(event))
        elif self.c.arm == "L":
            assert self.session is not None and self.c.adaptor is not None
            vector = self.c.adaptor(torch.tensor(self.code.indices()))
            self.features = self.session.feed_vector(vector)
            self.history.append(["code", list(self.code.indices())])
        else:
            self.history.append(["code", list(self.code.indices())])

    def answer(self) -> int:
        (logits,) = self.c.answer(self.features, self._codes())
        return int(logits.argmax())

    def positions(self) -> int:
        return self.session.tokens + self.session.vectors if self.session else 0

    def snapshot(self) -> dict[str, Any]:
        return {"history": self.history}

    def restore(self, data: dict[str, Any]) -> None:
        """Rebuild this controller's state by replaying a snapshot's history."""
        for kind, item in data["history"]:
            if kind == "text":
                self._text(item)
            elif kind == "statement":
                self.features, self.code = self.plain[item], ticks.START
                self.history.append(["statement", item])
            else:
                self.code = ticks.FeedbackCode(*item)
                if self.c.arm == "L":
                    assert self.session is not None and self.c.adaptor is not None
                    vector = self.c.adaptor(torch.tensor(item))
                    self.features = self.session.feed_vector(vector)
                self.history.append(["code", item])


@dataclass
class Outcome:
    """What one program's free-running execution produced."""

    actions: list[list[Action]] = field(default_factory=list)
    events: list[list[Event]] = field(default_factory=list)
    answers: list[int | None] = field(default_factory=list)
    exhausted: int = 0
    decisions: int = 0
    positions: int = 0
    feedback_positions: int = 0
    mismatches: int = 0
    work: int = 0
    faults: int = 0
    features: list[torch.Tensor] = field(default_factory=list)
    snapshot: dict[str, Any] | None = None
    executor_seconds: float = 0.0

    def comparable(self) -> str:
        return json.dumps(
            [
                [[lockstep.action_json(a) for a in acts] for acts in self.actions],
                [[lockstep.event_json(e) for e in evs] for evs in self.events],
                self.answers,
                self.exhausted,
            ]
        )


def first_copy(program: task.Program) -> int | None:
    for i, s in enumerate(program.statements):
        if s.kind == "copy":
            return i
    return None


@torch.no_grad()
def run_program(
    c: Controller,
    lm: FrozenLM,
    program: task.Program,
    plain: torch.Tensor,
    snapshot_at: int | None = None,
) -> Outcome:
    """Run one program free-running.

    With ``snapshot_at`` set to a copy statement's index, the run is
    snapshotted after that statement's first tick, everything goes through
    JSON, and execution continues on a new executor and controller restored
    from the snapshot.
    """
    if c.arm == "N":
        return run_native(c, program, plain, snapshot_at)
    out, executor = Outcome(), Executor()
    ctl = TickController(c, lm, plain)
    ctl.start()
    for i, statement in enumerate(program.statements):
        ctl.read_statement(i, statement)
        acts: list[Action] = []
        evs: list[Event] = []
        while True:
            if len(acts) == ticks.TICK_BUDGET:
                out.exhausted += 1
                break
            out.decisions += 1
            action = ctl.decide()
            if action is None:
                break
            event = executor.execute(action)
            acts.append(action)
            evs.append(event)
            ctl.observe(action, event)
            if snapshot_at == i and len(acts) == 1:
                snap = json.loads(
                    json.dumps(
                        {
                            "machine": executor.snapshot(),
                            "controller": ctl.snapshot(),
                            "statement": i,
                            "ticks_done": len(acts),
                        }
                    )
                )
                out.snapshot = snap
                counts = (executor.mismatches, executor.work, executor.faults)
                executor = Executor(snap["machine"])
                executor.mismatches, executor.work, executor.faults = counts
                features_before = ctl.decision_features
                ctl = TickController(c, lm, plain)
                ctl.restore(snap["controller"])
                ctl.decision_features = features_before
        out.actions.append(acts)
        out.events.append(evs)
        out.answers.append(ctl.answer() if statement.kind == "get" else None)
    out.mismatches, out.work, out.faults = (
        executor.mismatches,
        executor.work,
        executor.faults,
    )
    out.executor_seconds = executor.seconds
    out.features = ctl.decision_features
    out.positions = ctl.positions()
    return out


def run_native(
    c: Controller, program: task.Program, plain: torch.Tensor, snapshot_at: int | None
) -> Outcome:
    out, executor = Outcome(), Executor()
    for i, statement in enumerate(program.statements):
        out.decisions += 1
        out.features.append(plain[i])
        op, record, source, value = argmaxes(c.decide(plain[i]))
        loop = ticks.NativeLoop(ticks.Macro(op, record, source, value))
        acts: list[Action] = []
        evs: list[Event] = []
        while (action := loop.next_action()) is not None:
            event = executor.execute(action)
            acts.append(action)
            evs.append(event)
            loop.observe(event)
            if snapshot_at == i and len(acts) == 1:
                snap = json.loads(
                    json.dumps(
                        {
                            "machine": executor.snapshot(),
                            "loop": loop.snapshot(),
                            "statement": i,
                            "ticks_done": 1,
                        }
                    )
                )
                out.snapshot = snap
                counts = (executor.mismatches, executor.work, executor.faults)
                executor = Executor(snap["machine"])
                executor.mismatches, executor.work, executor.faults = counts
                loop = ticks.NativeLoop.restore(snap["loop"])
        out.actions.append(acts)
        out.events.append(evs)
        answer = None
        if statement.kind == "get":
            code = ticks.feedback(acts[-1], evs[-1]) if acts else ticks.START
            (logits,) = c.answer(plain[i], torch.tensor(code.indices()))
            answer = int(logits.argmax())
        out.answers.append(answer)
    out.mismatches, out.work, out.faults = (
        executor.mismatches,
        executor.work,
        executor.faults,
    )
    out.executor_seconds = executor.seconds
    return out


def copy_fidelity(program: task.Program, g: Gold, out: Outcome) -> tuple[int, int]:
    """Copies whose read and write targets were right, and of those, how many
    wrote exactly the value the read returned."""
    eligible = faithful = 0
    for statement, gold_acts, acts, evs in zip(
        program.statements, g.actions, out.actions, out.events, strict=True
    ):
        if statement.kind != "copy" or len(acts) < 2:
            continue
        read, write = acts[0], acts[1]
        if not (
            isinstance(read, Read)
            and isinstance(write, Write)
            and read == gold_acts[0]
            and write.address == statement.record
        ):
            continue
        eligible += 1
        faithful += write.value == evs[0].value
    return eligible, faithful


def evaluate(
    c: Controller,
    lm: FrozenLM,
    programs: list[task.Program],
    golds: list[Gold],
    plain: list[torch.Tensor],
    trace: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    questions = correct = statements = right = copies = eligible = faithful = 0
    exhausted = decisions = positions = mismatches = work = faults = 0
    plain_positions = 0
    executor_seconds = 0.0
    started = time.perf_counter()
    for p_index, (program, g) in enumerate(zip(programs, golds, strict=True)):
        out = run_program(c, lm, program, plain[p_index])
        answers = program.answers()
        for i, statement in enumerate(program.statements):
            statements += 1
            right += out.actions[i] == g.actions[i]
            copies += statement.kind == "copy"
            if statement.kind == "get":
                questions += 1
                correct += out.answers[i] == answers[i]
        e, f = copy_fidelity(program, g, out)
        eligible, faithful = eligible + e, faithful + f
        exhausted += out.exhausted
        decisions += out.decisions
        mismatches += out.mismatches
        work += out.work
        faults += out.faults
        executor_seconds += out.executor_seconds
        plain_tokens = sum(
            len(ids)
            for ids in lm.tokens([task.PREAMBLE, *(s.text for s in program.statements)])
        )
        plain_positions += plain_tokens
        positions += out.positions if c.arm in ("T", "L") else plain_tokens
        if trace is not None and p_index < TRACE_PROGRAMS:
            trace.append(
                {
                    "arm": c.arm,
                    "program": p_index,
                    "statements": [s.text.strip() for s in program.statements],
                    "actions": [
                        [lockstep.action_json(a) for a in acts] for acts in out.actions
                    ],
                    "events": [
                        [lockstep.event_json(e) for e in evs] for evs in out.events
                    ],
                    "answers": out.answers,
                    "gold_answers": answers,
                }
            )
    return {
        "questions": questions,
        "task_accuracy": correct / questions if questions else None,
        "statements": statements,
        "action_accuracy": right / statements if statements else None,
        "copies": copies,
        "copy_reads_and_targets_right": eligible,
        "copy_fidelity": faithful / eligible if eligible else None,
        "budget_exhausted": exhausted,
        "controller_decisions": decisions,
        "machine_work": work,
        "faults": faults,
        "executor_mismatches": mismatches,
        "lm_positions": positions,
        "feedback_positions": positions - plain_positions,
        "eval_seconds": round(time.perf_counter() - started, 3),
        "executor_seconds": round(executor_seconds, 3),
    }


def snapshot_check(
    c: Controller,
    lm: FrozenLM,
    programs: list[task.Program],
    plain: list[torch.Tensor],
    limit: int,
) -> dict[str, Any]:
    """Snapshot mid-copy and resume; the run must equal an uninterrupted one."""
    agree = differ = 0
    max_diff = 0.0
    example = None
    for p_index, program in enumerate(programs):
        at = first_copy(program)
        if at is None:
            continue
        whole = run_program(c, lm, program, plain[p_index])
        if not whole.actions[at]:
            continue  # the controller ended the copy before any tick
        resumed = run_program(c, lm, program, plain[p_index], snapshot_at=at)
        same = whole.comparable() == resumed.comparable()
        agree += same
        differ += not same
        for a, b in zip(whole.features, resumed.features, strict=False):
            max_diff = max(max_diff, float((a - b).abs().max()))
        example = example or resumed.snapshot
        if agree + differ == limit:
            break
    return {
        "arm": c.arm,
        "programs": agree + differ,
        "agree": agree,
        "differ": differ,
        "max_feature_difference": max_diff,
        "example": example,
    }


# ---------- run ----------


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["arm"], row["split"]), []).append(row)
    summary = []
    for (arm, split), group in sorted(groups.items()):
        entry: dict[str, Any] = {"arm": arm, "split": split, "seeds": len(group)}
        for key in ("task_accuracy", "action_accuracy", "copy_fidelity"):
            values = [r[key] for r in group if r[key] is not None]
            if values:
                entry[key] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
        summary.append(entry)
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="E03: token, latent, and head feedback versus a native loop.",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--long-test", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--latent-batch", type=int, default=32, help="programs per step for arm L"
    )
    parser.add_argument("--snapshot-programs", type=int, default=20)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="device for the frozen model")
    parser.add_argument("--out", default="runs")
    args = parser.parse_args(argv)
    if min(args.train, args.test, args.long_test, args.epochs) < 1:
        parser.error("sizes and epochs must be positive")
    if not set(args.arms.split(",")) <= set(ARMS):
        parser.error(f"arms must be among {','.join(ARMS)}")
    return args


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is not available")
    if args.threads:
        torch.set_num_threads(args.threads)
    arms = tuple(a for a in ARMS if a in args.arms.split(","))
    config = Config(
        model=args.model,
        revision=None if args.model == "tiny-random" else args.revision,
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        data_seed=args.data_seed,
        train=args.train,
        test=args.test,
        long_test=args.long_test,
        epochs=args.epochs,
        hidden=args.hidden,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        latent_batch=args.latent_batch,
        snapshot_programs=args.snapshot_programs,
    )
    run_dir, source, root = lockstep.start(args.out, "e03")
    data = {
        s.name: (s, list(task.programs(s, config.data_seed))) for s in splits(config)
    }
    golds = {name: [gold(p) for p in programs] for name, (_, programs) in data.items()}
    header = lockstep.run_header(
        experiment=EXPERIMENT,
        module=MODULE,
        argv=argv,
        run_dir=run_dir,
        source=source,
        root=root,
        config=asdict(config) | {"arms": list(arms)},
        device=args.device,
    )
    if args.device.startswith("cuda"):
        header["environment"] |= {
            "gpu": torch.cuda.get_device_name(args.device),
            "cuda": torch.version.cuda,
        }
    rows: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    trained: dict[str, Any] = {}
    error = None
    lm = None
    try:
        lm = FrozenLM.load(config.model, config.revision, args.device)
        train_programs, train_golds = data["train"][1], golds["train"]
        cached = {
            arm: cached_examples(arm, lm, train_programs, train_golds)
            for arm in arms
            if arm != "L"
        }
        latent = (
            [
                latent_program(lm, p, g)
                for p, g in zip(train_programs, train_golds, strict=True)
            ]
            if "L" in arms
            else []
        )
        plain = {
            name: [plain_features(lm, p) for p in programs]
            for name, (_, programs) in data.items()
            if name != "train"
        }
        for seed in config.seeds:
            for arm in arms:
                torch.manual_seed(seed)
                c = new_controller(arm, lm, config)
                if arm == "L":
                    train_latent(c, lm, latent, config)
                else:
                    train_cached(c, cached[arm], config)
                for m in c.modules():
                    m.eval()
                trained[arm] = {
                    "trainable_parameters": parameter_count(*c.modules()),
                    "adaptor_parameters": parameter_count(c.adaptor),
                    "training_steps": c.steps,
                    "decision_examples": (
                        sum(len(p.decision_targets) for p in latent)
                        if arm == "L"
                        else len(cached[arm].decision_targets)
                    ),
                    "answer_examples": (
                        sum(len(p.answers) for p in latent)
                        if arm == "L"
                        else len(cached[arm].answers)
                    ),
                }
                for name in plain:
                    result = evaluate(
                        c,
                        lm,
                        data[name][1],
                        golds[name],
                        plain[name],
                        trace if seed == config.seeds[0] else None,
                    )
                    rows.append(
                        {
                            "arm": arm,
                            "split": name,
                            "seed": seed,
                            "training_seconds": round(c.seconds, 1),
                            **result,
                        }
                    )
                    print(
                        f"seed {seed} arm {arm} {name:<22} "
                        f"task {result['task_accuracy']:.3f}  "
                        f"actions {result['action_accuracy']:.3f}  "
                        f"copy {result['copy_fidelity']}",
                        flush=True,
                    )
                if seed == config.seeds[0]:
                    name = "test_in_distribution"
                    snapshots.append(
                        snapshot_check(
                            c,
                            lm,
                            data[name][1],
                            plain[name],
                            config.snapshot_programs,
                        )
                    )
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)

    mismatches = sum(r["executor_mismatches"] for r in rows)
    differ = sum(s["differ"] for s in snapshots)
    if error:
        status = "error"
    elif mismatches or differ:
        status = "failed"
    else:
        status = "completed"
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
                "frozen": True,
                "features": "last-layer hidden state at the last position of a chunk "
                "or at a latent feedback vector",
            },
            "task": {
                "machine": "memory/v1, capacity 8, 4-bit words",
                "executor": "delta-rule tensor backend (E00) with a shadow oracle",
                "statements": "set (1 tick), copy (read then write, 2 ticks), "
                "question (1 tick); weights 0.4, 0.3, 0.3",
                "tick_budget": ticks.TICK_BUDGET,
                "splits": {
                    name: asdict(split) | {"sha256": digest(programs)}
                    for name, (split, programs) in data.items()
                },
            },
            "arms": {a: ARM_NAMES[a] for a in arms},
            "feedback_reentry": {
                "T": "request and reply text appended; next decision reads the "
                "last layer at the reply's last token",
                "L": "one adaptor vector per tick at the next input position; it "
                "passes through every layer and stays in the key-value cache",
                "H": "embedded code concatenated to the heads' input; the model "
                "and its cache never see it",
                "N": "no controller feedback within a statement; the loop passes "
                "the read value to the write",
            },
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
                "bounded stepping (at most "
                f"{ticks.TICK_BUDGET} machine actions per statement), deterministic "
                "events, zero executor mismatches, and snapshot/resume between the "
                "ticks of a copy identical to uninterrupted execution"
            ),
            "executor_mismatches": mismatches,
            "snapshots": snapshots,
            "peak_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "peak_gpu_mb": (
                round(torch.cuda.max_memory_allocated() / 2**20, 1)
                if args.device.startswith("cuda")
                else None
            ),
            "summary": summarize(rows),
            "runs": rows,
            "not_measured": [
                "unfrozen or fine-tuned models; only heads and the adaptor train",
                "latent feedback without the action fields (observation only)",
                "per-tick latency on matched hardware: T and L run the model on "
                "the GPU while H and N reuse features computed once",
                "faults from invalid actions: the heads can only emit valid "
                "addresses and values",
            ],
            "error": error,
        },
    )
    print(f"status: {status}\nrecord: {run_dir}")
    return {"completed": 0, "failed": 1}.get(status, 2)


if __name__ == "__main__":
    sys.exit(main())
