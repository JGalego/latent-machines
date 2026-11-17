"""E02: learned interface comparison on record-update programs.

A frozen language model reads each program; small trained heads choose
memory actions and answer questions. Four arms share the task, the frozen
features, and the head architecture:

- A: no machine. The answer head reads the question's features.
- B: text tool. Each action and the machine's reply are appended to the
  model's context as text; the answer head reads the features after the
  reply. Executor: the ``memory/v1`` oracle.
- C: typed internal interface. Actions go to the machine as ``Action``
  objects and replies return as an embedded integer. Executor: the oracle,
  run natively.
- D: C's trained heads with the delta-rule tensor backend as executor, plus
  a shadow oracle that counts executor mismatches.

C versus B tests the interface; D versus C tests the implementation.
Evaluation is free-running: predicted actions drive the machine, so a wrong
write affects later answers. Only the heads are trained.

    uv run --locked python -m latent_machines.experiments.e02
    uv run --locked python -m latent_machines.experiments.e02 --model tiny-random

Exit status: 0 when the run completes with no executor mismatches, 1 if D
disagrees with its shadow oracle, 2 if the run itself failed.
"""

import argparse
import hashlib
import json
import resource
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from latent_machines.backends.delta import DeltaMemory
from latent_machines.controllers.frozen_lm import FrozenLM
from latent_machines.controllers.heads import ActionHead, AnswerHead, parameter_count
from latent_machines.experiments import lockstep, records
from latent_machines.interfaces import memory as interface
from latent_machines.tasks import records as task

EXPERIMENT = "E02"
MODULE = "latent_machines.experiments.e02"
MODEL = "HuggingFaceTB/SmolLM2-135M"
REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
ARMS = ("A", "B", "C", "D")
TRACE_PROGRAMS = 5  # programs per split traced for the first seed


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


def splits(config: Config) -> list[task.Split]:
    held = task.HELD_OUT
    return [
        task.Split("train", config.train, 2, 8, held),
        task.Split("test_in_distribution", config.test, 2, 8, held),
        task.Split("test_length_32", config.long_test, 32, 32, held),
        task.Split("test_length_64", config.long_test, 64, 64, held),
        task.Split(
            "test_held_out_pairs", config.test, 2, 8, held, exclude_held_out=False
        ),
    ]


def gold_trace(program: task.Program) -> list[Any]:
    """Events from running the program's gold actions on the oracle."""
    machine, state, events = task.machine(), task.machine().initial_state(), []
    for statement in program.statements:
        transition = machine.step(state, statement.action)
        state = transition.next_state
        events.append(transition.event)
    return events


def plain_chunks(program: task.Program) -> list[str]:
    return [task.PREAMBLE, *(s.text for s in program.statements)]


def transcript_chunks(program: task.Program) -> list[str]:
    """Arm B's context under gold actions: statement, request, reply."""
    chunks = [task.PREAMBLE]
    for statement, event in zip(program.statements, gold_trace(program), strict=True):
        chunks += [
            statement.text,
            interface.text_request(statement.action),
            interface.text_reply(event),
        ]
    return chunks


def digest(programs: list[task.Program]) -> str:
    text = "\n".join("".join(plain_chunks(p)) for p in programs)
    return hashlib.sha256(text.encode()).hexdigest()


# ---------- training ----------


@dataclass
class Examples:
    """Training tensors for one arm, built under gold actions."""

    action_features: torch.Tensor  # (statements, d)
    operations: torch.Tensor
    records_: torch.Tensor
    values: torch.Tensor
    is_write: torch.Tensor
    answer_features: torch.Tensor  # (questions, d)
    observations: torch.Tensor  # gold typed observations (C only)
    answers: torch.Tensor


def build_examples(lm: FrozenLM, programs: list[task.Program], arm: str) -> Examples:
    action_rows, answer_rows, targets, obs, answers = [], [], [], [], []
    for program in programs:
        events = gold_trace(program)
        if arm == "B":
            features = lm.chunk_features(transcript_chunks(program))
            statement_features = features[1::3]
            reply_features = features[3::3]
        else:
            statement_features = lm.chunk_features(plain_chunks(program))[1:]
            reply_features = statement_features
        for i, (statement, event) in enumerate(
            zip(program.statements, events, strict=True)
        ):
            action_rows.append(statement_features[i])
            targets.append(interface.from_action(statement.action))
            if statement.kind == "get":
                answer_rows.append(reply_features[i])
                obs.append(interface.observation(event))
                answers.append(event.value)
    target = torch.tensor(targets)
    return Examples(
        torch.stack(action_rows),
        target[:, 0],
        target[:, 1],
        target[:, 2],
        target[:, 0] == interface.OPERATIONS.index("write"),
        torch.stack(answer_rows),
        torch.tensor(obs),
        torch.tensor(answers),
    )


@dataclass
class Controller:
    actions: ActionHead | None
    answers: AnswerHead
    steps: int


def train(
    arm: str, examples: Examples, lm: FrozenLM, config: Config, seed: int
) -> Controller:
    torch.manual_seed(seed)
    values = task.MAX_VALUE + 1
    typed = arm in ("C", "D")
    actions = (
        None
        if arm == "A"
        else ActionHead(lm.hidden_size, config.hidden, 2, len(task.NAMES), values)
    )
    answers = AnswerHead(lm.hidden_size, config.hidden, values, values if typed else 0)
    modules = [m for m in (actions, answers) if m is not None]
    optimizer = torch.optim.Adam(
        [p for m in modules for p in m.parameters()], lr=config.learning_rate
    )
    steps = 0
    n_actions, n_answers = len(examples.operations), len(examples.answers)
    for _ in range(config.epochs):
        order_actions = torch.randperm(n_actions)
        order_answers = torch.randperm(n_answers)
        batches = max(1, -(-max(n_actions, n_answers) // config.batch_size))
        for b in range(batches):
            loss = torch.zeros(())
            if actions is not None:
                idx = order_actions[b::batches]
                op, rec, val = actions(examples.action_features[idx])
                loss = loss + F.cross_entropy(op, examples.operations[idx])
                loss = loss + F.cross_entropy(rec, examples.records_[idx])
                write = examples.is_write[idx]
                if write.any():
                    loss = loss + F.cross_entropy(
                        val[write], examples.values[idx][write]
                    )
            idx = order_answers[b::batches]
            if len(idx):
                observation = examples.observations[idx] if typed else None
                logits = answers(examples.answer_features[idx], observation)
                loss = loss + F.cross_entropy(logits, examples.answers[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
    for module in modules:
        module.eval()
    return Controller(actions, answers, steps)


# ---------- free-running evaluation ----------


def choose(controller: Controller, features: torch.Tensor) -> Any:
    assert controller.actions is not None
    op, rec, val = (logits.argmax().item() for logits in controller.actions(features))
    return interface.to_action(op, rec, val)


@torch.no_grad()
def evaluate(
    arm: str,
    controller: Controller,
    lm: FrozenLM,
    programs: list[task.Program],
    plain: list[torch.Tensor],
    trace: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Run every program free-running; return counts for the arm."""
    oracle = task.machine()
    delta = DeltaMemory(oracle) if arm == "D" else None
    questions = correct = statements = right_actions = 0
    work = executor_mismatches = transcript_tokens = lm_tokens = 0
    started = time.perf_counter()
    for p_index, program in enumerate(programs):
        gold = program.answers()
        state = oracle.initial_state()
        tensor_state = delta.initial_state() if delta else None
        session = lm.session() if arm == "B" else None
        if session is not None:
            session.feed(task.PREAMBLE)
        steps = []
        for i, statement in enumerate(program.statements):
            features = session.feed(statement.text) if session else plain[p_index][i]
            action = choose(controller, features) if arm != "A" else None
            event = None
            if action is not None:
                statements += 1
                right_actions += action == statement.action
                expected = oracle.step(state, action)
                state = expected.next_state
                event, work = expected.event, work + expected.work
                if delta is not None:
                    actual = delta.step(tensor_state, action)
                    tensor_state = actual.next_state
                    if (
                        actual.event != expected.event
                        or delta.decode(tensor_state) != state
                    ):
                        executor_mismatches += 1
                    event = actual.event
            answer = None
            if session is not None and action is not None and event is not None:
                session.feed(interface.text_request(action))
                reply = session.feed(interface.text_reply(event))
                if statement.kind == "get":
                    answer = controller.answers(reply).argmax().item()
            elif statement.kind == "get":
                obs = (
                    torch.tensor(interface.observation(event))
                    if arm in ("C", "D") and event is not None
                    else None
                )
                answer = controller.answers(features, obs).argmax().item()
            if statement.kind == "get":
                questions += 1
                correct += answer == gold[i]
            if trace is not None and p_index < TRACE_PROGRAMS:
                steps.append(
                    {
                        "statement": statement.text.strip(),
                        "action": lockstep.action_json(action) if action else None,
                        "event": lockstep.event_json(event) if event else None,
                        "answer": answer,
                        "gold": gold[i],
                    }
                )
        if session is not None:
            transcript_tokens += session.tokens - sum(
                len(ids) for ids in lm.tokens(plain_chunks(program))
            )
            lm_tokens += session.tokens
        else:
            lm_tokens += sum(len(ids) for ids in lm.tokens(plain_chunks(program)))
        if trace is not None and p_index < TRACE_PROGRAMS:
            trace.append({"arm": arm, "program": p_index, "steps": steps})
    return {
        "questions": questions,
        "task_accuracy": correct / questions if questions else None,
        "actions": statements,
        "action_accuracy": right_actions / statements if statements else None,
        "executor_mismatches": executor_mismatches if arm == "D" else None,
        "machine_work": work,
        "lm_tokens": lm_tokens,
        "transcript_tokens": transcript_tokens,
        "eval_seconds": round(time.perf_counter() - started, 3),
    }


# ---------- run ----------


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean, min, and max over seeds for each arm and split."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["arm"], row["split"]), []).append(row)
    summary = []
    for (arm, split), group in sorted(groups.items()):
        entry: dict[str, Any] = {"arm": arm, "split": split, "seeds": len(group)}
        for key in ("task_accuracy", "action_accuracy"):
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
        description="E02: no machine vs text tool vs typed internal interface.",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=2000, help="training programs")
    parser.add_argument(
        "--test", type=int, default=300, help="programs per short split"
    )
    parser.add_argument(
        "--long-test", type=int, default=100, help="programs per long split"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--threads", type=int, default=0, help="torch threads (0: default)"
    )
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
    )
    run_dir, source, root = lockstep.start(args.out, "e02")
    data = {
        s.name: (s, list(task.programs(s, config.data_seed))) for s in splits(config)
    }
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
    error = None
    lm = None
    try:
        lm = FrozenLM.load(config.model, config.revision, args.device)
        train_programs = data["train"][1]
        examples = {
            arm: build_examples(lm, train_programs, arm) for arm in ("A", "B", "C")
        }
        plain = {
            name: [lm.chunk_features(plain_chunks(p))[1:] for p in programs]
            for name, (_, programs) in data.items()
            if name != "train"
        }
        for seed in config.seeds:
            controllers = {
                arm: train(arm, examples[arm], lm, config, seed) for arm in "ABC"
            }
            controllers["D"] = controllers["C"]  # same heads, different executor
            for arm in ARMS:
                c = controllers[arm]
                trained[arm] = {
                    "trainable_parameters": parameter_count(c.actions, c.answers),
                    "training_steps": c.steps,
                    "action_examples": len(
                        examples["C" if arm == "D" else arm].operations
                    )
                    if arm != "A"
                    else 0,
                    "answer_examples": len(
                        examples["C" if arm == "D" else arm].answers
                    ),
                    "heads": "same trained heads as C" if arm == "D" else "trained",
                }
                for name in plain:
                    programs = data[name][1]
                    result = evaluate(
                        arm,
                        c,
                        lm,
                        programs,
                        plain[name],
                        trace if seed == config.seeds[0] else None,
                    )
                    rows.append({"arm": arm, "split": name, "seed": seed, **result})
                    print(
                        f"seed {seed} arm {arm} {name:<22} "
                        f"task {result['task_accuracy']:.3f}"
                        + (
                            f"  actions {result['action_accuracy']:.3f}"
                            if result["action_accuracy"] is not None
                            else ""
                        ),
                        flush=True,
                    )
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)

    mismatches = sum(r["executor_mismatches"] or 0 for r in rows)
    status = (
        "error" if error else ("completed" if mismatches == 0 else "executor_mismatch")
    )
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
                "features": "last-layer hidden state at each chunk's last token",
            },
            "task": {
                "machine": "memory/v1, capacity 8, 4-bit words",
                "records": task.NAMES,
                "held_out_pairs": sorted(task.HELD_OUT),
                "splits": {
                    name: asdict(split)
                    | {"held_out": sorted(split.held_out), "sha256": digest(programs)}
                    for name, (split, programs) in data.items()
                },
            },
            "arms": {
                "A": "no machine; answer head on question features",
                "B": "text tool; request and reply in the context; oracle executor",
                "C": "typed actions and embedded observations; native oracle executor",
                "D": "C's heads; delta-rule tensor executor with shadow oracle",
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
                "executor equivalence reported apart from task and action accuracy; "
                "comparisons need several seeds on paired test programs"
            ),
            "executor_mismatches": mismatches,
            "peak_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "summary": summarize(rows),
            "runs": rows,
            "not_measured": [
                "learned feedback into the language model: typed observations reach "
                "only the answer head (latent feedback is E03)",
                "validity-constraint ablation and capacity changes",
                "fine-tuned or unfrozen baselines",
            ],
            "error": error,
        },
    )
    print(f"status: {status}\nrecord: {run_dir}")
    return {"completed": 0, "executor_mismatch": 1}.get(status, 2)


if __name__ == "__main__":
    sys.exit(main())
