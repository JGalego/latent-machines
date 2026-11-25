"""E11: learning to use the machine from outcomes.

E02 and E03 train heads on gold actions. E11 asks whether heads can learn to
write and read when the only signal is whether the final answers are right.
It reuses E02's task, heads, and free-running evaluation, and E09's windows,
so the machine is the only carrier of state. Four arms differ in the training
signal:

- I: imitation of gold actions (E09's S, retrained here).
- O: outcome only, through a differentiable copy of the machine: Gumbel-softmax
  straight-through one-hots for the operation, record, and value drive an
  8x16 one-hot memory, and the loss is the answer cross-entropy at questions.
- R: outcome only, REINFORCE over the integer machine used as an external tool.
- W: no machine; the answer head sees only the question window (E09's W).

O and R never see gold actions or gold values read. After training, every arm
runs free-running on the delta-rule backend with a shadow oracle. The
pre-registered predictions are in EXPERIMENTS.md and are evaluated here.

    uv run --locked python -m latent_machines.experiments.e11 --model tiny-random
    uv run --locked python -m latent_machines.experiments.e11 --threads 4

Exit status: 0 when the run completes with no executor mismatches, 1 if the
delta-rule executor disagrees with its shadow oracle, 2 if the run failed.
"""

import argparse
import json
import resource
import sys
import traceback
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from latent_machines.controllers.frozen_lm import FrozenLM
from latent_machines.controllers.heads import ActionHead, AnswerHead, parameter_count
from latent_machines.experiments import e02, lockstep, records
from latent_machines.experiments.e09 import WindowedLM, bound, mean
from latent_machines.interfaces import memory as interface
from latent_machines.tasks import records as task

EXPERIMENT = "E11"
MODULE = "latent_machines.experiments.e11"
ARMS = ("I", "O", "R", "W")
TEST_SPLITS = (
    "test_in_distribution",
    "test_length_32",
    "test_length_64",
    "test_held_out_pairs",
)
RECORDS = len(task.NAMES)
VALUES = task.MAX_VALUE + 1
WRITE = interface.OPERATIONS.index("write")
READ = interface.OPERATIONS.index("read")
TEMPERATURE = 1.0
DISCOVERY = 0.90  # prediction 1 and the steps-to-threshold measurement
FAILURE = 0.50  # prediction 1 below this counts against the thesis
OPERATION_MIN = 0.95  # prediction 2
SEEN_MIN = 0.99  # sanity: I at 2-8 statements
FLOOR = 0.55  # sanity: W at 2-8 statements
PROGRESS_PROGRAMS = 200  # training programs scored after every epoch


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
    batch_size: int  # statements per batch for I and W, as in E02
    program_batch: int  # programs per batch for O and R
    arms: tuple[str, ...] = ARMS


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


# ---------- programs as padded tensors ----------


@dataclass
class Batch:
    """What outcome-only training may see: features, which statements are
    questions, and the answers to them. There are no gold actions."""

    features: torch.Tensor  # (programs, steps, d), window features
    valid: torch.Tensor  # (programs, steps) bool, False on padding
    is_question: torch.Tensor  # (programs, steps) bool
    answers: torch.Tensor  # (programs, steps) long, 0 where not a question

    def select(self, index: torch.Tensor) -> "Batch":
        return Batch(
            self.features[index],
            self.valid[index],
            self.is_question[index],
            self.answers[index],
        )


def make_batch(windows: WindowedLM, programs: list[task.Program]) -> Batch:
    steps = max(len(p.statements) for p in programs)
    features = torch.zeros(len(programs), steps, windows.hidden_size)
    valid = torch.zeros(len(programs), steps, dtype=torch.bool)
    is_question = torch.zeros(len(programs), steps, dtype=torch.bool)
    answers = torch.zeros(len(programs), steps, dtype=torch.long)
    for i, program in enumerate(programs):
        gold = program.answers()
        for t, statement in enumerate(program.statements):
            features[i, t] = windows.feature(statement.text)
            valid[i, t] = True
            if statement.kind == "get":
                is_question[i, t] = True
                answers[i, t] = gold[t]
    return Batch(features, valid, is_question, answers)


# ---------- the differentiable memory ----------


def initial_memory(programs: int) -> torch.Tensor:
    """``(programs, records, values)``: every record holds the one-hot of 0."""
    memory = torch.zeros(programs, RECORDS, VALUES)
    memory[:, :, 0] = 1.0
    return memory


def memory_step(
    memory: torch.Tensor,
    operation: torch.Tensor,
    record: torch.Tensor,
    value: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One action on the differentiable memory; returns the memory and the
    observation. All three action inputs are (soft or hard) one-hots.

    A write sets ``M <- M + w * y_record (x) (y_value - M)``, which with hard
    one-hots replaces the addressed row and leaves the others. A read outputs
    ``y_record . M``; a write outputs the one-hot of 0, as the typed
    observation of the integer machine does.
    """
    w = operation[:, WRITE] * valid
    r = operation[:, READ] * valid
    memory = memory + w[:, None, None] * record[:, :, None] * (
        value[:, None, :] - memory
    )
    read = (record[:, :, None] * memory).sum(1)
    zero = torch.zeros(VALUES)
    zero[0] = 1.0
    return memory, r[:, None] * read + w[:, None] * zero


def gumbel_straight_through(
    logits: torch.Tensor, temperature: float = TEMPERATURE
) -> torch.Tensor:
    """A hard one-hot sample in the forward pass and the gradient of a
    Gumbel-softmax in the backward pass."""
    uniform = torch.rand_like(logits).clamp_min(1e-10)
    noisy = logits - torch.log(-torch.log(uniform))
    soft = F.softmax(noisy / temperature, dim=-1)
    hard = F.one_hot(soft.argmax(-1), logits.shape[-1]).to(soft.dtype)
    return hard + soft - soft.detach()


# ---------- training ----------


def new_heads(lm: Any, config: Config) -> e02.Controller:
    return e02.Controller(
        ActionHead(lm.hidden_size, config.hidden, 2, RECORDS, VALUES),
        AnswerHead(lm.hidden_size, config.hidden, VALUES, VALUES),
        0,
    )


def outcome_loss(controller: e02.Controller, batch: Batch) -> torch.Tensor:
    """Answer cross-entropy at questions, through the differentiable memory."""
    assert controller.actions is not None
    programs, steps = batch.valid.shape
    memory = initial_memory(programs)
    total, count = torch.zeros(()), 0
    for t in range(steps):
        features = batch.features[:, t]
        op, rec, val = controller.actions(features)
        memory, obs = memory_step(
            memory,
            gumbel_straight_through(op),
            gumbel_straight_through(rec),
            gumbel_straight_through(val),
            batch.valid[:, t].float(),
        )
        asked = batch.is_question[:, t]
        if asked.any():
            logits = controller.answers(features[asked], observation_probs=obs[asked])
            total = total + F.cross_entropy(
                logits, batch.answers[asked, t], reduction="sum"
            )
            count += int(asked.sum())
    return total / max(count, 1)


def rollout(
    controller: e02.Controller, batch: Batch, sample: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run programs on the integer machine, one action per statement.

    Returns the log-probability of each chosen action ``(programs, steps)``,
    the answer cross-entropy summed over questions, and ``correct``
    ``(programs, steps)``: 1.0 where a question's argmax answer is right.
    """
    assert controller.actions is not None
    programs, steps = batch.valid.shape
    memory = torch.zeros(programs, RECORDS, dtype=torch.long)
    rows = torch.arange(programs)
    logp = torch.zeros(programs, steps)
    correct = torch.zeros(programs, steps)
    ce = torch.zeros(())
    for t in range(steps):
        features = batch.features[:, t]
        heads = controller.actions(features)
        if sample:
            dists = [torch.distributions.Categorical(logits=h) for h in heads]
            op, rec, val = (d.sample() for d in dists)
            writes = (op == WRITE).float()
            logp[:, t] = (
                dists[0].log_prob(op)
                + dists[1].log_prob(rec)
                + writes * dists[2].log_prob(val)
            ) * batch.valid[:, t]
        else:
            op, rec, val = (h.argmax(-1) for h in heads)
        write = (op == WRITE) & batch.valid[:, t]
        read = (op == READ) & batch.valid[:, t]
        observation = torch.where(read, memory[rows, rec], torch.zeros_like(rec))
        memory[rows[write], rec[write]] = val[write]
        asked = batch.is_question[:, t]
        if asked.any():
            logits = controller.answers(features[asked], observation[asked])
            ce = ce + F.cross_entropy(logits, batch.answers[asked, t], reduction="sum")
            correct[asked, t] = (logits.argmax(-1) == batch.answers[asked, t]).float()
    return logp, ce, correct


def progress(controller: e02.Controller, batch: Batch) -> float:
    """Free-running task accuracy of greedy actions on the integer machine."""
    with torch.no_grad():
        _, _, correct = rollout(controller, batch, sample=False)
    return float(correct.sum() / batch.is_question.sum())


def train_outcome(
    arm: str,
    batch: Batch,
    lm: Any,
    config: Config,
    seed: int,
) -> tuple[e02.Controller, list[dict[str, float]]]:
    """Train O (``arm == "O"``) or R from answers alone. Returns the heads and
    the per-epoch curve measured on the first training programs."""
    torch.manual_seed(seed)
    controller = new_heads(lm, config)
    modules = [controller.actions, controller.answers]
    optimizer = torch.optim.Adam(
        [p for m in modules for p in m.parameters()], lr=config.learning_rate
    )
    n = batch.valid.shape[0]
    watch = batch.select(torch.arange(min(PROGRESS_PROGRAMS, n)))
    curve: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        order = torch.randperm(n)
        for start in range(0, n, config.program_batch):
            part = batch.select(order[start : start + config.program_batch])
            if arm == "O":
                loss = outcome_loss(controller, part)
            else:
                loss = reinforce_loss(controller, part)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            controller.steps += 1
        curve.append(
            {
                "epoch": epoch + 1,
                "steps": controller.steps,
                "task_accuracy": progress(controller, watch),
            }
        )
    for module in modules:
        module.eval()
    return controller, curve


def reinforce_loss(controller: e02.Controller, batch: Batch) -> torch.Tensor:
    """REINFORCE with reward-to-go: a correct answer is reward 1 at its
    statement, each action is credited with the answers at that statement and
    after it, and the baseline is the batch mean of that return at each step.
    The answer head trains with cross-entropy on its answers."""
    logp, ce, correct = rollout(controller, batch, sample=True)
    returns = correct.flip(1).cumsum(1).flip(1)
    valid = batch.valid.float()
    baseline = (returns * valid).sum(0) / valid.sum(0).clamp_min(1)
    advantage = ((returns - baseline) * valid).detach()
    policy = -(advantage * logp).sum() / valid.sum()
    return policy + ce / batch.is_question.sum().clamp_min(1)


# ---------- measurements ----------


@torch.no_grad()
def agreement(
    controller: e02.Controller, windows: WindowedLM, programs: list[task.Program]
) -> dict[str, float]:
    """How often the heads' choice matches the gold action, from window
    features alone: the operation, the record, and the value of writes."""
    assert controller.actions is not None
    statements = [s for p in programs for s in p.statements]
    features = torch.stack([windows.feature(s.text) for s in statements])
    op, rec, val = (h.argmax(-1) for h in controller.actions(features))
    gold = torch.tensor([interface.from_action(s.action) for s in statements])
    writes = gold[:, 0] == WRITE
    return {
        "operation_accuracy": (op == gold[:, 0]).float().mean().item(),
        "record_agreement": (rec == gold[:, 1]).float().mean().item(),
        "value_agreement": (val[writes] == gold[writes, 2]).float().mean().item(),
    }


def steps_to_threshold(curve: list[dict[str, float]]) -> int | None:
    for point in curve:
        if point["task_accuracy"] >= DISCOVERY:
            return int(point["steps"])
    return None


def evaluate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split = TEST_SPLITS[0]

    def by(arm: str, key: str, name: str = split) -> dict[int, Any]:
        return {
            r["seed"]: r[key] for r in rows if r["arm"] == arm and r["split"] == name
        }

    seeds = sorted(by("O", "task_accuracy"))
    o_acc = mean([by("O", "task_accuracy")[s] for s in seeds])
    o_op = mean([by("O", "operation_accuracy")[s] for s in seeds])
    questions = by("O", "questions")[seeds[0]]
    diff = mean(
        [by("O", "task_accuracy")[s] - by("R", "task_accuracy")[s] for s in seeds]
    )
    i_acc = mean([by("I", "task_accuracy")[s] for s in seeds])
    w_acc = mean([by("W", "task_accuracy")[s] for s in seeds])
    return {
        "1_o_discovers_the_protocol": {
            "mean_o_task_accuracy": o_acc,
            "minimum": DISCOVERY,
            "met": o_acc >= DISCOVERY,
            "counts_against_thesis": o_acc < FAILURE,
        },
        "2_o_writes_and_reads": {
            "mean_o_operation_accuracy": o_op,
            "minimum": OPERATION_MIN,
            "met": o_op >= OPERATION_MIN,
        },
        "3_o_beats_reinforce": {
            "mean_o_minus_r": diff,
            "bound": bound(questions),
            "met": diff > bound(questions),
        },
        "sanity_i_reaches_imitation_ceiling": {
            "mean_i_task_accuracy": i_acc,
            "minimum": SEEN_MIN,
            "ok": i_acc >= SEEN_MIN,
        },
        "sanity_w_stays_at_floor": {
            "mean_w_task_accuracy": w_acc,
            "ceiling": FLOOR,
            "ok": w_acc <= FLOOR,
        },
    }


# ---------- run ----------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="E11: learning to use the machine from outcomes.",
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
        "--program-batch", type=int, default=32, help="programs per O and R batch"
    )
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="device for the frozen model")
    parser.add_argument("--out", default="runs")
    args = parser.parse_args(argv)
    if min(args.train, args.test, args.long_test, args.epochs, args.program_batch) < 1:
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
        program_batch=args.program_batch,
    )
    run_dir, source, root = lockstep.start(args.out, "e11")
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
    rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    trained: dict[str, Any] = {}
    curves: dict[str, Any] = {}
    predictions = None
    error = None
    lm = None
    try:
        lm = FrozenLM.load(config.model, config.revision, args.device)
        windows = WindowedLM(lm)
        train_programs = data["train"][1]
        imitation = e02.build_examples(windows, train_programs, "C")
        answers_only = make_batch(windows, train_programs)
        plain = {
            name: [windows.chunk_features(e02.plain_chunks(p))[1:] for p in programs]
            for name, (_, programs) in data.items()
            if name != "train"
        }
        for seed in config.seeds:
            controllers: dict[str, e02.Controller] = {
                "I": e02.train("C", imitation, windows, config, seed),  # type: ignore[arg-type]
                "W": e02.train("A", imitation, windows, config, seed),  # type: ignore[arg-type]
            }
            for arm in ("O", "R"):
                controllers[arm], curve = train_outcome(
                    arm, answers_only, windows, config, seed
                )
                curves.setdefault(arm, {})[str(seed)] = curve
                print(
                    f"seed {seed} arm {arm} trained: final progress "
                    f"{curve[-1]['task_accuracy']:.3f}, "
                    f"steps to {DISCOVERY}: {steps_to_threshold(curve)}",
                    flush=True,
                )
            for arm in ARMS:
                c = controllers[arm]
                trained[arm] = {
                    "trainable_parameters": parameter_count(c.actions, c.answers),
                    "training_steps": c.steps,
                    "steps_to_discovery": steps_to_threshold(
                        curves[arm][str(seed)] if arm in curves else []
                    ),
                }
                for name in plain:
                    programs = data[name][1]
                    local: list[dict[str, Any]] = []
                    result = e02.evaluate(
                        "A" if arm == "W" else "D",
                        c,
                        windows,
                        programs,
                        plain[name],
                        local if seed == config.seeds[0] else None,
                    )
                    for entry in local:
                        trace.append(entry | {"arm": arm})
                    result["lm_tokens"] = sum(
                        windows.window_positions(s.text)
                        for p in programs
                        for s in p.statements
                    )
                    if arm != "W":
                        result |= agreement(c, windows, programs)
                    rows.append({"arm": arm, "split": name, "seed": seed, **result})
                    print(
                        f"seed {seed} arm {arm} {name:<20} "
                        f"task {result['task_accuracy']:.3f}"
                        + (
                            f"  actions {result['action_accuracy']:.3f}"
                            f"  op {result['operation_accuracy']:.3f}"
                            f"  record {result['record_agreement']:.3f}"
                            f"  value {result['value_agreement']:.3f}"
                            if result["action_accuracy"] is not None
                            else ""
                        ),
                        flush=True,
                    )
        predictions = evaluate_predictions(rows)
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
                "features": "last-layer hidden state at the last token of the "
                "preamble plus one statement",
            },
            "task": {
                "machine": "memory/v1, capacity 8, 4-bit words",
                "records": task.NAMES,
                "held_out_pairs": sorted(task.HELD_OUT),
                "splits": {
                    name: asdict(split)
                    | {
                        "held_out": sorted(split.held_out),
                        "sha256": e02.digest(programs),
                    }
                    for name, (split, programs) in data.items()
                },
            },
            "arms": {
                "I": "imitation of gold actions; the answer head trains on gold "
                "typed observations (E09's S)",
                "O": "answer loss only, through a differentiable 8x16 memory with "
                f"Gumbel-softmax straight-through one-hots, temperature {TEMPERATURE}",
                "R": "REINFORCE over the integer machine, reward-to-go on "
                "correct answers, batch-mean baseline per step; the answer head "
                "trains with cross-entropy on its answers",
                "W": "no machine; the answer head sees only the question window",
            },
            "evaluation": "free-running on the delta-rule backend with a shadow "
            "oracle, greedy actions; W has no machine",
            "trained": trained,
            "curves": curves,
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
                "zero executor mismatches for the delta-rule backend"
            ),
            "executor_mismatches": mismatches,
            "peak_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "summary": e02.summarize(rows),
            "runs": rows,
            "predictions": predictions,
            "not_measured": [
                "other temperatures, learning rates, or batch sizes for O and R",
                "other models, capacities, or tasks",
                "statements that depend on earlier statements",
            ],
            "error": error,
        },
    )
    print(f"status: {status}\nrecord: {run_dir}")
    return {"completed": 0, "executor_mismatch": 1}.get(status, 2)


if __name__ == "__main__":
    sys.exit(main())
