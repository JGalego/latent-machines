"""E09: state in the machine, not the context.

E02 found that every controller loses accuracy on long programs while the
executors stay exact. E09 asks whether that loss comes from reading a long
context. It reuses E02's task, frozen model, heads, training programs, and
free-running evaluation, and adds two arms that read one statement at a
time:

- B: text tool, full context (E02's arm B, rerun here).
- C: typed interface, full context (E02's arm C, rerun here).
- S: typed interface, one statement per forward pass. The window is the
  preamble plus that statement, so the machine is the only carrier of state.
- W: no machine, same windows; the answer head sees only the current
  question. A floor, not a competitor.

S runs again with the delta-rule executor (arm ``S+delta``) as a conformance
check. The pre-registered predictions are in EXPERIMENTS.md. This runner also
computes the compounding baseline defined there and evaluates the
predictions, so the report states each one as met or not met.

    uv run --locked python -m latent_machines.experiments.e09 --model tiny-random
    uv run --locked python -m latent_machines.experiments.e09 --device cuda

Exit status: 0 when the run completes with no executor mismatches, 1 if the
delta-rule executor disagrees with its shadow oracle, 2 if the run failed.
"""

import argparse
import json
import math
import random
import resource
import sys
import traceback
from dataclasses import asdict, dataclass
from typing import Any

import torch

from latent_machines.controllers.frozen_lm import FrozenLM
from latent_machines.controllers.heads import parameter_count
from latent_machines.experiments import e02, lockstep, records
from latent_machines.interfaces import memory as interface
from latent_machines.machines.memory import Read, Write
from latent_machines.tasks import records as task

EXPERIMENT = "E09"
MODULE = "latent_machines.experiments.e09"
ARMS = ("B", "C", "S", "S+delta", "W")
TEST_SPLITS = (
    "test_in_distribution",
    "test_length_32",
    "test_length_64",
    "test_length_256",
)
LENGTH_THRESHOLDS = (64, 256)  # splits where prediction 1 is evaluated
WITHIN = 0.05  # prediction 2 tolerance
FLOOR = 0.15  # prediction 3 ceiling for W
ACTION_DRIFT = 0.03  # implementation check on S's action accuracy across lengths
REPLAYS = 20  # corruption draws per program for the compounding baseline


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
    arms: tuple[str, ...] = ARMS


def splits(config: Config) -> list[task.Split]:
    held = task.HELD_OUT
    return [
        task.Split("train", config.train, 2, 8, held),
        task.Split("test_in_distribution", config.test, 2, 8, held),
        task.Split("test_length_32", config.long_test, 32, 32, held),
        task.Split("test_length_64", config.long_test, 64, 64, held),
        task.Split("test_length_256", config.long_test, 256, 256, held),
    ]


class WindowedLM:
    """Features of one statement read alone, behind ``FrozenLM``'s interface.

    A statement's window is the preamble followed by the statement. Its
    feature depends only on that text, so a statement gets the same feature
    wherever it sits in a program. ``chunk_features`` takes E02's chunk list
    (preamble, then statements) and returns the same shape of result, so
    E02's example builder and evaluator run unchanged.
    """

    def __init__(self, lm: FrozenLM) -> None:
        self.lm = lm
        self.hidden_size = lm.hidden_size
        self.cache: dict[str, torch.Tensor] = {}
        self.preamble = lm.chunk_features([task.PREAMBLE])[0]

    def tokens(self, chunks: list[str]) -> list[list[int]]:
        return self.lm.tokens(chunks)

    def feature(self, text: str) -> torch.Tensor:
        if text not in self.cache:
            self.cache[text] = self.lm.chunk_features([task.PREAMBLE, text])[1]
        return self.cache[text]

    def chunk_features(self, chunks: list[str]) -> torch.Tensor:
        return torch.stack([self.preamble, *(self.feature(c) for c in chunks[1:])])

    def window_positions(self, text: str) -> int:
        return sum(len(ids) for ids in self.lm.tokens([task.PREAMBLE, text]))


def context_positions(
    arm: str, lm: FrozenLM, windows: WindowedLM, programs: list[task.Program]
) -> dict[str, float]:
    """Positions the model attends over in one forward pass, per program.

    For B this uses the gold-action transcript, which free-running control
    matches whenever its actions are right.
    """
    if arm in ("S", "S+delta", "W"):
        counts = [
            max(windows.window_positions(s.text) for s in p.statements)
            for p in programs
        ]
        totals = [
            sum(windows.window_positions(s.text) for s in p.statements)
            for p in programs
        ]
    else:
        chunks = e02.transcript_chunks if arm == "B" else e02.plain_chunks
        counts = [sum(len(ids) for ids in lm.tokens(chunks(p))) for p in programs]
        totals = counts
    return {
        "max_positions": max(counts),
        "mean_positions_per_program": sum(totals) / len(totals),
    }


# ---------- compounding baseline ----------


def replay_accuracy(
    programs: list[task.Program], error_rate: float, seed: int, replays: int = REPLAYS
) -> float:
    """Fraction of questions answered correctly when each statement's action is
    replaced, with probability ``error_rate``, by a random one of its kind."""
    rng = random.Random(f"baseline:{seed}")
    oracle = task.machine()
    names = len(task.NAMES)
    questions = correct = 0
    for program in programs:
        gold = program.answers()
        for _ in range(replays):
            state = oracle.initial_state()
            for i, statement in enumerate(program.statements):
                action = statement.action
                if rng.random() < error_rate:
                    record = rng.randrange(names)
                    action = (
                        Write(record, rng.randrange(task.MAX_VALUE + 1))
                        if statement.kind == "set"
                        else Read(record)
                    )
                transition = oracle.step(state, action)
                state = transition.next_state
                if statement.kind == "get":
                    questions += 1
                    value = transition.event.value
                    correct += value == gold[i]
    return correct / questions


@torch.no_grad()
def answer_given_gold(
    controller: e02.Controller, windows: WindowedLM, programs: list[task.Program]
) -> float:
    """The answer head's accuracy when its reply is the gold value (``a``)."""
    oracle = task.machine()
    right = total = 0
    for program in programs:
        state, gold = oracle.initial_state(), program.answers()
        for i, statement in enumerate(program.statements):
            transition = oracle.step(state, statement.action)
            state = transition.next_state
            if statement.kind != "get":
                continue
            obs = torch.tensor(interface.observation(transition.event))
            logits = controller.answers(windows.feature(statement.text), obs)
            right += logits.argmax().item() == gold[i]
            total += 1
    return right / total


# ---------- evaluation of the pre-registered predictions ----------


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def bound(questions: int) -> float:
    """E02's conservative two-standard-error bound for a difference of two
    proportions over ``questions`` questions."""
    return 2 * math.sqrt(0.5 / questions)


def evaluate_predictions(
    rows: list[dict[str, Any]], baselines: list[dict[str, Any]]
) -> dict[str, Any]:
    def by(arm: str, split: str, key: str) -> dict[int, Any]:
        return {
            r["seed"]: r[key] for r in rows if r["arm"] == arm and r["split"] == split
        }

    seeds = sorted(by("S", "test_in_distribution", "task_accuracy"))
    one, two, three, checks = [], [], [], []
    for split in TEST_SPLITS:
        length = int(split.rsplit("_", 1)[1]) if split != TEST_SPLITS[0] else None
        s_acc, c_acc = by("S", split, "task_accuracy"), by("C", split, "task_accuracy")
        questions = by("S", split, "questions")[seeds[0]]
        base = [b["accuracy"][split] for b in baselines]
        diff = mean([s_acc[s] - c_acc[s] for s in seeds])
        gap = mean([s_acc[s] for s in seeds]) - mean(base)
        if length in LENGTH_THRESHOLDS:
            one.append(
                {
                    "split": split,
                    "mean_s_minus_c": diff,
                    "bound": bound(questions),
                    "met": diff > bound(questions),
                }
            )
        two.append(
            {
                "split": split,
                "s_accuracy": mean([s_acc[s] for s in seeds]),
                "baseline": mean(base),
                "difference": gap,
                "within": abs(gap) <= WITHIN,
                "below_by_more_than_tolerance": gap < -WITHIN,
            }
        )
        w_acc = mean(list(by("W", split, "task_accuracy").values()))
        three.append({"split": split, "w_accuracy": w_acc, "met": w_acc <= FLOOR})
        drift = abs(
            mean(list(by("S", split, "action_accuracy").values()))
            - mean(list(by("S", TEST_SPLITS[0], "action_accuracy").values()))
        )
        checks.append(
            {"split": split, "s_action_drift": drift, "ok": drift <= ACTION_DRIFT}
        )
    return {
        "1_s_beats_c_beyond_bound": {"results": one, "met": all(r["met"] for r in one)},
        "2_s_tracks_compounding_baseline": {
            "tolerance": WITHIN,
            "results": two,
            "met": all(r["within"] for r in two),
            "counts_against_thesis": any(
                r["below_by_more_than_tolerance"] for r in two
            ),
        },
        "3_w_stays_at_floor": {
            "ceiling": FLOOR,
            "results": three,
            "met": all(r["met"] for r in three),
        },
        "sanity_s_action_accuracy_constant": {
            "threshold": ACTION_DRIFT,
            "results": checks,
            "ok": all(r["ok"] for r in checks),
        },
    }


# ---------- run ----------


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="E09: bounded-context controller with the machine holding state.",
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
    run_dir, source, root = lockstep.start(args.out, "e09")
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
    baselines: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    trained: dict[str, Any] = {}
    positions: dict[str, Any] = {}
    predictions = None
    error = None
    lm = None
    try:
        lm = FrozenLM.load(config.model, config.revision, args.device)
        windows = WindowedLM(lm)
        train_programs = data["train"][1]
        full = {arm: e02.build_examples(lm, train_programs, arm) for arm in "BC"}
        windowed = e02.build_examples(windows, train_programs, "C")
        features = {
            "full": {
                name: [lm.chunk_features(e02.plain_chunks(p))[1:] for p in programs]
                for name, (_, programs) in data.items()
                if name in TEST_SPLITS
            },
            "window": {
                name: [
                    windows.chunk_features(e02.plain_chunks(p))[1:] for p in programs
                ]
                for name, (_, programs) in data.items()
                if name in TEST_SPLITS
            },
        }
        for arm in ARMS:
            positions[arm] = {
                name: context_positions(arm, lm, windows, data[name][1])
                for name in TEST_SPLITS
            }
        for seed in config.seeds:
            controllers = {
                "B": e02.train("B", full["B"], lm, config, seed),
                "C": e02.train("C", full["C"], lm, config, seed),
                "S": e02.train("C", windowed, windows, config, seed),
                "W": e02.train("A", windowed, windows, config, seed),
            }
            controllers["S+delta"] = controllers["S"]  # same heads, delta executor
            arm_setup = {  # arm -> (e02 arm name, features, executor note)
                "B": ("B", features["full"], lm),
                "C": ("C", features["full"], lm),
                "S": ("C", features["window"], windows),
                "S+delta": ("D", features["window"], windows),
                "W": ("A", features["window"], windows),
            }
            for arm in ARMS:
                c = controllers[arm]
                examples = {"B": full["B"], "C": full["C"]}.get(arm, windowed)
                trained[arm] = {
                    "trainable_parameters": parameter_count(c.actions, c.answers),
                    "training_steps": c.steps,
                    "action_examples": len(examples.operations) if c.actions else 0,
                    "answer_examples": len(examples.answers),
                }
                e02_arm, plain, model = arm_setup[arm]
                for name in TEST_SPLITS:
                    local: list[dict[str, Any]] = []
                    result = e02.evaluate(
                        e02_arm,
                        c,
                        model,
                        data[name][1],
                        plain[name],
                        local if seed == config.seeds[0] else None,
                    )
                    for entry in local:
                        trace.append(entry | {"arm": arm})
                    if arm in ("S", "S+delta", "W"):
                        # E02's token count assumes full-context features.
                        result["lm_tokens"] = int(
                            positions[arm][name]["mean_positions_per_program"]
                            * len(data[name][1])
                        )
                    rows.append({"arm": arm, "split": name, "seed": seed, **result})
                    print(
                        f"seed {seed} arm {arm:<7} {name:<20} "
                        f"task {result['task_accuracy']:.3f}"
                        + (
                            f"  actions {result['action_accuracy']:.3f}"
                            if result["action_accuracy"] is not None
                            else ""
                        ),
                        flush=True,
                    )
            s_rows = [
                r
                for r in rows
                if r["arm"] == "S"
                and r["seed"] == seed
                and r["split"] == "test_in_distribution"
            ]
            epsilon = 1 - s_rows[0]["action_accuracy"]
            a = answer_given_gold(
                controllers["S"], windows, data["test_in_distribution"][1]
            )
            accuracy = {
                name: a * replay_accuracy(data[name][1], epsilon, seed=seed * 1000 + i)
                for i, name in enumerate(TEST_SPLITS)
            }
            baselines.append(
                {"seed": seed, "epsilon": epsilon, "a": a, "accuracy": accuracy}
            )
            print(
                f"seed {seed} baseline epsilon {epsilon:.4f} a {a:.3f} "
                + " ".join(f"{k.split('_')[-1]}:{v:.3f}" for k, v in accuracy.items()),
                flush=True,
            )
        predictions = evaluate_predictions(rows, baselines)
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
                    | {
                        "held_out": sorted(split.held_out),
                        "sha256": e02.digest(programs),
                    }
                    for name, (split, programs) in data.items()
                },
            },
            "arms": {
                "B": "text tool, full context; oracle executor (E02 arm B)",
                "C": "typed interface, full context; oracle executor (E02 arm C)",
                "S": "typed interface; one statement per forward pass "
                "(preamble plus statement); oracle executor",
                "S+delta": "S's heads with the delta-rule executor and shadow oracle",
                "W": "no machine; one statement per forward pass",
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
                "zero executor mismatches for the delta-rule arm"
            ),
            "executor_mismatches": mismatches,
            "peak_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1
            ),
            "summary": e02.summarize(rows),
            "runs": rows,
            "compounding_baseline": baselines,
            "predictions": predictions,
            "context_positions": positions,
            "not_measured": [
                "tasks whose statements need earlier statements to be read",
                "other model sizes, fine-tuned or unfrozen baselines",
                "free-running context length for B (positions use the gold transcript)",
            ],
            "error": error,
        },
    )
    print(f"status: {status}\nrecord: {run_dir}")
    return {"completed": 0, "executor_mismatch": 1}.get(status, 2)


if __name__ == "__main__":
    sys.exit(main())
