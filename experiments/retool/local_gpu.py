"""12 ReTool: code-interlaced rollout on one local GPU.

The original ReTool chapter uses PyTRIO for sampling and training.  This file
keeps the original rollout protocol, Python sandbox, and reward logic, and
replaces only the remote sampler/trainer boundary with:

* Qwen3-4B-Instruct-2507 + 4-bit QLoRA on the local ModelScope GPU;
* a small PyTRIO-compatible sampling client in ``scripts/local_gpu_trainer.py``;
* the original multi-turn ``rollout_batch`` state machine;
* one local PPO-style ``backward()`` + optimizer step per rollout batch.

The default run is deliberately small enough to inspect in one sitting.  It is
still a real model rollout, real code execution, and a real policy update; it is
not a paper-scale ReTool reproduction.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import local_gpu_trainer as trainer  # noqa: E402
import local_rollout_trainer as rollout_trainer  # noqa: E402
import modelscope_lib as lib  # noqa: E402

# The original modules import ``pytrio`` at module import time.  Inject the
# minimal sampling API before importing the rollout state machine.
sys.modules["pytrio"] = trainer.local_trio_compat()
sys.path.insert(0, str(EXPERIMENT_DIR))

import data as retool_data  # noqa: E402
from rollout import RolloutConfig, rollout_batch  # noqa: E402
from sandbox import LocalPythonSandbox  # noqa: E402


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_ID = "BytedTsinghua-SIA/DAPO-Math-17k"
DATASET_PATTERN = "data/dapo-math-17k.parquet"

# DAPO-Math wraps every question in this template.  ReTool uses its own
# \boxed{} protocol, so the wrapper must not be passed to the model.
PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the "
    "answer to the problem.\n\n"
)
PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 GPU 版 ReTool / DAPO-Math-17k")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--profile",
        default=os.environ.get("RUN_PROFILE", "smoke"),
        choices=("smoke", "full"),
    )
    parser.add_argument("--steps", type=int, default=0, help="0 表示使用 profile 默认值")
    parser.add_argument("--questions-per-batch", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=0, help="0 表示使用 profile 默认值")
    parser.add_argument("--max-code-calls", type=int, default=2)
    parser.add_argument("--max-assistant-turns", type=int, default=3)
    parser.add_argument("--max-trajectory-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-assistant-tokens",
        type=int,
        default=0,
        help="单轮 assistant 最大 token；0 表示使用 profile 默认值",
    )
    parser.add_argument("--max-tool-response-tokens", type=int, default=512)
    parser.add_argument("--sandbox-timeout", type=float, default=15.0)
    parser.add_argument("--sandbox-workers", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=512,
        help="先载入多少条 DAPO 样本；0 表示使用全部（不建议）",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force-first-tool-call",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="教学模式：首轮预填工具调用骨架，只让模型生成 Python 代码",
    )
    parser.add_argument(
        "--dynamic-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="整组 reward 相同则换下一题，直到出现可训练组",
    )
    parser.add_argument("--max-question-attempts", type=int, default=3)
    parser.add_argument(
        "--prefer-short-questions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="教学运行优先短题，增加组内出现一正一负的概率",
    )
    return parser.parse_args()


def strip_dapo_template(question: str) -> str:
    if question.startswith(PROMPT_PREFIX):
        question = question[len(PROMPT_PREFIX) :]
    if question.endswith(PROMPT_SUFFIX):
        question = question[: -len(PROMPT_SUFFIX)]
    return question.strip()


def _find_cached_parquet(root: Path) -> Path | None:
    cache = root / "cache" / "modelscope" / "datasets"
    if not cache.exists():
        return None
    matches = sorted(cache.rglob("dapo-math-17k.parquet"))
    return matches[0] if matches else None


def _dataset_parquet(root: Path) -> Path:
    cached = _find_cached_parquet(root)
    if cached is not None:
        return cached
    dataset_dir = trainer.download_modelscope_dataset(
        DATASET_ID, root, DATASET_PATTERN
    )
    candidates = sorted(Path(dataset_dir).rglob("*.parquet"))
    if not candidates:
        raise trainer.DataUnavailable(f"No DAPO-Math parquet under {dataset_dir}")
    return candidates[0]


def load_examples(
    root: Path,
    seed: int,
    max_samples: int,
    prefer_short: bool = False,
) -> list[retool_data.MathExample]:
    rows = trainer.load_parquet_rows(
        _dataset_parquet(root),
        limit=max_samples or 512,
        seed=seed,
    )
    examples: list[retool_data.MathExample] = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            continue
        first = prompt[0] if isinstance(prompt[0], dict) else {}
        question = strip_dapo_template(str(first.get("content") or ""))
        reward_model = row.get("reward_model") or {}
        ground_truth = (
            reward_model.get("ground_truth")
            if isinstance(reward_model, dict)
            else None
        )
        if isinstance(ground_truth, list):
            ground_truth = ground_truth[0] if ground_truth else None
        answer = str(ground_truth or "").strip()
        if not question or not answer:
            continue
        examples.append(
            retool_data.MathExample(
                id=str((row.get("extra_info") or {}).get("index") or index),
                question=question,
                answer=answer,
                data_source=str(row.get("data_source") or "dapo_math"),
            )
        )
    if not examples:
        raise trainer.DataUnavailable("DAPO-Math rows had no usable question/answer")
    random.Random(seed).shuffle(examples)
    if prefer_short:
        examples.sort(key=lambda item: len(item.question))
    if max_samples > 0:
        examples = examples[:max_samples]
    return examples


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def degenerate_group_rate(trajectories: list[Any], question_count: int) -> float:
    groups: dict[int, list[float]] = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.question_index, []).append(float(trajectory.advantage))
    degenerate = sum(all(value == 0.0 for value in values) for values in groups.values())
    return degenerate / max(question_count, 1)


def rollout_metrics(
    trajectories: list[Any],
    losses: list[float],
    updated_trajectories: int,
    question_count: int,
    sandbox: LocalPythonSandbox,
    *,
    groups_sampled: int,
    groups_kept: int,
) -> dict[str, float]:
    tool_attempts = sum(
        "<tool_call>" in turn.text
        for trajectory in trajectories
        for turn in trajectory.turns
    )
    valid_tool_calls = sum(trajectory.code_calls for trajectory in trajectories)
    trajectory_lengths = [
        len(trajectory.turns[-1].prompt_tokens)
        + len(trajectory.turns[-1].completion_tokens)
        for trajectory in trajectories
        if trajectory.turns
    ]
    metrics = {
        "reward/mean": mean([float(item.reward) for item in trajectories]),
        "reward/correct": mean([float(item.correct) for item in trajectories]),
        "reward/format": mean([float(item.valid_format) for item in trajectories]),
        "rollout/turns": mean([float(len(item.turns)) for item in trajectories]),
        "rollout/code_calls": mean([float(item.code_calls) for item in trajectories]),
        "rollout/trajectory_tokens": mean([float(value) for value in trajectory_lengths]),
        "rollout/valid_tool_call_rate": valid_tool_calls / max(tool_attempts, 1),
        "rollout/degenerate_group_rate": degenerate_group_rate(
            trajectories, question_count
        ),
        "train/trajectories": float(len(trajectories)),
        "train/updated_trajectories": float(updated_trajectories),
        "train/loss_mean": mean(losses),
        "dynamic_sampling/groups_sampled": float(groups_sampled),
        "dynamic_sampling/groups_kept": float(groups_kept),
        "dynamic_sampling/groups_dropped": float(groups_sampled - groups_kept),
        "dynamic_sampling/drop_rate": (
            (groups_sampled - groups_kept) / groups_sampled
            if groups_sampled
            else 0.0
        ),
    }
    metrics.update(sandbox.stats.metrics())
    return metrics


def trajectory_record(
    step: int,
    trajectory: Any,
    *,
    attempt: int = 1,
    kept: bool = True,
) -> dict[str, Any]:
    return {
        "step": step,
        "attempt": attempt,
        "kept": kept,
        "question_index": trajectory.question_index,
        "group_index": trajectory.group_index,
        "id": trajectory.example.id,
        "question": trajectory.example.question,
        "reference": trajectory.example.answer,
        "reward": trajectory.reward,
        "advantage": trajectory.advantage,
        "correct": trajectory.correct,
        "valid_format": trajectory.valid_format,
        "code_calls": trajectory.code_calls,
        "final_text": trajectory.final_text,
        "turns": [
            {
                "text": turn.text,
                "prompt_tokens": len(turn.prompt_tokens),
                "completion_tokens": len(turn.completion_tokens),
            }
            for turn in trajectory.turns
        ],
    }


def main() -> int:
    args = parse_args()
    root, run_dir, run_id, config = trainer.prepare_run(12, args)
    steps = args.steps or (4 if args.profile == "smoke" else config["steps"])
    group_size = args.group_size or (4 if args.profile == "smoke" else config["group_size"])
    max_assistant_tokens = args.max_assistant_tokens or (
        (384 if args.force_first_tool_call else 768)
        if args.profile == "smoke"
        else max(config["max_new_tokens"], 768)
    )
    if (
        steps < 1
        or args.questions_per_batch < 1
        or group_size < 2
        or args.max_code_calls < 1
        or args.max_assistant_turns < 1
        or args.max_trajectory_tokens < 256
        or max_assistant_tokens < 1
    ):
        raise ValueError("steps/questions/group/turns/token limits are invalid")

    summary: dict[str, Any] = {
        "chapter": 12,
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        **config,
        "profile": args.profile,
        "steps": steps,
        "questions_per_batch": args.questions_per_batch,
        "group_size": group_size,
        "max_code_calls": args.max_code_calls,
        "max_assistant_turns": args.max_assistant_turns,
        "max_trajectory_tokens": args.max_trajectory_tokens,
        "max_assistant_tokens": max_assistant_tokens,
        "max_tool_response_tokens": args.max_tool_response_tokens,
        "force_first_tool_call": args.force_first_tool_call,
        "dynamic_sampling": args.dynamic_sampling,
        "max_question_attempts": args.max_question_attempts,
        "prefer_short_questions": args.prefer_short_questions,
        "external_api_calls": 0,
        "study_order": [
            "experiments/retool/readme.md",
            "experiments/retool/protocol.py",
            "experiments/retool/sandbox.py",
            "experiments/retool/rollout.py",
            "experiments/retool/local_gpu.py",
        ],
    }
    trainer.save_run_config(run_dir, summary)

    try:
        print("Loading DAPO-Math-17k and the local Qwen3-4B model...", flush=True)
        examples = load_examples(
            root,
            args.seed,
            args.max_train_samples,
            prefer_short=args.prefer_short_questions,
        )
        model_dir = trainer.download_modelscope_model(MODEL_ID, root)
        import torch

        device = trainer.check_gpu()
        model, tokenizer, _ = trainer.build_qlora_model(
            model_dir, rank=args.lora_rank, vision=False
        )
        model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
        )
        sandbox = LocalPythonSandbox(
            timeout=args.sandbox_timeout,
            max_workers=args.sandbox_workers,
        )
        rollout_config = RolloutConfig(
            group_size=group_size,
            max_code_calls=args.max_code_calls,
            max_assistant_turns=args.max_assistant_turns,
            max_trajectory_tokens=args.max_trajectory_tokens,
            max_assistant_tokens=max_assistant_tokens,
            max_tool_response_tokens=args.max_tool_response_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            force_first_tool_call=args.force_first_tool_call,
        )

        history: list[dict[str, float]] = []
        next_question = args.start_index
        for step in range(steps):
            sampling_client = trainer.LocalSamplingClient(
                model,
                tokenizer,
                torch,
                device,
                task_id=f"retool-{run_id}-{step}",
            )
            trajectories: list[Any] = []
            question_count = 0
            attempts = args.max_question_attempts if args.dynamic_sampling else 1
            attempt_records: list[tuple[int, list[Any], bool]] = []
            attempts_used = 0
            for attempt in range(1, max(1, attempts) + 1):
                batch = [
                    examples[
                        (next_question + index) % len(examples)
                    ]
                    for index in range(args.questions_per_batch)
                ]
                next_question += args.questions_per_batch
                group = rollout_batch(
                    sampling_client,
                    tokenizer,
                    sandbox,
                    batch,
                    rollout_config,
                )
                attempts_used = attempt
                has_signal = any(
                    abs(float(item.advantage)) > 1e-12 for item in group
                )
                kept = (not args.dynamic_sampling) or has_signal
                attempt_records.append((attempt, group, kept))
                if not kept:
                    continue
                for item in group:
                    item.question_index += question_count
                trajectories.extend(group)
                question_count += len(batch)
                if not args.dynamic_sampling or has_signal:
                    break
            if not trajectories:
                raise RuntimeError(
                    f"No effective question found after {attempts_used} attempts at step {step}; "
                    "refusing to record an optimizer update that did not happen"
                )
            losses, updated = rollout_trainer.train_trajectories(
                model,
                optimizer,
                trajectories,
                loss_fn="ppo",
                clip_low=0.8,
                clip_high=1.28,
                torch_module=torch,
                device=device,
            )
            if updated < 1:
                raise RuntimeError(
                    f"The rollout at step {step} produced no trainable trajectories "
                    "despite a nonzero group advantage"
                )
            metric = {
                "step": step,
                **rollout_metrics(
                    trajectories,
                    losses,
                    updated,
                    question_count,
                    sandbox,
                    groups_sampled=len(attempt_records),
                    groups_kept=sum(kept for _, _, kept in attempt_records),
                ),
                "dynamic_sampling/attempts": float(attempts_used),
            }
            history.append(metric)
            lib.append_metrics(root, 12, run_id, metric)
            print(json.dumps(metric, ensure_ascii=False), flush=True)
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                for attempt, group, kept in attempt_records:
                    for trajectory in group:
                        handle.write(
                            json.dumps(
                                trajectory_record(
                                    step,
                                    trajectory,
                                    attempt=attempt,
                                    kept=kept,
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

        adapter = trainer.save_adapter(model, run_dir)
        (run_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary.update(
            {
                "history": history,
                "adapter": str(adapter),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "dataset_rows": len(examples),
                "effective_updates": sum(
                    float(item.get("train/updated_trajectories", 0.0)) > 0
                    for item in history
                ),
                "updated_trajectories": sum(
                    int(item.get("train/updated_trajectories", 0.0))
                    for item in history
                ),
            }
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        trainer.update_training_state(
            root,
            12,
            run_id,
            "completed",
            {
                "adapter": str(adapter),
                "steps": len(history),
                "effective_updates": summary["effective_updates"],
                "updated_trajectories": summary["updated_trajectories"],
                "summary": str(run_dir / "summary.json"),
                "mechanism_only": True,
            },
            replace_details=True,
        )
        print(f"chapter 12 completed: {run_dir}", flush=True)
        return 0
    except trainer.DataUnavailable as exc:
        (run_dir / "data_unavailable.txt").write_text(str(exc) + "\n", encoding="utf-8")
        trainer.update_training_state(
            root,
            12,
            run_id,
            "data_unavailable",
            {"dataset": DATASET_ID, "reason": str(exc), "note": "no fake success"},
            replace_details=True,
        )
        print(f"chapter 12 data_unavailable: {exc}", flush=True)
        return 0
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        trainer.update_training_state(
            root,
            12,
            run_id,
            "failed",
            {"error": f"{type(exc).__name__}: {exc}", "error_log": str(run_dir / "error.txt")},
            replace_details=True,
        )
        traceback.print_exc()
        return 1
    finally:
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
