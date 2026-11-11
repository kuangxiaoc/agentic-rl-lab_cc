"""10 AgentOPSD: turn-level credit on one local ModelScope GPU.

This keeps the original AgentOPSD environment, rollout state machine, SkillBank,
Teacher evidence alignment, and GRPO/turn-credit math. The remote PyTRIO
sampler/trainer is replaced with one 4-bit QLoRA model on the local GPU:

    student rollout without skills
    -> same model with a privileged skill scores the exact action tokens
    -> per-turn evidence becomes turn-level advantages
    -> one PPO update over the local LoRA adapter

The Teacher is not a second downloaded model. It is the current policy snapshot
with a different prompt. The adapter is updated only after Teacher scoring, so
Student and Teacher see the same weights within one update.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import local_gpu_trainer as trainer  # noqa: E402
import local_rollout_trainer as rollout_trainer  # noqa: E402
import modelscope_lib as lib  # noqa: E402

# The original modules import PyTRIO when they are imported. Inject the small
# local sampling/logprob surface before loading the original state machine.
sys.modules["pytrio"] = trainer.local_trio_compat()
sys.path.insert(0, str(EXPERIMENT_DIR))

import data as agent_data  # noqa: E402


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
ALFWORLD_URL = "https://github.com/alfworld/alfworld/releases"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 GPU 版 AgentOPSD / ALFWorld")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--profile",
        default=os.environ.get("RUN_PROFILE", "smoke"),
        choices=("smoke", "full"),
    )
    parser.add_argument("--steps", type=int, default=0, help="0 表示使用 profile 默认值")
    parser.add_argument("--tasks-per-update", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=0, help="0 表示使用 profile 默认值")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-trajectory-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-action-tokens",
        type=int,
        default=0,
        help="单轮 assistant 最大 token；0 表示使用 profile 默认值",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--teacher-concurrency", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--reshape-lambda", type=float, default=0.5)
    parser.add_argument("--weight-bound", type=float, default=0.2)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--ppo-clip-low", type=float, default=0.8)
    parser.add_argument("--ppo-clip-high", type=float, default=1.24)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-train-games", type=int, default=64)
    parser.add_argument(
        "--task-types",
        default="",
        help="逗号分隔的教学任务子集；留空表示使用原版支持的全部类型",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--dynamic-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="跳过整组全成功/全失败的更新，最多重试 max-attempts 次",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--download-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="数据不存在时调用官方 alfworld-download",
    )
    parser.add_argument("--data-root", default=None, help="默认 WORKSPACE_ROOT/datasets/alfworld")
    parser.add_argument("--skills-dir", default=None)
    return parser.parse_args()


def _data_root(root: Path, value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else root / "datasets" / "alfworld"


def _skills_dir(value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else EXPERIMENT_DIR / "skills" / "alfworld"


def ensure_alfworld_dependencies() -> None:
    """Fail with an actionable message instead of importing a half environment."""
    missing = [
        name
        for name in ("alfworld", "textworld", "spacy")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "ALFWorld dependencies are missing: "
            + ", ".join(missing)
            + ". Run `bash scripts/bootstrap_modelscope.sh` on the server first."
        )


def download_alfworld(data_root: Path) -> None:
    """Download only the official ALFWorld assets into persistent storage."""
    executable = shutil.which("alfworld-download")
    if executable is None:
        raise trainer.DataUnavailable(
            "alfworld-download is not installed; run scripts/bootstrap_modelscope.sh"
        )
    data_root.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [executable, "--data-dir", str(data_root)],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise trainer.DataUnavailable(
            f"Could not start alfworld-download: {type(exc).__name__}: {exc}"
        ) from exc
    if result.returncode != 0:
        tail = (result.stdout + "\n" + result.stderr).strip().splitlines()[-12:]
        raise trainer.DataUnavailable(
            "alfworld-download failed; no data was fabricated. "
            f"exit={result.returncode}; " + " | ".join(tail)
        )


def load_games(
    data_root: Path,
    *,
    seed: int,
    max_games: int,
    download: bool,
    task_types: set[str] | frozenset[str] | None = None,
) -> list[Any]:
    """Discover text-only games, downloading once only when explicitly allowed."""
    def discover() -> list[Any]:
        directory = agent_data.split_directory(data_root, "train")
        if not directory.is_dir():
            raise FileNotFoundError(f"ALFWorld train split not found: {directory}")
        return agent_data.shuffled_games(
            data_root,
            "train",
            seed,
            max_games=max_games,
            task_types=task_types,
        )

    try:
        games = discover()
    except (FileNotFoundError, ValueError) as first_error:
        if not download:
            raise trainer.DataUnavailable(str(first_error)) from first_error
        print(f"ALFWorld data not found at {data_root}; downloading...", flush=True)
        download_alfworld(data_root)
        try:
            games = discover()
        except (FileNotFoundError, ValueError) as exc:
            raise trainer.DataUnavailable(str(exc)) from exc
    if not games:
        raise trainer.DataUnavailable(f"No solvable ALFWorld games under {data_root}")
    return games


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_rewards(trajectories: list[Any]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for trajectory in trajectories:
        groups[str(trajectory.group_id)].append(float(trajectory.reward))
    return groups


def has_mixed_group(trajectories: list[Any]) -> bool:
    """A group is useful only when its rewards are not all equal."""
    return any(
        len(set(rewards)) > 1
        for rewards in _group_rewards(trajectories).values()
    )


def rollout_metrics(
    trajectories: list[Any],
    losses: list[float],
    updated_trajectories: int,
    *,
    groups_sampled: int,
    groups_kept: int,
    teacher_requests: int,
    teacher_tokens: int,
) -> dict[str, float]:
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    credits = [credit for trajectory in trajectories for credit in trajectory.turn_credits]
    rewards = [float(trajectory.reward) for trajectory in trajectories]
    groups = _group_rewards(trajectories)
    all_success = sum(all(value == 1.0 for value in values) for values in groups.values())
    all_failure = sum(all(value == 0.0 for value in values) for values in groups.values())
    group_count = max(len(groups), 1)
    multipliers = [
        float(credit.advantage) / float(trajectory.sequence_advantage)
        for trajectory in trajectories
        if abs(float(trajectory.sequence_advantage)) > 1e-12
        for credit in trajectory.turn_credits
    ]
    metrics = {
        "reward/mean": _mean(rewards),
        "reward/success_rate": _mean([float(item.won) for item in trajectories]),
        "reward/all_success_group_rate": all_success / group_count,
        "reward/all_failure_group_rate": all_failure / group_count,
        "trajectory/turns_mean": _mean([float(len(item.steps)) for item in trajectories]),
        "trajectory/action_tokens_mean": _mean(
            [float(sum(len(step.completion_tokens) for step in item.steps)) for item in trajectories]
        ),
        "trajectory/valid_tool_call_rate": _mean([float(step.valid_format) for step in steps]),
        "trajectory/admissible_action_rate": _mean([float(step.admissible) for step in steps]),
        "trajectory/valid_action_rate": _mean([float(step.valid_action) for step in steps]),
        "advantage/degenerate_group_rate": (
            sum(len(set(values)) == 1 for values in groups.values()) / group_count
        ),
        "opsd/evidence_mean": _mean([float(credit.evidence) for credit in credits]),
        "opsd/evidence_abs_mean": _mean([abs(float(credit.evidence)) for credit in credits]),
        "opsd/weight_mean": _mean([float(credit.weight) for credit in credits]),
        "opsd/adv_multiplier_min": min(multipliers, default=1.0),
        "opsd/adv_multiplier_max": max(multipliers, default=1.0),
        "teacher/requests": float(teacher_requests),
        "teacher/tokens": float(teacher_tokens),
        "train/trajectories": float(len(trajectories)),
        "train/updated_trajectories": float(updated_trajectories),
        "train/loss_mean": _mean(losses),
        "dynamic_sampling/groups_sampled": float(groups_sampled),
        "dynamic_sampling/groups_kept": float(groups_kept),
        "dynamic_sampling/groups_dropped": float(groups_sampled - groups_kept),
    }
    return metrics


def main() -> int:
    args = parse_args()
    root, run_dir, run_id, config = trainer.prepare_run(10, args)
    steps = args.steps or (2 if args.profile == "smoke" else config["steps"])
    group_size = args.group_size or (2 if args.profile == "smoke" else config["group_size"])
    max_action_tokens = args.max_action_tokens or (
        256 if args.profile == "smoke" else config["max_new_tokens"]
    )
    task_types = {
        item.strip()
        for item in args.task_types.split(",")
        if item.strip()
    } or None
    unknown_task_types = (task_types or set()) - agent_data.SUPPORTED_TASK_TYPES
    if unknown_task_types:
        raise ValueError(
            "unknown --task-types: " + ", ".join(sorted(unknown_task_types))
        )
    if (
        steps < 1
        or args.tasks_per_update < 1
        or group_size < 2
        or args.max_turns < 1
        or args.max_trajectory_tokens < 256
        or max_action_tokens < 1
        or args.teacher_concurrency < 1
    ):
        raise ValueError("steps/tasks/group/turns/token/concurrency limits are invalid")

    data_root = _data_root(root, args.data_root)
    skills_dir = _skills_dir(args.skills_dir)
    summary: dict[str, Any] = {
        "chapter": 10,
        "model": MODEL_ID,
        "dataset": "alfworld/json_2.1.1",
        "data_root": str(data_root),
        "profile": args.profile,
        "steps": steps,
        "tasks_per_update": args.tasks_per_update,
        "group_size": group_size,
        "max_turns": args.max_turns,
        "max_trajectory_tokens": args.max_trajectory_tokens,
        "max_action_tokens": max_action_tokens,
        "task_types": sorted(task_types) if task_types else "all-supported",
        "dynamic_sampling": args.dynamic_sampling,
        "max_attempts": args.max_attempts,
        "teacher": "same local policy snapshot with privileged skill prompt",
        "external_api_calls": 0,
        "study_order": [
            "experiments/agentopsd/readme.md",
            "experiments/agentopsd/rollout.py",
            "experiments/agentopsd/teacher.py",
            "experiments/agentopsd/advantages.py",
            "experiments/agentopsd/local_gpu.py",
        ],
    }
    trainer.save_run_config(run_dir, summary)

    try:
        ensure_alfworld_dependencies()
        games = load_games(
            data_root,
            seed=args.seed,
            max_games=args.max_train_games,
            download=args.download_data,
            task_types=task_types,
        )
        print(f"alfworld_games={len(games)}", flush=True)

        from advantages import AgentOPSDConfig, assign_agentopsd_advantages
        from rollout import RolloutConfig, rollout_batch, trajectory_record
        from skills import SkillProvider
        from teacher import score_trajectories_async

        import asyncio
        import torch

        model_dir = trainer.download_modelscope_model(MODEL_ID, root)
        device = trainer.check_gpu()
        model, tokenizer, _ = trainer.build_qlora_model(
            model_dir,
            rank=args.lora_rank,
            vision=False,
        )
        model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
        )
        skill_provider = SkillProvider(skills_dir)
        rollout_config = RolloutConfig(
            group_size=group_size,
            max_turns=args.max_turns,
            max_trajectory_tokens=args.max_trajectory_tokens,
            max_assistant_tokens=max_action_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            include_admissible_actions=True,
            environment_asynchronous=True,
        )
        agentopsd_config = AgentOPSDConfig(
            gamma=args.gamma,
            reshape_lambda=args.reshape_lambda,
            weight_bound=args.weight_bound,
            epsilon=args.epsilon,
        )

        history: list[dict[str, float]] = []
        next_game = 0
        for step in range(steps):
            sampling_client = trainer.LocalSamplingClient(
                model,
                tokenizer,
                torch,
                device,
                task_id=f"agentopsd-{run_id}-{step}",
            )
            attempts = args.max_attempts if args.dynamic_sampling else 1
            attempt_records: list[tuple[int, list[Any], bool]] = []
            kept_trajectories: list[Any] = []
            teacher_requests = 0
            teacher_tokens = 0

            for attempt in range(1, max(1, attempts) + 1):
                batch = [
                    games[(next_game + index) % len(games)]
                    for index in range(args.tasks_per_update)
                ]
                next_game += args.tasks_per_update
                rollout = rollout_batch(
                    sampling_client,
                    tokenizer,
                    batch,
                    rollout_config,
                    policy_snapshot_id=f"local-{run_id}-{step}-{attempt}",
                )
                trajectories = rollout.trajectories
                print(
                    json.dumps(
                        {
                            "event": "rollout",
                            "step": step,
                            "attempt": attempt,
                            "rewards": [trajectory.reward for trajectory in trajectories],
                            "turns": [len(trajectory.steps) for trajectory in trajectories],
                            "action_tokens": [
                                sum(len(step.completion_tokens) for step in trajectory.steps)
                                for trajectory in trajectories
                            ],
                            "stop_reasons": [
                                trajectory.stop_reason for trajectory in trajectories
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                keep = (not args.dynamic_sampling) or has_mixed_group(trajectories)
                attempt_records.append((attempt, trajectories, keep))
                if not keep:
                    continue

                _, teacher_stats = asyncio.run(
                    score_trajectories_async(
                        sampling_client,
                        tokenizer,
                        trajectories,
                        skill_provider,
                        max_concurrency=args.teacher_concurrency,
                    )
                )
                teacher_requests += teacher_stats.requests
                teacher_tokens += teacher_stats.teacher_input_tokens
                assign_agentopsd_advantages(trajectories, agentopsd_config)
                kept_trajectories.extend(trajectories)
                break

            losses, updated = rollout_trainer.train_turn_advantage_trajectories(
                model,
                optimizer,
                kept_trajectories,
                clip_low=args.ppo_clip_low,
                clip_high=args.ppo_clip_high,
                torch_module=torch,
                device=device,
            )
            if args.dynamic_sampling and not kept_trajectories:
                raise RuntimeError(
                    f"No mixed-reward group found after {attempts} attempts at step {step}; "
                    "refusing to record an optimizer update that did not happen"
                )
            recorded_trajectories = [
                trajectory
                for _, trajectories, _ in attempt_records
                for trajectory in trajectories
            ]
            metric = {
                "step": step,
                **rollout_metrics(
                    recorded_trajectories,
                    losses,
                    updated,
                    groups_sampled=len(attempt_records),
                    groups_kept=sum(keep for _, _, keep in attempt_records),
                    teacher_requests=teacher_requests,
                    teacher_tokens=teacher_tokens,
                ),
                "train/kept_trajectories": float(len(kept_trajectories)),
            }
            history.append(metric)
            lib.append_metrics(root, 10, run_id, metric)
            print(json.dumps(metric, ensure_ascii=False), flush=True)
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                for attempt, trajectories, keep in attempt_records:
                    for trajectory in trajectories:
                        record = trajectory_record(trajectory)
                        handle.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "attempt": attempt,
                                    "kept": keep,
                                    **record,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                trainer.save_adapter(model, run_dir / f"adapter-step-{step + 1}")

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
                "games_available": len(games),
                "effective_updates": sum(
                    float(item.get("train/updated_trajectories", 0.0)) > 0
                    for item in history
                ),
                "attempted_groups": sum(
                    int(item.get("dynamic_sampling/groups_sampled", 0.0))
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
            10,
            run_id,
            "completed",
            {
                "adapter": str(adapter),
                "steps": len(history),
                "effective_updates": summary["effective_updates"],
                "summary": str(run_dir / "summary.json"),
                "mechanism_only": True,
                "teacher": "same local policy snapshot with skill prompt",
            },
            replace_details=True,
        )
        print(f"chapter 10 completed: {run_dir}", flush=True)
        return 0
    except trainer.DataUnavailable as exc:
        (run_dir / "data_unavailable.txt").write_text(str(exc) + "\n", encoding="utf-8")
        trainer.update_training_state(
            root,
            10,
            run_id,
            "data_unavailable",
            {"dataset": "ALFWorld", "reason": str(exc), "note": "no fake success"},
            replace_details=True,
        )
        print(f"chapter 10 data_unavailable: {exc}", flush=True)
        return 0
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        trainer.update_training_state(
            root,
            10,
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
