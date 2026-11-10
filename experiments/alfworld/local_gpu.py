"""13 ALFWorld: standard grouped GRPO on one local ModelScope GPU.

The original chapter already separates the environment, rollout state machine,
advantage calculation, and PyTRIO trainer cleanly. This local version keeps the
first three unchanged and replaces only the remote sampler and PPO trainer:

    local Qwen3 generation -> original ALFWorld rollout -> group-relative reward
    -> local PPO backward over the same assistant action tokens

The reward remains the original chapter reward:

    task_success - 0.1 * invalid_action_count

so this chapter is a faithful standard-GRPO baseline for the AgentOPSD chapter.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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

sys.modules["pytrio"] = trainer.local_trio_compat()
sys.path.insert(0, str(EXPERIMENT_DIR))

import data as alfworld_data  # noqa: E402


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 GPU 版 ALFWorld GRPO")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--profile",
        default=os.environ.get("RUN_PROFILE", "smoke"),
        choices=("smoke", "full"),
    )
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--games-per-update", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=0)
    parser.add_argument("--max-trajectory-tokens", type=int, default=4096)
    parser.add_argument("--max-action-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ppo-clip-low", type=float, default=0.8)
    parser.add_argument("--ppo-clip-high", type=float, default=1.2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-train-games", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--dynamic-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument(
        "--download-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅当数据缺失时调用官方 alfworld-download；默认要求预先准备数据",
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("ALFWORLD_DATA_ROOT"),
        help="默认 WORKSPACE_ROOT/datasets/alfworld，可指向服务器本地快速解压目录",
    )
    return parser.parse_args()


def _data_root(root: Path, value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else root / "datasets" / "alfworld"


def load_games(
    data_root: Path,
    *,
    seed: int,
    max_games: int,
    download: bool,
) -> list[Any]:
    """Discover solvable text games without treating a missing split as empty."""

    def discover() -> list[Any]:
        directory = alfworld_data.split_directory(data_root, "train")
        if not directory.is_dir():
            raise FileNotFoundError(f"ALFWorld train split not found: {directory}")
        return alfworld_data.shuffled_games(
            data_root,
            "train",
            seed,
            max_games=max_games,
        )

    try:
        games = discover()
    except (FileNotFoundError, ValueError) as first_error:
        if not download:
            raise trainer.DataUnavailable(str(first_error)) from first_error
        print(f"ALFWorld data not found at {data_root}; downloading...", flush=True)
        import shutil
        import subprocess

        executable = shutil.which("alfworld-download")
        if executable is None:
            raise trainer.DataUnavailable(
                "alfworld-download is missing; run scripts/bootstrap_modelscope.sh"
            )
        data_root.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [executable, "--data-dir", str(data_root)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            tail = (result.stdout + "\n" + result.stderr).strip().splitlines()[-12:]
            raise trainer.DataUnavailable(
                f"alfworld-download failed; exit={result.returncode}; " + " | ".join(tail)
            )
        try:
            games = discover()
        except (FileNotFoundError, ValueError) as exc:
            raise trainer.DataUnavailable(str(exc)) from exc
    if not games:
        raise trainer.DataUnavailable(f"No solvable ALFWorld games under {data_root}")
    return games


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def has_mixed_group(trajectories: list[Any]) -> bool:
    groups: dict[str, list[float]] = defaultdict(list)
    for trajectory in trajectories:
        groups[str(trajectory.group_id)].append(float(trajectory.reward))
    return any(len(set(rewards)) > 1 for rewards in groups.values())


def rollout_metrics(
    trajectories: list[Any],
    losses: list[float],
    updated_trajectories: int,
    *,
    groups_sampled: int,
    groups_kept: int,
) -> dict[str, float]:
    steps = [step for trajectory in trajectories for step in trajectory.steps]
    rewards = [float(trajectory.reward) for trajectory in trajectories]
    groups: dict[str, list[float]] = defaultdict(list)
    for trajectory in trajectories:
        groups[str(trajectory.group_id)].append(float(trajectory.reward))
    group_count = max(len(groups), 1)
    return {
        "reward/mean": _mean(rewards),
        "reward/success_rate": _mean([float(item.won) for item in trajectories]),
        "reward/all_success_group_rate": (
            sum(all(value == 1.0 for value in values) for values in groups.values())
            / group_count
        ),
        "reward/all_failure_group_rate": (
            sum(all(value == 0.0 for value in values) for values in groups.values())
            / group_count
        ),
        "advantage/degenerate_group_rate": (
            sum(len(set(values)) == 1 for values in groups.values()) / group_count
        ),
        "trajectory/turns_mean": _mean(
            [float(len(item.steps)) for item in trajectories]
        ),
        "trajectory/action_tokens_mean": _mean(
            [float(sum(len(step.completion_tokens) for step in item.steps)) for item in trajectories]
        ),
        "trajectory/valid_tool_call_rate": _mean(
            [float(step.valid_format) for step in steps]
        ),
        "trajectory/admissible_action_rate": _mean(
            [float(step.admissible) for step in steps]
        ),
        "trajectory/valid_action_rate": _mean(
            [float(step.valid_action) for step in steps]
        ),
        "trajectory/invalid_actions_mean": _mean(
            [float(item.invalid_action_count) for item in trajectories]
        ),
        "train/trajectories": float(len(trajectories)),
        "train/updated_trajectories": float(updated_trajectories),
        "train/loss_mean": _mean(losses),
        "dynamic_sampling/groups_sampled": float(groups_sampled),
        "dynamic_sampling/groups_kept": float(groups_kept),
        "dynamic_sampling/groups_dropped": float(groups_sampled - groups_kept),
    }


def trajectory_record(trajectory: Any) -> dict[str, Any]:
    return {
        "game_id": trajectory.example.id,
        "game_file": str(trajectory.example.game_file),
        "split": trajectory.example.split,
        "task_type": trajectory.example.task_type,
        "group_id": trajectory.group_id,
        "group_index": trajectory.group_index,
        "task": trajectory.task,
        "initial_observation": trajectory.initial_observation,
        "won": trajectory.won,
        "truncated": trajectory.truncated,
        "stop_reason": trajectory.stop_reason,
        "invalid_action_count": trajectory.invalid_action_count,
        "reward": trajectory.reward,
        "advantage": trajectory.advantage,
        "steps": [
            {
                "step": step.index,
                "observation": step.observation,
                "admissible_actions": list(step.admissible_actions),
                "assistant_text": step.assistant_text,
                "action": step.action,
                "valid_format": step.valid_format,
                "admissible": step.admissible,
                "valid_action": step.valid_action,
                "tool_response": step.tool_response,
                "next_observation": step.next_observation,
                "environment_score": step.score,
                "done": step.done,
                "won": step.won,
                "prompt_tokens": len(step.prompt_tokens),
                "completion_tokens": len(step.completion_tokens),
                "mean_old_logprob": (
                    sum(step.logprobs) / len(step.logprobs)
                    if step.logprobs
                    else 0.0
                ),
            }
            for step in trajectory.steps
        ],
    }


def main() -> int:
    args = parse_args()
    root, run_dir, run_id, config = trainer.prepare_run(13, args)
    steps = args.steps or (2 if args.profile == "smoke" else config["steps"])
    group_size = args.group_size or (4 if args.profile == "smoke" else config["group_size"])
    max_turns = args.max_turns or (12 if args.profile == "smoke" else 50)
    max_action_tokens = args.max_action_tokens or (
        384 if args.profile == "smoke" else config["max_new_tokens"]
    )
    if (
        steps < 1
        or args.games_per_update < 1
        or group_size < 2
        or max_turns < 1
        or args.max_trajectory_tokens < 256
        or max_action_tokens < 1
        or args.max_attempts < 1
    ):
        raise ValueError("steps/games/group/turns/token limits are invalid")

    data_root = _data_root(root, args.data_root)
    summary: dict[str, Any] = {
        "chapter": 13,
        "model": MODEL_ID,
        "dataset": "alfworld/json_2.1.1",
        "data_root": str(data_root),
        "profile": args.profile,
        "steps": steps,
        "games_per_update": args.games_per_update,
        "group_size": group_size,
        "max_turns": max_turns,
        "max_trajectory_tokens": args.max_trajectory_tokens,
        "max_action_tokens": max_action_tokens,
        "reward": "task success - 0.1 * invalid_action_count (original chapter reward)",
        "dynamic_sampling": args.dynamic_sampling,
        "max_attempts": args.max_attempts,
        "external_api_calls": 0,
        "study_order": [
            "experiments/alfworld/readme.md",
            "experiments/alfworld/data.py",
            "experiments/alfworld/protocol.py",
            "experiments/alfworld/environment.py",
            "experiments/alfworld/rollout.py",
            "experiments/alfworld/advantages.py",
            "experiments/alfworld/local_gpu.py",
        ],
    }
    trainer.save_run_config(run_dir, summary)

    try:
        games = load_games(
            data_root,
            seed=args.seed,
            max_games=args.max_train_games,
            download=args.download_data,
        )
        print(f"alfworld_games={len(games)}", flush=True)

        from rollout import RolloutConfig, rollout_batch
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
        rollout_config = RolloutConfig(
            group_size=group_size,
            max_steps=max_turns,
            max_trajectory_tokens=args.max_trajectory_tokens,
            max_assistant_tokens=max_action_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            include_admissible_actions=True,
            environment_asynchronous=True,
        )

        history: list[dict[str, float]] = []
        next_game = 0
        for step in range(steps):
            sampling_client = trainer.LocalSamplingClient(
                model,
                tokenizer,
                torch,
                device,
                task_id=f"alfworld-grpo-{run_id}-{step}",
            )
            attempts = args.max_attempts if args.dynamic_sampling else 1
            attempt_records: list[tuple[int, list[Any], bool]] = []
            kept_trajectories: list[Any] = []
            for attempt in range(1, attempts + 1):
                batch = [
                    games[(next_game + index) % len(games)]
                    for index in range(args.games_per_update)
                ]
                next_game += args.games_per_update
                rollout = rollout_batch(
                    sampling_client,
                    tokenizer,
                    batch,
                    rollout_config,
                )
                trajectories = rollout.trajectories
                keep = (not args.dynamic_sampling) or has_mixed_group(trajectories)
                attempt_records.append((attempt, trajectories, keep))
                if keep:
                    kept_trajectories.extend(trajectories)
                    break

            losses, updated = rollout_trainer.train_step_trajectories(
                model,
                optimizer,
                kept_trajectories,
                clip_low=args.ppo_clip_low,
                clip_high=args.ppo_clip_high,
                torch_module=torch,
                device=device,
            )
            recorded = [
                trajectory
                for _, trajectories, _ in attempt_records
                for trajectory in trajectories
            ]
            metric = {
                "step": step,
                **rollout_metrics(
                    recorded,
                    losses,
                    updated,
                    groups_sampled=len(attempt_records),
                    groups_kept=sum(keep for _, _, keep in attempt_records),
                ),
                "train/kept_trajectories": float(len(kept_trajectories)),
            }
            history.append(metric)
            lib.append_metrics(root, 13, run_id, metric)
            print(json.dumps(metric, ensure_ascii=False), flush=True)
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                for attempt, trajectories, keep in attempt_records:
                    for trajectory in trajectories:
                        handle.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "attempt": attempt,
                                    "kept": keep,
                                    **trajectory_record(trajectory),
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
            }
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        trainer.update_training_state(
            root,
            13,
            run_id,
            "completed",
            {
                "adapter": str(adapter),
                "steps": len(history),
                "summary": str(run_dir / "summary.json"),
                "mechanism_only": True,
                "reward": "task success - 0.1 * invalid_action_count",
            },
            replace_details=True,
        )
        print(f"chapter 13 completed: {run_dir}", flush=True)
        return 0
    except trainer.DataUnavailable as exc:
        (run_dir / "data_unavailable.txt").write_text(str(exc) + "\n", encoding="utf-8")
        trainer.update_training_state(
            root,
            13,
            run_id,
            "data_unavailable",
            {"dataset": "alfworld", "reason": str(exc), "note": "no fake success"},
            replace_details=True,
        )
        print(f"chapter 13 data_unavailable: {exc}", flush=True)
        return 0
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        trainer.update_training_state(
            root,
            13,
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
