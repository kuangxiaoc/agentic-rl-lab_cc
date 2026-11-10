"""14 TEMPO: macro-step actor/critic training on one local ModelScope GPU.

The original chapter already contains the hard TEMPO pieces: deterministic
environment replay, macro-step boundaries, the privileged generative critic,
TD returns, and critic error rewards. This local version reuses those modules
and replaces only PyTRIO sampling, logprobs, backward, and optimizer updates.

The same Qwen3-4B adapter acts as actor and critic through different prompts.
Warm-up trains the critic with full-trajectory Monte Carlo returns; TD updates
then train both roles from saved macro-step states.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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

sys.modules["pytrio"] = trainer.local_trio_compat()
sys.path.insert(0, str(EXPERIMENT_DIR))


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 GPU 版 TEMPO")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--profile",
        default=os.environ.get("RUN_PROFILE", "smoke"),
        choices=("smoke", "full"),
    )
    parser.add_argument("--warmup-updates", type=int, default=0)
    parser.add_argument("--td-updates", type=int, default=0)
    parser.add_argument("--warmup-games-per-batch", type=int, default=1)
    parser.add_argument("--states-per-batch", type=int, default=1)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--critic-samples", type=int, default=2)
    parser.add_argument("--endpoint-samples", type=int, default=1)
    parser.add_argument("--macro-rounds", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=8)
    parser.add_argument("--state-store-capacity", type=int, default=32)
    parser.add_argument("--max-trajectory-tokens", type=int, default=4096)
    parser.add_argument("--max-assistant-tokens", type=int, default=256)
    parser.add_argument("--critic-max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ppo-clip-low", type=float, default=0.8)
    parser.add_argument("--ppo-clip-high", type=float, default=1.2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-train-games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("ALFWORLD_DATA_ROOT"),
    )
    return parser.parse_args()


def _data_root(root: Path, value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else root / "datasets" / "alfworld"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _record_trajectory(trajectory: Any) -> dict[str, Any]:
    return {
        "game_id": trajectory.example.id,
        "task_type": trajectory.example.task_type,
        "group_id": trajectory.group_id,
        "group_index": trajectory.group_index,
        "start_round": trajectory.start_round,
        "task": trajectory.task,
        "won": trajectory.won,
        "truncated": trajectory.truncated,
        "stop_reason": trajectory.stop_reason,
        "segment_reward": trajectory.segment_reward,
        "endpoint_value": trajectory.endpoint_value,
        "branch_return": trajectory.branch_return,
        "advantage": trajectory.advantage,
        "steps": [
            {
                "step": step.index,
                "observation": step.observation,
                "assistant_text": step.assistant_text,
                "action": step.action,
                "valid_format": step.valid_format,
                "admissible": step.admissible,
                "valid_action": step.valid_action,
                "next_observation": step.next_observation,
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


def _record_critic_sample(sample: Any) -> dict[str, Any]:
    return {
        "prompt_tokens": len(sample.prompt_tokens),
        "completion_tokens": len(sample.completion_tokens),
        "text": sample.text,
        "value": sample.value,
    }


def _main() -> int:
    args = parse_args()
    root, run_dir, run_id, config = trainer.prepare_run(14, args)
    warmup_updates = args.warmup_updates or (1 if args.profile == "smoke" else 2)
    td_updates = args.td_updates or (1 if args.profile == "smoke" else config["steps"])
    if (
        warmup_updates < 0
        or td_updates < 0
        or warmup_updates + td_updates < 1
        or args.warmup_games_per_batch < 1
        or args.states_per_batch < 1
        or args.branches < 2
        or args.critic_samples < 2
        or args.endpoint_samples < 1
        or args.macro_rounds < 1
        or args.max_episode_steps % args.macro_rounds != 0
        or args.max_trajectory_tokens < 256
        or args.max_assistant_tokens < 1
        or args.critic_max_tokens < 1
    ):
        raise ValueError("TEMPO update/macro/token limits are invalid")

    data_root = _data_root(root, args.data_root)
    summary: dict[str, Any] = {
        "chapter": 14,
        "model": MODEL_ID,
        "dataset": "alfworld/json_2.1.1",
        "data_root": str(data_root),
        "profile": args.profile,
        "warmup_updates": warmup_updates,
        "td_updates": td_updates,
        "branches": args.branches,
        "critic_samples": args.critic_samples,
        "endpoint_samples": args.endpoint_samples,
        "macro_rounds": args.macro_rounds,
        "max_episode_steps": args.max_episode_steps,
        "max_trajectory_tokens": args.max_trajectory_tokens,
        "max_assistant_tokens": args.max_assistant_tokens,
        "critic_max_tokens": args.critic_max_tokens,
        "actor_critic": "same local Qwen3 adapter, different prompts",
        "privileged_critic": "game.tw-pddl walkthrough",
        "external_api_calls": 0,
        "study_order": [
            "experiments/tempo/readme.md",
            "experiments/tempo/environment.py",
            "experiments/tempo/states.py",
            "experiments/tempo/rollout.py",
            "experiments/tempo/critic.py",
            "experiments/tempo/tempo.py",
            "experiments/tempo/local_gpu.py",
        ],
    }
    trainer.save_run_config(run_dir, summary)

    try:
        from critic import (
            CriticRequest,
            critic_prompt_for_state,
            endpoint_value,
            sample_critic_values,
        )
        from data import load_walkthrough, shuffled_games, take_batch
        from rollout import RolloutConfig, rollout_macro_steps
        from states import StateStore, boundary_states, endpoint_state
        from tempo import assemble_actor_group, assemble_critic_group

        import torch

        games = shuffled_games(
            data_root,
            "train",
            args.seed,
            max_games=args.max_train_games,
        )
        if not games:
            raise trainer.DataUnavailable(f"No solvable alfworld games under {data_root}")
        print(f"alfworld_games={len(games)}", flush=True)

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
            branches=args.branches,
            macro_rounds=args.macro_rounds,
            max_episode_steps=args.max_episode_steps,
            max_trajectory_tokens=args.max_trajectory_tokens,
            max_assistant_tokens=args.max_assistant_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            include_admissible_actions=True,
            environment_asynchronous=True,
        )
        from dataclasses import replace

        warmup_config = replace(
            rollout_config,
            macro_rounds=args.max_episode_steps,
        )
        store = StateStore(args.state_store_capacity, seed=args.seed + 1)
        history: list[dict[str, Any]] = []
        cursor = 0
        update_index = 0

        def critic_budget() -> int:
            return args.max_trajectory_tokens - args.critic_max_tokens

        def sampling_client(label: str) -> Any:
            return trainer.LocalSamplingClient(
                model,
                tokenizer,
                torch,
                device,
                task_id=f"tempo-{run_id}-{label}",
            )

        for warmup_index in range(warmup_updates):
            client = sampling_client(f"warmup-{warmup_index}")
            fresh = take_batch(games, cursor, args.warmup_games_per_batch)
            cursor += len(fresh)
            groups = rollout_macro_steps(
                client,
                tokenizer,
                [],
                fresh,
                warmup_config,
                seed_offset=update_index * 7919,
            )
            snapshots: list[Any] = []
            mc_returns: list[float] = []
            for group in groups:
                for trajectory in group.trajectories:
                    won = 1.0 if trajectory.won else 0.0
                    for state in boundary_states(
                        group.start_state,
                        trajectory,
                        macro_rounds=args.macro_rounds,
                        max_trajectory_tokens=args.max_trajectory_tokens,
                    ):
                        snapshots.append(state)
                        mc_returns.append(won)

            requests: list[CriticRequest] = []
            kept: list[tuple[Any, float]] = []
            for index, (state, mc_return) in enumerate(
                zip(snapshots, mc_returns, strict=True)
            ):
                prompt = critic_prompt_for_state(
                    tokenizer,
                    state.messages,
                    task=state.task,
                    walkthrough=load_walkthrough(state.example.game_file),
                )
                if len(prompt) > critic_budget():
                    continue
                kept.append((state, mc_return))
                requests.append(
                    CriticRequest(
                        prompt_tokens=prompt,
                        num_samples=args.critic_samples,
                        seed=args.seed + update_index * 7919 + index,
                    )
                )
            batches = sample_critic_values(
                client,
                tokenizer,
                requests,
                max_tokens=args.critic_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            samples: list[trainer.Sample] = []
            critic_records: list[dict[str, Any]] = []
            abs_errors: list[float] = []
            parse_failures = 0
            for (state, mc_return), critic_samples in zip(
                kept, batches, strict=True
            ):
                signals = assemble_critic_group(critic_samples, mc_return)
                abs_errors.extend(signals.errors)
                parse_failures += signals.parse_failures
                for critic_sample, advantage, reward in zip(
                    critic_samples,
                    signals.advantages,
                    signals.rewards,
                    strict=True,
                ):
                    critic_records.append(
                        {
                            **_record_critic_sample(critic_sample),
                            "mc_return": mc_return,
                            "reward": reward,
                            "advantage": advantage,
                        }
                    )
                    if abs(advantage) <= 1e-12:
                        continue
                    sample = rollout_trainer.build_critic_sample(
                        critic_sample,
                        advantage,
                        reward=reward,
                    )
                    if sample is not None:
                        samples.append(sample)

            losses, updated = rollout_trainer.train_mixed_samples(
                model,
                optimizer,
                samples,
                clip_low=args.ppo_clip_low,
                clip_high=args.ppo_clip_high,
                torch_module=torch,
                device=device,
            )
            trajectories = [
                trajectory
                for group in groups
                for trajectory in group.trajectories
            ]
            steps = [
                step for trajectory in trajectories for step in trajectory.steps
            ]
            metric = {
                "update": update_index,
                "phase": "warmup",
                "warmup/success_rate": _mean(
                    [float(trajectory.won) for trajectory in trajectories]
                ),
                "warmup/steps_mean": _mean(
                    [float(len(trajectory.steps)) for trajectory in trajectories]
                ),
                "warmup/boundary_states": float(len(snapshots)),
                "target/mc_return_mean": _mean(mc_returns),
                "critic/samples": float(len(critic_records)),
                "critic/abs_error_mean": _mean(abs_errors),
                "critic/parse_failure_rate": (
                    parse_failures / len(critic_records)
                    if critic_records
                    else 0.0
                ),
                "rollout/valid_action_rate": _mean(
                    [float(step.valid_action) for step in steps]
                ),
                "train/samples": float(len(samples)),
                "train/updated_samples": float(updated),
                "train/loss_mean": _mean(losses),
            }
            history.append(metric)
            lib.append_metrics(root, 14, run_id, metric)
            print(json.dumps(metric, ensure_ascii=False), flush=True)
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                for trajectory in trajectories:
                    handle.write(
                        json.dumps(
                            {
                                "update": update_index,
                                "phase": "warmup",
                                **_record_trajectory(trajectory),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                for record in critic_records:
                    handle.write(
                        json.dumps(
                            {
                                "update": update_index,
                                "phase": "warmup-critic",
                                **record,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            update_index += 1

        for td_index in range(td_updates):
            client = sampling_client(f"td-{td_index}")
            start_states = store.sample(args.states_per_batch)
            fresh_count = args.states_per_batch - len(start_states)
            fresh = take_batch(games, cursor, fresh_count) if fresh_count else []
            cursor += len(fresh)
            groups = rollout_macro_steps(
                client,
                tokenizer,
                start_states,
                fresh,
                rollout_config,
                seed_offset=update_index * 7919,
            )

            endpoint_requests: list[tuple[int, int, CriticRequest]] = []
            for group_index, group in enumerate(groups):
                for branch_index, trajectory in enumerate(group.trajectories):
                    if trajectory.stop_reason != "macro_boundary":
                        continue
                    prompt = critic_prompt_for_state(
                        tokenizer,
                        trajectory.messages,
                        task=trajectory.task,
                        walkthrough=load_walkthrough(trajectory.example.game_file),
                    )
                    if len(prompt) > critic_budget():
                        continue
                    endpoint_requests.append(
                        (
                            group_index,
                            branch_index,
                            CriticRequest(
                                prompt_tokens=prompt,
                                num_samples=args.endpoint_samples,
                                seed=(
                                    args.seed
                                    + update_index * 7919
                                    + group_index * 10_000
                                    + branch_index
                                ),
                            ),
                        )
                    )
            endpoint_batches = sample_critic_values(
                client,
                tokenizer,
                [request for _, _, request in endpoint_requests],
                max_tokens=args.critic_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            endpoint_records: list[dict[str, Any]] = []
            for (group_index, branch_index, _), critic_samples in zip(
                endpoint_requests, endpoint_batches, strict=True
            ):
                value, failures = endpoint_value(critic_samples)
                groups[group_index].trajectories[branch_index].endpoint_value = value
                endpoint_records.append(
                    {
                        "group_index": group_index,
                        "branch_index": branch_index,
                        "value": value,
                        "parse_failures": failures,
                        "samples": [
                            _record_critic_sample(sample)
                            for sample in critic_samples
                        ],
                    }
                )

            actor_signals = [assemble_actor_group(group) for group in groups]
            critic_requests: list[CriticRequest] = []
            critic_states: list[tuple[Any, float]] = []
            for index, group in enumerate(groups):
                prompt = critic_prompt_for_state(
                    tokenizer,
                    group.start_state.messages,
                    task=group.start_state.task,
                    walkthrough=load_walkthrough(group.start_state.example.game_file),
                )
                if len(prompt) > critic_budget():
                    continue
                critic_states.append((group, actor_signals[index].value_target))
                critic_requests.append(
                    CriticRequest(
                        prompt_tokens=prompt,
                        num_samples=args.critic_samples,
                        seed=args.seed + update_index * 7919 + 100_000 + index,
                    )
                )
            critic_batches = sample_critic_values(
                client,
                tokenizer,
                critic_requests,
                max_tokens=args.critic_max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            samples: list[trainer.Sample] = []
            for group in groups:
                for trajectory in group.trajectories:
                    if abs(float(trajectory.advantage)) <= 1e-12:
                        continue
                    sample = rollout_trainer.build_segment_sample(
                        group.start_state, trajectory
                    )
                    if sample is not None:
                        samples.append(sample)

            critic_records = []
            abs_errors: list[float] = []
            parse_failures = 0
            for (group, value_target), critic_samples in zip(
                critic_states, critic_batches, strict=True
            ):
                signals = assemble_critic_group(critic_samples, value_target)
                abs_errors.extend(signals.errors)
                parse_failures += signals.parse_failures
                for critic_sample, advantage, reward in zip(
                    critic_samples,
                    signals.advantages,
                    signals.rewards,
                    strict=True,
                ):
                    critic_records.append(
                        {
                            **_record_critic_sample(critic_sample),
                            "value_target": value_target,
                            "reward": reward,
                            "advantage": advantage,
                        }
                    )
                    if abs(advantage) <= 1e-12:
                        continue
                    sample = rollout_trainer.build_critic_sample(
                        critic_sample,
                        advantage,
                        reward=reward,
                    )
                    if sample is not None:
                        samples.append(sample)

            losses, updated = rollout_trainer.train_mixed_samples(
                model,
                optimizer,
                samples,
                clip_low=args.ppo_clip_low,
                clip_high=args.ppo_clip_high,
                torch_module=torch,
                device=device,
            )
            saved_endpoints: list[Any] = []
            for group in groups:
                for trajectory in group.trajectories:
                    state = endpoint_state(
                        group.start_state,
                        trajectory,
                        config=rollout_config,
                    )
                    if state is not None:
                        saved_endpoints.append(state)
            store.add(saved_endpoints)

            trajectories = [
                trajectory
                for group in groups
                for trajectory in group.trajectories
            ]
            steps = [
                step for trajectory in trajectories for step in trajectory.steps
            ]
            metric = {
                "update": update_index,
                "phase": "td",
                "td/store_size": float(len(store)),
                "td/new_endpoints": float(len(saved_endpoints)),
                "td/fresh_states": float(len(fresh)),
                "actor/segment_reward_mean": _mean(
                    [trajectory.segment_reward for trajectory in trajectories]
                ),
                "actor/terminal_won_rate": _mean(
                    [float(trajectory.won) for trajectory in trajectories]
                ),
                "actor/macro_boundary_rate": (
                    sum(
                        trajectory.stop_reason == "macro_boundary"
                        for trajectory in trajectories
                    )
                    / len(trajectories)
                    if trajectories
                    else 0.0
                ),
                "actor/degenerate_group_rate": _mean(
                    [float(signal.degenerate) for signal in actor_signals]
                ),
                "actor/value_target_mean": _mean(
                    [signal.value_target for signal in actor_signals]
                ),
                "critic/abs_error_mean": _mean(abs_errors),
                "critic/parse_failure_rate": (
                    parse_failures / len(critic_records)
                    if critic_records
                    else 0.0
                ),
                "critic/endpoint_value_mean": _mean(
                    [record["value"] for record in endpoint_records]
                ),
                "rollout/valid_action_rate": _mean(
                    [float(step.valid_action) for step in steps]
                ),
                "train/samples": float(len(samples)),
                "train/updated_samples": float(updated),
                "train/loss_mean": _mean(losses),
            }
            history.append(metric)
            lib.append_metrics(root, 14, run_id, metric)
            print(json.dumps(metric, ensure_ascii=False), flush=True)
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                for trajectory in trajectories:
                    handle.write(
                        json.dumps(
                            {
                                "update": update_index,
                                "phase": "td",
                                **_record_trajectory(trajectory),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                for record in endpoint_records:
                    handle.write(
                        json.dumps(
                            {
                                "update": update_index,
                                "phase": "td-endpoint-critic",
                                **record,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                for record in critic_records:
                    handle.write(
                        json.dumps(
                            {
                                "update": update_index,
                                "phase": "td-start-critic",
                                **record,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            update_index += 1

            if args.save_every > 0 and (td_index + 1) % args.save_every == 0:
                trainer.save_adapter(model, run_dir / f"adapter-update-{update_index}")

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
            14,
            run_id,
            "completed",
            {
                "adapter": str(adapter),
                "updates": len(history),
                "summary": str(run_dir / "summary.json"),
                "mechanism_only": True,
                "actor_critic": "same adapter, different prompts",
            },
            replace_details=True,
        )
        print(f"chapter 14 completed: {run_dir}", flush=True)
        return 0
    except trainer.DataUnavailable as exc:
        (run_dir / "data_unavailable.txt").write_text(str(exc) + "\n", encoding="utf-8")
        trainer.update_training_state(
            root,
            14,
            run_id,
            "data_unavailable",
            {"dataset": "alfworld", "reason": str(exc), "note": "no fake success"},
            replace_details=True,
        )
        print(f"chapter 14 data_unavailable: {exc}", flush=True)
        return 0
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        trainer.update_training_state(
            root,
            14,
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
    raise SystemExit(_main())
