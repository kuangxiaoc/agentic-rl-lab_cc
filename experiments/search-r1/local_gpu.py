"""11 Search-R1: local-GPU mechanism run with an offline evidence backend.

This preserves the original Search-R1 rollout, tool protocol, reward, group
advantage, and policy-update loop.  The only substitution is the search service:
the GPU box in this workspace cannot reach Wikipedia or DeepSeek Search, so the
run uses ``local_search.py``, an explicitly labelled oracle-like lexical corpus.

Do not read this chapter's correctness numbers as an online Search-R1 result.
The useful learning signal here is the multi-turn state machine and the
retrieved-observation mask, not retrieval quality.
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

sys.modules["pytrio"] = trainer.local_trio_compat()
sys.path.insert(0, str(EXPERIMENT_DIR))

import data as search_data  # noqa: E402
import local_search  # noqa: E402

# ``rollout.py`` imports ``search``.  Inject the offline module under that name
# so the original state machine is reused without importing httpx/dotenv.
sys.modules["search"] = local_search
from rollout import RolloutConfig, rollout_batch  # noqa: E402


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_ID = "zhuangzhuang2023/nq_hotpotqa_train"
DATASET_PATTERN = "train.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 GPU 版 Search-R1（离线证据后端）")
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
    parser.add_argument("--max-search-calls", type=int, default=2)
    parser.add_argument("--max-assistant-turns", type=int, default=3)
    parser.add_argument("--max-trajectory-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-assistant-tokens",
        type=int,
        default=0,
        help="单轮 assistant 最大 token；0 表示使用 profile 默认值",
    )
    parser.add_argument("--max-tool-response-tokens", type=int, default=512)
    parser.add_argument("--search-count", type=int, default=2)
    parser.add_argument("--search-concurrency", type=int, default=2)
    parser.add_argument("--corpus-size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=512,
        help="先载入多少条 NQ/HotpotQA 样本；0 表示使用全部（不建议）",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force-first-tool-call",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="教学模式：首轮预填工具调用骨架，只让模型生成 query",
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


def _find_cached_parquet(root: Path) -> Path | None:
    cache = root / "cache" / "modelscope" / "datasets"
    if not cache.exists():
        return None
    matches = sorted(cache.rglob("train.parquet"))
    for path in matches:
        if "nq_hotpotqa_train" in str(path):
            return path
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
        raise trainer.DataUnavailable(f"No NQ/HotpotQA parquet under {dataset_dir}")
    return candidates[0]


def _answers(row: dict[str, Any]) -> list[str]:
    values = row.get("golden_answers")
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        reward_model = row.get("reward_model") or {}
        target = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        if isinstance(target, dict):
            target = target.get("target")
        values = target
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple)):
        return []
    result: list[str] = []
    for value in values:
        answer = str(value).strip()
        if answer and answer not in result:
            result.append(answer)
    return result


def load_examples(
    root: Path,
    seed: int,
    max_samples: int,
    prefer_short: bool = False,
) -> list[search_data.SearchExample]:
    rows = trainer.load_parquet_rows(
        _dataset_parquet(root),
        limit=max_samples or 512,
        seed=seed,
    )
    examples: list[search_data.SearchExample] = []
    for index, row in enumerate(rows):
        question = str(row.get("question") or "").strip()
        answers = _answers(row)
        if not question or not answers:
            continue
        examples.append(
            search_data.SearchExample(
                id=str(row.get("id") or index),
                question=question,
                answers=answers,
                data_source=str(row.get("data_source") or "unknown"),
            )
        )
    if not examples:
        raise trainer.DataUnavailable("NQ/HotpotQA rows had no usable question/answers")
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
    search_client: local_search.OfflineEvidenceSearch,
    *,
    groups_sampled: int,
    groups_kept: int,
) -> dict[str, float]:
    tool_attempts = sum(
        "<tool_call>" in turn.text
        for trajectory in trajectories
        for turn in trajectory.turns
    )
    valid_tool_calls = sum(trajectory.search_calls for trajectory in trajectories)
    trajectory_lengths = [
        len(trajectory.turns[-1].prompt_tokens)
        + len(trajectory.turns[-1].completion_tokens)
        for trajectory in trajectories
        if trajectory.turns
    ]
    metrics = {
        "reward/mean": mean([float(item.reward) for item in trajectories]),
        "reward/correct": mean([float(item.exact_match) for item in trajectories]),
        "reward/format": mean([float(item.valid_format) for item in trajectories]),
        "rollout/turns": mean([float(len(item.turns)) for item in trajectories]),
        "rollout/search_calls": mean([float(item.search_calls) for item in trajectories]),
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
    metrics.update(search_client.stats.metrics())
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
        "references": trajectory.example.answers,
        "reward": trajectory.reward,
        "advantage": trajectory.advantage,
        "exact_match": trajectory.exact_match,
        "valid_format": trajectory.valid_format,
        "search_calls": trajectory.search_calls,
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
    root, run_dir, run_id, config = trainer.prepare_run(11, args)
    steps = args.steps or (4 if args.profile == "smoke" else config["steps"])
    group_size = args.group_size or (4 if args.profile == "smoke" else config["group_size"])
    max_assistant_tokens = args.max_assistant_tokens or (
        384 if args.profile == "smoke" else config["max_new_tokens"]
    )
    if (
        steps < 1
        or args.questions_per_batch < 1
        or group_size < 2
        or args.max_search_calls < 1
        or args.max_assistant_turns < 1
        or args.max_trajectory_tokens < 256
        or max_assistant_tokens < 1
    ):
        raise ValueError("steps/questions/group/turns/token limits are invalid")

    summary: dict[str, Any] = {
        "chapter": 11,
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        **config,
        "profile": args.profile,
        "steps": steps,
        "questions_per_batch": args.questions_per_batch,
        "group_size": group_size,
        "max_search_calls": args.max_search_calls,
        "max_assistant_turns": args.max_assistant_turns,
        "max_trajectory_tokens": args.max_trajectory_tokens,
        "max_assistant_tokens": max_assistant_tokens,
        "max_tool_response_tokens": args.max_tool_response_tokens,
        "force_first_tool_call": args.force_first_tool_call,
        "dynamic_sampling": args.dynamic_sampling,
        "max_question_attempts": args.max_question_attempts,
        "prefer_short_questions": args.prefer_short_questions,
        "search_backend": "offline-evidence-oracle",
        "search_backend_is_online": False,
        "external_api_calls": 0,
        "study_order": [
            "experiments/search-r1/readme.md",
            "experiments/search-r1/protocol.py",
            "experiments/search-r1/search.py",
            "experiments/search-r1/rollout.py",
            "experiments/search-r1/local_gpu.py",
        ],
    }
    trainer.save_run_config(run_dir, summary)

    try:
        print("Loading NQ/HotpotQA and the local Qwen3-4B model...", flush=True)
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
        corpus = examples[: max(1, args.corpus_size)]
        search_client = local_search.OfflineEvidenceSearch(
            corpus,
            max_results=args.search_count,
        )
        rollout_config = RolloutConfig(
            group_size=group_size,
            max_search_calls=args.max_search_calls,
            max_assistant_turns=args.max_assistant_turns,
            max_trajectory_tokens=args.max_trajectory_tokens,
            max_assistant_tokens=max_assistant_tokens,
            max_tool_response_tokens=args.max_tool_response_tokens,
            search_concurrency=args.search_concurrency,
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
                task_id=f"search-r1-{run_id}-{step}",
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
                    search_client,
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
            losses, updated = rollout_trainer.train_trajectories(
                model,
                optimizer,
                trajectories,
                loss_fn="importance_sampling",
                clip_low=1.0 / 1.2,
                clip_high=1.2,
                torch_module=torch,
                device=device,
            )
            metric = {
                "step": step,
                **rollout_metrics(
                    trajectories,
                    losses,
                    updated,
                    question_count,
                    search_client,
                    groups_sampled=len(attempt_records),
                    groups_kept=sum(kept for _, _, kept in attempt_records),
                ),
                "dynamic_sampling/attempts": float(attempts_used),
            }
            history.append(metric)
            lib.append_metrics(root, 11, run_id, metric)
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
                "corpus_rows": len(corpus),
            }
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        trainer.update_training_state(
            root,
            11,
            run_id,
            "completed",
            {
                "adapter": str(adapter),
                "steps": len(history),
                "summary": str(run_dir / "summary.json"),
                "mechanism_only": True,
                "search_backend": "offline-evidence-oracle",
            },
            replace_details=True,
        )
        print(f"chapter 11 completed: {run_dir}", flush=True)
        return 0
    except trainer.DataUnavailable as exc:
        (run_dir / "data_unavailable.txt").write_text(str(exc) + "\n", encoding="utf-8")
        trainer.update_training_state(
            root,
            11,
            run_id,
            "data_unavailable",
            {"dataset": DATASET_ID, "reason": str(exc), "note": "no fake success"},
            replace_details=True,
        )
        print(f"chapter 11 data_unavailable: {exc}", flush=True)
        return 0
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        trainer.update_training_state(
            root,
            11,
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
