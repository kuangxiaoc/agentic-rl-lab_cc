"""08 GSPO: sequence-level importance ratios on the same single-GPU loop.

Chapters 02 and 07 clip one ratio per token. GSPO (arXiv:2507.18071) replaces
that with a single ratio per answer, computed as the geometric mean of the
token ratios:

    log_ratio_i = mean_t(log pi_current(o_t) - log pi_sampling(o_t))
    ratio_i     = exp(log_ratio_i)

so a 200-token answer and a 20-token answer are compared on the same scale.
That is the entire algorithmic change; everything else is the chapter-02 loop.

The default clip bounds are GSPO's paper values (3e-4 / 4e-4), which are far
tighter than PPO's because a sequence-level ratio is much more sensitive.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import local_gpu_common as common  # noqa: E402
import local_gpu_trainer as trainer  # noqa: E402
import modelscope_lib as lib  # noqa: E402


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_ID = "AI-ModelScope/gsm8k"
DATASET_PATTERN = "main/train-00000-of-00001.parquet"
QUESTION_SUFFIX = "Provide a numerical answer without units, written inside \\boxed{}."

# GSPO paper defaults. Note these are offsets from 1.0, not absolute bounds.
GSPO_CLIP_LOW = 3e-4
GSPO_CLIP_HIGH = 4e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 GPU 版 GSPO / GSM8K")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--profile",
        default=os.environ.get("RUN_PROFILE", "smoke"),
        choices=("smoke", "full"),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="有效更新次数；0 表示使用 profile 默认值",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=0,
        help="每题采样数；0 表示使用 profile 默认值",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="每条 completion 的最大 token 数；0 表示使用 profile 默认值",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="最多采样尝试次数；0 表示 steps 的四倍",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="从 GSM8K 的第几条开始，避免总在同样几道简单题上退化",
    )
    parser.add_argument(
        "--effective-only",
        action="store_true",
        help="只把组内既有答对又有答错的样本计入有效更新",
    )
    parser.add_argument("--clip-low", type=float, default=GSPO_CLIP_LOW)
    parser.add_argument("--clip-high", type=float, default=GSPO_CLIP_HIGH)
    parser.add_argument(
        "--token-level-clip",
        action="store_true",
        help="diagnostic: use chapter 02's per-token clipping instead of GSPO's "
        "sequence ratio, so the two can be compared on identical samples",
    )
    return parser.parse_args()


def build_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": question + QUESTION_SUFFIX}]


def main() -> int:
    args = parse_args()
    root, run_dir, run_id, config = trainer.prepare_run(8, args)
    steps = args.steps or config["steps"]
    group_size = args.group_size or config["group_size"]
    max_new_tokens = args.max_new_tokens or config["max_new_tokens"]
    max_attempts = args.max_attempts or (steps * 4 if args.effective_only else steps)
    if steps < 1 or group_size < 2 or max_new_tokens < 1 or max_attempts < steps:
        raise ValueError(
            "steps >= 1, group-size >= 2, max-new-tokens >= 1 and "
            "max-attempts >= steps are required"
        )
    summary: dict[str, object] = {
        "chapter": 8,
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        **config,
        "profile": args.profile,
        "clip_low": args.clip_low,
        "clip_high": args.clip_high,
        "sequence_level": not args.token_level_clip,
        "steps": steps,
        "group_size": group_size,
        "max_new_tokens": max_new_tokens,
        "max_attempts": max_attempts,
        "start_index": args.start_index,
        "effective_only": args.effective_only,
        "external_api_calls": 0,
        "study_order": [
            "experiments/gspo/readme.md",
            "experiments/gspo/loss.py",
            "experiments/gspo/local_gpu.py",
        ],
    }
    trainer.save_run_config(run_dir, summary)
    try:
        print("Downloading model and GSM8K from ModelScope...", flush=True)
        model_dir = trainer.download_modelscope_model(MODEL_ID, root)
        dataset_dir = trainer.download_modelscope_dataset(DATASET_ID, root, DATASET_PATTERN)
        parquet_path = dataset_dir / DATASET_PATTERN
        if not parquet_path.exists():
            candidates = sorted(dataset_dir.rglob("*.parquet"))
            if not candidates:
                raise trainer.DataUnavailable(f"No GSM8K parquet under {dataset_dir}")
            parquet_path = candidates[0]
        rows = trainer.load_parquet_rows(parquet_path)
        if not rows:
            raise trainer.DataUnavailable("GSM8K train parquet is empty")
        import torch

        device = trainer.check_gpu()
        print(f"model_dir={model_dir} dataset_rows={len(rows)}", flush=True)

        model, tokenizer, _ = trainer.build_qlora_model(model_dir, vision=False)
        model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=4e-5,
        )
        history: list[dict[str, float]] = []
        effective_updates = 0
        attempts = 0
        skipped_groups = 0
        while effective_updates < steps and attempts < max_attempts:
            row = rows[(args.start_index + attempts) % len(rows)]
            question = str(row["question"])
            ground_truth = str(row["answer"]).split("####")[-1].strip()
            inputs = trainer.apply_chat_template(
                tokenizer, build_messages(question), vision=False
            )
            inputs = trainer.move_inputs(inputs, device, torch)
            completions, old_logprobs, texts, _ = trainer.generate_group(
                model,
                tokenizer,
                inputs,
                group_size,
                max_new_tokens,
                temperature=1.0,
                top_p=1.0,
                seed=42 + attempts,
                torch_module=torch,
                device=device,
            )
            # Same verifiable reward as chapter 02, so only the loss differs.
            rewards = [
                1.0 if _boxed_matches(text, ground_truth) else 0.0 for text in texts
            ]
            effective = common.is_effective_group(rewards, group_size)
            if args.effective_only and not effective:
                skipped_groups += 1
                metric = {
                    "step": attempts,
                    "effective_step": effective_updates,
                    "reward": sum(rewards) / len(rewards),
                    "correct_rate": sum(rewards) / len(rewards),
                    "degenerate": 1.0,
                    "train_samples": 0,
                    "skipped_degenerate_group": 1.0,
                    "avg_completion_tokens": sum(len(c) for c in completions) / len(completions),
                    "loss": 0.0,
                }
                history.append(metric)
                lib.append_metrics(root, 8, run_id, metric)
                print(json.dumps(metric, ensure_ascii=False), flush=True)
                with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": attempts,
                                "effective_step": effective_updates,
                                "question": question,
                                "ground_truth": ground_truth,
                                "answers": texts,
                                "rewards": rewards,
                                "advantages": None,
                                "completion_tokens": [len(c) for c in completions],
                                "skipped_degenerate_group": True,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                attempts += 1
                continue
            advantages = common.group_advantages(rewards)
            samples: list[trainer.Sample] = []
            prompt_tokens = inputs["input_ids"][0].tolist()
            for completion, logprobs, advantage, reward in zip(
                completions, old_logprobs, advantages, rewards, strict=True
            ):
                if not completion or advantage == 0.0:
                    continue
                samples.append(
                    trainer.Sample(
                        input_ids=prompt_tokens + completion,
                        completion_ids=completion,
                        old_logprobs=logprobs,
                        # GSPO uses one advantage per sequence; repeating it here
                        # keeps Sample's shape identical to chapters 02/07.
                        advantages=[advantage] * len(logprobs),
                        reward=reward,
                        text="",
                    )
                )

            loss_value = 0.0
            ratios: list[float] = []
            if samples:
                optimizer.zero_grad(set_to_none=True)
                train_losses = []
                for sample in samples:
                    if args.token_level_clip:
                        # Diagnostic baseline: chapter 02's per-token clipping.
                        # GSPO's bounds are offsets from 1.0, so convert them to
                        # the absolute [low, high] interval PPO expects.
                        loss = trainer.sequence_policy_loss(
                            model,
                            sample,
                            torch,
                            device,
                            "ppo",
                            clip_low=1.0 - args.clip_low,
                            clip_high=1.0 + args.clip_high,
                        )
                    else:
                        loss = trainer.sequence_policy_loss_gspo(
                            model,
                            sample,
                            torch,
                            device,
                            clip_low=args.clip_low,
                            clip_high=args.clip_high,
                        )
                    train_losses.append(loss.detach())
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                loss_value = float(torch.stack(train_losses).mean().item())

            metric = {
                "step": attempts,
                "effective_step": effective_updates,
                "reward": sum(rewards) / len(rewards),
                "correct_rate": sum(1.0 for r in rewards if r > 0) / len(rewards),
                "degenerate": 0.0 if samples else 1.0,
                "train_samples": len(samples),
                "avg_completion_tokens": sum(len(c) for c in completions) / len(completions),
                "loss": loss_value,
            }
            history.append(metric)
            lib.append_metrics(root, 8, run_id, metric)
            print(json.dumps(metric, ensure_ascii=False), flush=True)
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": attempts,
                            "effective_step": effective_updates,
                            "question": question,
                            "ground_truth": ground_truth,
                            "answers": texts,
                            "rewards": rewards,
                            "advantages": advantages,
                            "completion_tokens": [len(c) for c in completions],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            effective_updates += 1
            attempts += 1
            del samples, completions, old_logprobs, inputs
            torch.cuda.empty_cache()

        adapter = trainer.save_adapter(model, run_dir)
        (run_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary.update(
            {
                "history": history,
                "adapter": str(adapter),
                "attempts": attempts,
                "effective_updates": effective_updates,
                "skipped_groups": skipped_groups,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
            }
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        trainer.update_training_state(
            root,
            8,
            run_id,
            "completed",
            {
                "adapter": str(adapter),
                "steps": effective_updates,
                "attempts": attempts,
                "skipped_groups": skipped_groups,
                "summary": str(run_dir / "summary.json"),
            },
        )
        print(f"chapter 08 completed: {run_dir}", flush=True)
        return 0
    except KeyboardInterrupt:
        trainer.update_training_state(root, 8, run_id, "failed", {"reason": "interrupted"})
        raise
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        trainer.update_training_state(
            root,
            8,
            run_id,
            "failed",
            {"error": f"{type(exc).__name__}: {exc}", "error_log": str(run_dir / "error.txt")},
        )
        traceback.print_exc()
        print(f"chapter 08 failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


def _boxed_matches(text: str, ground_truth: str) -> bool:
    """Last \\boxed{...} compared after the same light normalisation as chapter 02."""
    import re

    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if not matches:
        return False
    answer = matches[-1].strip().replace(",", "").rstrip(".")
    return answer == ground_truth.replace(",", "").rstrip(".")


if __name__ == "__main__":
    raise SystemExit(main())
