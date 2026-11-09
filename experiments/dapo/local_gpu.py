"""07 DAPO: the same single-GPU QLoRA loop as GRPO, with DAPO's four changes.

Every difference from chapter 02 is a flag, so you can turn one on at a time and
watch the metrics change:

  --clip-low/--clip-high   Clip-Higher: asymmetric clip (0.8 / 1.28 by default)
  --dynamic-sampling       drop all-correct and all-wrong groups before backward
  --token-level-loss       pool completion tokens instead of averaging per answer
  --overlong-cache N       soft length penalty in the last N tokens of the budget

Reward parsing and the length penalty are imported from the original
experiments/dapo/reward.py, so this file only contains the training loop.
"""

from __future__ import annotations

import argparse
import json
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

# DAPO's official preset. clip_high=1.28 is the "Clip-Higher" change.
DAPO_CLIP_LOW = 0.8
DAPO_CLIP_HIGH = 1.28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 GPU 版 DAPO / GSM8K")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--profile", default="smoke", choices=("smoke", "full"))
    parser.add_argument("--clip-low", type=float, default=DAPO_CLIP_LOW)
    parser.add_argument("--clip-high", type=float, default=DAPO_CLIP_HIGH)
    parser.add_argument(
        "--dynamic-sampling",
        action="store_true",
        help="skip groups whose rewards are all 0 or all 1 (no learning signal)",
    )
    parser.add_argument(
        "--token-level-loss",
        action="store_true",
        help="pool completion tokens instead of averaging per answer",
    )
    parser.add_argument(
        "--overlong-cache",
        type=int,
        default=0,
        help="soft length penalty over the last N tokens; 0 disables it",
    )
    return parser.parse_args()


def build_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": question + QUESTION_SUFFIX}]


def load_dapo_reward() -> object:
    """Reuse the original reward module without installing math_verify."""
    try:
        from math_verify import parse, verify
    except ImportError:
        # Fallback: exact string comparison after trimming whitespace and $.
        def parse(value: str) -> str:
            return value.strip().strip("$").strip()

        def verify(left: str, right: str) -> bool:
            return left == right

        print(
            "math_verify not installed; answers compare as exact strings",
            flush=True,
        )
    return common.load_original_module(
        "dapo_reward",
        REPO_ROOT / "experiments" / "dapo" / "reward.py",
        required=(
            "extract_last_boxed",
            "answers_equivalent",
            "soft_overlong_penalty",
            "score_answer",
            "RewardResult",
        ),
        namespace={"parse": parse, "verify": verify},
    )


def main() -> int:
    args = parse_args()
    root, run_dir, run_id, config = trainer.prepare_run(7, args)
    reward_module = load_dapo_reward()
    summary: dict[str, object] = {
        "chapter": 7,
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "profile": args.profile,
        "clip_low": args.clip_low,
        "clip_high": args.clip_high,
        "dynamic_sampling": args.dynamic_sampling,
        "token_level_loss": args.token_level_loss,
        "overlong_cache": args.overlong_cache,
        **config,
        "external_api_calls": 0,
        "study_order": [
            "experiments/dapo/readme.md",
            "experiments/dapo/reward.py",
            "experiments/dapo/rollout.py",
            "experiments/dapo/local_gpu.py",
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
        enable_overlong = args.overlong_cache > 0
        if enable_overlong and args.overlong_cache > config["max_new_tokens"]:
            raise ValueError(
                f"--overlong-cache {args.overlong_cache} exceeds "
                f"max_new_tokens {config['max_new_tokens']}"
            )

        history: list[dict[str, float]] = []
        skipped_groups = 0
        for step in range(config["steps"]):
            row = rows[step % len(rows)]
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
                config["group_size"],
                config["max_new_tokens"],
                temperature=1.0,
                top_p=1.0,
                seed=42 + step,
                torch_module=torch,
                device=device,
            )
            # DAPO's reward is +-1 with an optional soft length penalty, so a wrong
            # answer can still carry a small negative shaped reward.
            results = [
                reward_module.score_answer(
                    text,
                    ground_truth,
                    completion_tokens=len(tokens),
                    max_tokens=config["max_new_tokens"],
                    overlong_cache=max(args.overlong_cache, 1),
                    enable_overlong=enable_overlong,
                )
                for text, tokens in zip(texts, completions, strict=True)
            ]
            rewards = [result.shaped_reward for result in results]

            # Change 2: Dynamic Sampling. A unanimous group has zero variance and
            # therefore zero advantage, so training on it only wastes compute.
            effective = common.is_effective_group(
                [1.0 if result.correct else 0.0 for result in results],
                config["group_size"],
            )
            if args.dynamic_sampling and not effective:
                skipped_groups += 1
                metric = {
                    "step": step,
                    "reward": sum(rewards) / len(rewards),
                    "degenerate": 1.0,
                    "train_samples": 0,
                    "skipped_by_dynamic_sampling": 1.0,
                    "loss": 0.0,
                }
                history.append(metric)
                lib.append_metrics(root, 7, run_id, metric)
                print(json.dumps(metric, ensure_ascii=False), flush=True)
                continue

            # DAPO standardises by the group's sample standard deviation; GRPO only
            # mean-centres. With --dynamic-sampling the group is guaranteed mixed.
            advantages = common.normalized_group_advantages(rewards)
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
                        advantages=[advantage] * len(logprobs),
                        reward=reward,
                        text="",
                    )
                )

            loss_value = 0.0
            if samples:
                optimizer.zero_grad(set_to_none=True)
                train_losses = []
                # Change 3: token-level reduction. sequence_policy_loss returns
                # -mean(objective) per answer; reweight by n_i / N so every
                # completion token, not every answer, contributes equally.
                total_tokens = sum(len(sample.completion_ids) for sample in samples)
                for sample in samples:
                    loss = trainer.sequence_policy_loss(
                        model,
                        sample,
                        torch,
                        device,
                        "ppo",
                        clip_low=args.clip_low,
                        clip_high=args.clip_high,
                    )
                    if args.token_level_loss:
                        loss = loss * (len(sample.completion_ids) / total_tokens)
                    train_losses.append(loss.detach())
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                loss_value = float(torch.stack(train_losses).mean().item())

            metric = {
                "step": step,
                "reward": sum(rewards) / len(rewards),
                "correct_rate": sum(1.0 for r in results if r.correct) / len(results),
                "degenerate": 0.0 if samples else 1.0,
                "train_samples": len(samples),
                "skipped_by_dynamic_sampling": 0.0,
                "avg_completion_tokens": sum(len(c) for c in completions) / len(completions),
                "length_penalty_mean": sum(r.length_penalty for r in results) / len(results),
                "loss": loss_value,
            }
            history.append(metric)
            lib.append_metrics(root, 7, run_id, metric)
            print(json.dumps(metric, ensure_ascii=False), flush=True)
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "question": question,
                            "ground_truth": ground_truth,
                            "answers": texts,
                            "rewards": rewards,
                            "advantages": advantages,
                            "correct": [r.correct for r in results],
                            "length_penalties": [r.length_penalty for r in results],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
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
            7,
            run_id,
            "completed",
            {
                "adapter": str(adapter),
                "steps": len(history),
                "skipped_groups": skipped_groups,
                "summary": str(run_dir / "summary.json"),
            },
        )
        print(f"chapter 07 completed: {run_dir}", flush=True)
        return 0
    except KeyboardInterrupt:
        trainer.update_training_state(root, 7, run_id, "failed", {"reason": "interrupted"})
        raise
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        trainer.update_training_state(
            root,
            7,
            run_id,
            "failed",
            {"error": f"{type(exc).__name__}: {exc}", "error_log": str(run_dir / "error.txt")},
        )
        traceback.print_exc()
        print(f"chapter 07 failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
