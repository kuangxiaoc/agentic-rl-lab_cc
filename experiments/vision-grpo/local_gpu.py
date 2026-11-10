"""03 Vision GRPO: real single-GPU QLoRA loop placed beside the original chapter.

GeoQA is preflighted from the original HF source. Any network/schema failure is
recorded as `data_unavailable`; GSM8K text data is never silently substituted.
"""

from __future__ import annotations

import io
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


MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
GEOQA_ID = "hz2475/geoQA"


def as_rgb_image(value: object) -> object:
    """Accept a PIL image or the bytes/dict representation stored in parquet."""
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        value = value.get("bytes") or value.get("image")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    raise trainer.DataUnavailable(f"Unsupported GeoQA image value: {type(value).__name__}")


def build_messages(original: object, row: dict[str, object], image: object) -> list[dict[str, object]]:
    subject = str(row.get("subject", ""))
    choices = [str(choice) for choice in row.get("choices", [])]
    if len(choices) != 4:
        raise trainer.DataUnavailable(f"GeoQA row has {len(choices)} choices, expected 4")
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": original.format_question(subject, choices)},
            ],
        }
    ]


def parse_args() -> object:
    import argparse

    parser = argparse.ArgumentParser(description="本地 GPU 版 Vision GRPO / GeoQA")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--profile",
        default=os.environ.get("RUN_PROFILE", "smoke"),
        choices=("smoke", "full"),
    )
    parser.add_argument("--loss-fn", choices=("importance_sampling", "ppo", "cispo"), default="importance_sampling")
    parser.add_argument("--steps", type=int, default=0, help="目标有效更新次数；0 表示使用 profile 默认值")
    parser.add_argument("--group-size", type=int, default=0, help="0 表示使用 profile 默认值")
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0 表示使用 profile 默认值")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=0, help="0 表示有效更新次数的四倍")
    parser.add_argument(
        "--effective-only",
        action="store_true",
        help="跳过全对/全错组，只把有组内差异的样本计入目标更新次数",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root, run_dir, run_id, config = trainer.prepare_run(3, args)
    steps = args.steps or config["steps"]
    group_size = args.group_size or config["group_size"]
    max_new_tokens = args.max_new_tokens or config["max_new_tokens"]
    max_attempts = args.max_attempts or (steps * 4 if args.effective_only else steps)
    if steps < 1 or group_size < 2 or max_new_tokens < 1 or max_attempts < steps:
        raise ValueError(
            "steps >= 1, group-size >= 2, max-new-tokens >= 1 and "
            "max-attempts >= steps are required"
        )
    original = common.load_original_module(
        "vision_train",
        REPO_ROOT / "experiments" / "vision-grpo" / "train.py",
        required=("CHOICE_LETTERS", "format_question", "extract_choice"),
    )
    summary: dict[str, object] = {
        "chapter": 3,
        "model": MODEL_ID,
        "dataset": GEOQA_ID,
        "profile": args.profile,
        "loss_fn": args.loss_fn,
        **config,
        "steps": steps,
        "group_size": group_size,
        "max_new_tokens": max_new_tokens,
        "effective_only": args.effective_only,
        "max_attempts": max_attempts,
        "start_index": args.start_index,
        "external_api_calls": 0,
        "study_order": [
            "experiments/vision-grpo/readme.md",
            "experiments/vision-grpo/train.py",
            "experiments/vision-grpo/local_gpu.py",
        ],
    }
    trainer.save_run_config(run_dir, summary)
    try:
        rows = trainer.download_geoqa(root)
        (run_dir / "data_preflight.json").write_text(
            json.dumps({"dataset": GEOQA_ID, "train_rows": len(rows)}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"GeoQA preflight ok: {len(rows)} rows; downloading {MODEL_ID}", flush=True)
        model_dir = trainer.download_modelscope_model(MODEL_ID, root)
        import torch

        device = trainer.check_gpu()
        model, processor, _ = trainer.build_qlora_model(model_dir, vision=True)
        model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=4e-5,
        )
        history: list[dict[str, float]] = []
        tracker = common.EffectiveUpdateTracker(steps, max_attempts)
        while tracker.can_attempt:
            attempts = tracker.attempts
            row = rows[(args.start_index + attempts) % len(rows)]
            image = as_rgb_image(row.get("image"))
            gold_choice = original.CHOICE_LETTERS[int(row["label"])]
            inputs = trainer.apply_chat_template(
                processor, build_messages(original, row, image), vision=True
            )
            inputs = trainer.move_inputs(inputs, device, torch)
            completions, old_logprobs, texts, _ = trainer.generate_group(
                model,
                processor,
                inputs,
                group_size,
                max_new_tokens,
                temperature=1.0,
                top_p=1.0,
                seed=42 + attempts,
                torch_module=torch,
                device=device,
            )
            predicted = [original.extract_choice(text) for text in texts]
            rewards = [float(choice == gold_choice) for choice in predicted]
            advantages = common.group_advantages(rewards)
            effective = common.is_effective_group(rewards, group_size)
            if args.effective_only and not effective:
                metric = {
                    "step": attempts,
                    "effective_step": tracker.effective_updates,
                    "reward": sum(rewards) / len(rewards),
                    "format_rate": sum(choice is not None for choice in predicted) / len(predicted),
                    "degenerate": 1.0,
                    "train_samples": 0,
                    "skipped_degenerate_group": 1.0,
                    "avg_completion_tokens": sum(len(item) for item in completions) / len(completions),
                    "loss": 0.0,
                }
                history.append(metric)
                lib.append_metrics(root, 3, run_id, metric)
                print(json.dumps(metric, ensure_ascii=False), flush=True)
                with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": attempts,
                                "effective_step": tracker.effective_updates,
                                "id": row.get("id"),
                                "gold_choice": gold_choice,
                                "predicted": predicted,
                                "answers": [lib.redact_text(text) for text in texts],
                                "rewards": rewards,
                                "advantages": advantages,
                                "skipped_degenerate_group": True,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                tracker.record(False)
                del completions, old_logprobs, inputs
                torch.cuda.empty_cache()
                continue
            samples: list[trainer.Sample] = []
            prompt_tokens = inputs["input_ids"][0].tolist()
            vision_inputs = {
                key: value[:1]
                for key, value in inputs.items()
                if key not in {"input_ids", "attention_mask"} and hasattr(value, "shape")
            }
            for completion, logprobs, reward, advantage in zip(
                completions, old_logprobs, rewards, advantages, strict=True
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
                        extra_inputs=vision_inputs,
                    )
                )
            loss_value = 0.0
            if samples:
                optimizer.zero_grad(set_to_none=True)
                train_losses = []
                for sample in samples:
                    loss = trainer.sequence_policy_loss(
                        model, sample, torch, device, args.loss_fn
                    )
                    train_losses.append(loss.detach())
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                loss_value = float(torch.stack(train_losses).mean().item())
            metric: dict[str, float] = {
                "step": attempts,
                "effective_step": tracker.effective_updates,
                "reward": sum(rewards) / len(rewards),
                "format_rate": sum(choice is not None for choice in predicted) / len(predicted),
                "degenerate": 0.0 if samples else 1.0,
                "train_samples": len(samples),
                "avg_completion_tokens": sum(len(item) for item in completions) / len(completions),
                "loss": loss_value,
            }
            history.append(metric)
            lib.append_metrics(root, 3, run_id, metric)
            print(json.dumps(metric, ensure_ascii=False), flush=True)
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": attempts,
                            "effective_step": tracker.effective_updates,
                            "id": row.get("id"),
                            "gold_choice": gold_choice,
                            "predicted": predicted,
                            "answers": [lib.redact_text(text) for text in texts],
                            "rewards": rewards,
                            "advantages": advantages,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            tracker.record(bool(samples))
            del samples, completions, old_logprobs, inputs, vision_inputs
            torch.cuda.empty_cache()
        tracker.require_complete()
        adapter = trainer.save_adapter(model, run_dir)
        (run_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary.update(
            {
                "history": history,
                "adapter": str(adapter),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
            }
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        trainer.update_training_state(
            root,
            3,
            run_id,
            "completed",
            {
                "adapter": str(adapter),
                "steps": tracker.effective_updates,
                "attempts": tracker.attempts,
                "skipped_groups": tracker.skipped_groups,
                "summary": str(run_dir / "summary.json"),
            },
            replace_details=True,
        )
        print(f"chapter 03 completed: {run_dir}", flush=True)
        return 0
    except trainer.DataUnavailable as exc:
        (run_dir / "data_unavailable.txt").write_text(str(exc) + "\n", encoding="utf-8")
        trainer.update_training_state(
            root,
            3,
            run_id,
            "data_unavailable",
            {"dataset": GEOQA_ID, "reason": str(exc), "note": "no text-data substitute"},
        )
        print(f"chapter 03 data_unavailable: {exc}", flush=True)
        return 0
    except KeyboardInterrupt:
        trainer.update_training_state(root, 3, run_id, "failed", {"reason": "interrupted"})
        raise
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        trainer.update_training_state(
            root,
            3,
            run_id,
            "failed",
            {"error": f"{type(exc).__name__}: {exc}", "error_log": str(run_dir / "error.txt")},
        )
        # The runner captures stderr in the chapter log, so the traceback must reach it.
        traceback.print_exc()
        print(f"chapter 03 failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
