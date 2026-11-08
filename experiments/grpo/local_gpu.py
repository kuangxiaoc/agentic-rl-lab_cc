"""02 GRPO: real single-GPU QLoRA loop placed beside the original PyTRIO chapter.

This script reuses the original boxed-answer reward and GSM8K answer parser.
It never imports PyTRIO or calls an external judge/teacher/search API.
"""

from __future__ import annotations

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


def build_messages(original: object, question: str) -> list[dict[str, str]]:
    return [
        *original.FEWSHOT_PREFIX,
        {
            "role": "user",
            "content": question + original.QUESTION_SUFFIX,
        },
    ]


def main() -> int:
    args = trainer.chapter_cli("本地 GPU 版 GRPO / GSM8K")
    root, run_dir, run_id, config = trainer.prepare_run(2, args)
    original = common.load_original_module(
        "grpo_03", REPO_ROOT / "experiments" / "grpo" / "03-grpo.py"
    )
    summary: dict[str, object] = {
        "chapter": 2,
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "profile": args.profile,
        "loss_fn": args.loss_fn,
        **config,
        "external_api_calls": 0,
        "study_order": [
            "experiments/grpo/readme.md",
            "experiments/grpo/03-grpo.py",
            "experiments/grpo/local_gpu.py",
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
        for step in range(config["steps"]):
            row = rows[step % len(rows)]
            question = str(row["question"])
            ground_truth = original.extract_gsm8k_answer(str(row["answer"]))
            inputs = trainer.apply_chat_template(
                tokenizer, build_messages(original, question), vision=False
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
            rewards = [original.grade_answer(text, ground_truth) for text in texts]
            advantages = common.group_advantages(rewards)
            samples: list[trainer.Sample] = []
            prompt_tokens = inputs["input_ids"][0].tolist()
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
                "step": step,
                "reward": sum(rewards) / len(rewards),
                "degenerate": 0.0 if samples else 1.0,
                "train_samples": len(samples),
                "avg_completion_tokens": sum(len(item) for item in completions) / len(completions),
                "loss": loss_value,
            }
            history.append(metric)
            lib.append_metrics(root, 2, run_id, metric)
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
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
            }
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        trainer.update_training_state(
            root,
            2,
            run_id,
            "completed",
            {"adapter": str(adapter), "steps": len(history), "summary": str(run_dir / "summary.json")},
        )
        print(f"chapter 02 completed: {run_dir}", flush=True)
        return 0
    except KeyboardInterrupt:
        trainer.update_training_state(root, 2, run_id, "failed", {"reason": "interrupted"})
        raise
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        trainer.update_training_state(
            root,
            2,
            run_id,
            "failed",
            {"error": f"{type(exc).__name__}: {exc}", "error_log": str(run_dir / "error.txt")},
        )
        print(f"chapter 02 failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
