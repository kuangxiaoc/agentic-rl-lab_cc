"""09 OPSD: local-GPU on-policy self-distillation.

The student and teacher are the same frozen base model. The student sees only
the problem; the teacher sees the problem plus the reference solution. The
student samples one completion, then both prompts score those exact tokens:

    reverse_kl = student_logprob - teacher_logprob
    advantage  = -beta * reverse_kl

Only the student's QLoRA adapter is trained. The teacher is a frozen 4-bit copy
of step 0, so this chapter does not need a second model download.
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import local_gpu_trainer as trainer  # noqa: E402
import local_opd  # noqa: E402
import modelscope_lib as lib  # noqa: E402


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_ID = "siyanzhao/Openthoughts_math_30k_opsd"
DATASET_REVISION = "1f33e9dc2e8a1c639ca74f8024ad4a9f1f5eae62"
DATASET_PATTERN = "*.parquet"

STUDENT_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
TEACHER_TRANSITION = (
    "After reading the reference solution above, make sure you truly understand "
    "the reasoning behind each step - do not copy or paraphrase it. Now, using "
    "your own words and independent reasoning, derive the same final answer to "
    "the problem above. Think step by step and do not be afraid to backtrack."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 GPU 版 OPSD / OpenThoughts Math")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--profile",
        default=os.environ.get("RUN_PROFILE", "smoke"),
        choices=("smoke", "full"),
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--steps", type=int, default=0, help="0 表示使用 profile 默认值")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0 表示使用 profile 默认值")
    parser.add_argument("--max-train-prompts", type=int, default=0, help="0 表示使用当前数据全部样本")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--kl-penalty-coef", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=0, help=">0 时按 step 保存中间 adapter")
    return parser.parse_args()


def build_student_prompt(tokenizer: Any, problem: str) -> list[int]:
    return local_opd.build_prompt_ids(
        tokenizer,
        [
            {
                "role": "user",
                "content": f"Problem: {problem.strip()}\n\n{STUDENT_INSTRUCTION}",
            }
        ],
        enable_thinking=False,
    )


def build_teacher_prompt(tokenizer: Any, problem: str, solution: str) -> list[int]:
    user_message = (
        f"Problem: {problem.strip()}\n\n"
        "Here is a reference solution to this problem:\n"
        "=== Reference Solution Begin ===\n"
        f"{solution.strip()}\n"
        "=== Reference Solution End ===\n\n"
        f"{TEACHER_TRANSITION}\n\n"
        f"{STUDENT_INSTRUCTION}"
    )
    return local_opd.build_prompt_ids(
        tokenizer,
        [{"role": "user", "content": user_message}],
        enable_thinking=False,
    )


def download_hf_dataset(root: Path) -> list[dict[str, Any]]:
    """Download the fixed OPSD revision through HF_ENDPOINT into persistent storage."""
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        print(f"OPSD dataset endpoint: {endpoint}", flush=True)
    from datasets import load_dataset

    cache_dir = root / "cache" / "huggingface" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        dataset = load_dataset(
            DATASET_ID,
            revision=DATASET_REVISION,
            split="train",
            cache_dir=str(cache_dir),
        )
    except Exception as exc:
        raise trainer.DataUnavailable(
            f"OPSD dataset unavailable from {endpoint or 'huggingface.co'}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    missing = sorted({"problem", "solution"} - set(dataset.column_names))
    if missing:
        raise trainer.DataUnavailable(
            f"OPSD dataset is missing columns {missing}: {dataset.column_names}"
        )
    rows = [
        {"problem": str(row["problem"]).strip(), "solution": str(row["solution"]).strip()}
        for row in dataset
        if str(row.get("problem", "")).strip() and str(row.get("solution", "")).strip()
    ]
    if not rows:
        raise trainer.DataUnavailable("OPSD dataset has no non-empty problem/solution rows")
    return rows


def main() -> int:
    args = parse_args()
    root, run_dir, run_id, config = trainer.prepare_run(9, args)
    steps = args.steps or config["steps"]
    max_new_tokens = args.max_new_tokens or config["max_new_tokens"]
    if steps < 1 or args.batch_size < 1 or args.group_size < 1 or max_new_tokens < 1:
        raise ValueError("steps, batch-size, group-size and max-new-tokens must be >= 1")
    summary: dict[str, Any] = {
        "chapter": 9,
        "model": args.model,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        **config,
        "profile": args.profile,
        "steps": steps,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
        "max_new_tokens": max_new_tokens,
        "teacher": "frozen step-0 base model with privileged reference solution",
        "external_api_calls": 0,
        "study_order": [
            "experiments/opsd/readme.md",
            "experiments/opsd/01-opsd-sync.py",
            "experiments/opsd/local_gpu.py",
        ],
    }
    trainer.save_run_config(run_dir, summary)
    try:
        print("Downloading OPSD problem/solution data...", flush=True)
        rows = download_hf_dataset(root)
        random.Random(args.seed).shuffle(rows)
        if args.max_train_prompts > 0:
            rows = rows[: args.max_train_prompts]
        print(f"dataset_rows={len(rows)}", flush=True)

        print("Downloading the shared OPSD student/teacher base model...", flush=True)
        model_dir = trainer.download_modelscope_model(args.model, root)
        import torch

        device = trainer.check_gpu()
        student, tokenizer, _ = trainer.build_qlora_model(
            model_dir,
            rank=args.lora_rank,
            vision=False,
        )
        teacher = trainer.build_quantized_model(model_dir)
        teacher_tokenizer = trainer.load_text_tokenizer(model_dir)
        local_opd.assert_compatible_tokenizers(tokenizer, teacher_tokenizer)
        student.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in student.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
        )

        history: list[dict[str, Any]] = []
        for step in range(steps):
            batch = [
                rows[(step * args.batch_size + index) % len(rows)]
                for index in range(args.batch_size)
            ]
            student_prompts = [
                build_student_prompt(tokenizer, str(row["problem"])) for row in batch
            ]
            teacher_prompts = [
                build_teacher_prompt(
                    tokenizer,
                    str(row["problem"]),
                    str(row["solution"]),
                )
                for row in batch
            ]
            metrics, records = local_opd.run_opd_update_with_prompts(
                student_model=student,
                teacher_model=teacher,
                tokenizer=tokenizer,
                optimizer=optimizer,
                student_prompt_ids_list=student_prompts,
                teacher_prompt_ids_list=teacher_prompts,
                group_size=args.group_size,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_p=1.0,
                kl_penalty_coef=args.kl_penalty_coef,
                loss_fn="importance_sampling",
                torch_module=torch,
                device=device,
                seed=args.seed + step * 1000,
            )
            metric = {"step": step, **metrics}
            history.append(metric)
            lib.append_metrics(root, 9, run_id, metric)
            print(json.dumps(metric, ensure_ascii=False), flush=True)
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps({"step": step, **record}, ensure_ascii=False) + "\n")
            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                trainer.save_adapter(student, run_dir / f"adapter-step-{step + 1}")

        adapter = trainer.save_adapter(student, run_dir)
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
            }
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        trainer.update_training_state(
            root,
            9,
            run_id,
            "completed",
            {"adapter": str(adapter), "steps": len(history), "summary": str(run_dir / "summary.json")},
        )
        print(f"chapter 09 completed: {run_dir}", flush=True)
        return 0
    except trainer.DataUnavailable as exc:
        (run_dir / "data_unavailable.txt").write_text(str(exc) + "\n", encoding="utf-8")
        trainer.update_training_state(
            root,
            9,
            run_id,
            "data_unavailable",
            {"dataset": DATASET_ID, "reason": str(exc), "note": "no fake success"},
        )
        print(f"chapter 09 data_unavailable: {exc}", flush=True)
        return 0
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        trainer.update_training_state(
            root,
            9,
            run_id,
            "failed",
            {"error": f"{type(exc).__name__}: {exc}", "error_log": str(run_dir / "error.txt")},
        )
        traceback.print_exc()
        return 1
    finally:
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
