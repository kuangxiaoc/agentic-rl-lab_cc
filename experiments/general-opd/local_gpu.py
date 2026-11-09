"""05 General OPD: local-GPU on-policy distillation on DeepMath.

The original chapter uses PyTRIO's remote ``sample`` and ``compute_logprobs``.
This version keeps the same OPD semantics on one ModelScope A10:

    student = Qwen3-1.7B + fresh QLoRA
    teacher = Qwen3-4B-Instruct-2507, frozen 4-bit
    prompt  = DeepMath question only
    signal  = -beta * (student_logprob - teacher_logprob)

Both models use the Qwen3 tokenizer family. Never replace the teacher with
Qwen3.5: the vocabulary and token ids differ, so token-level OPD is invalid.
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


STUDENT_MODEL = "Qwen/Qwen3-1.7B"
TEACHER_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_ID = "AI-ModelScope/DeepMath-103K"
DATASET_PATTERN = "data/train-00000-of-00010.parquet"
QUESTION_SUFFIX = (
    "Please solve the problem step by step and put the final answer in \\boxed{}."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 GPU 版 General OPD / DeepMath")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--profile",
        default=os.environ.get("RUN_PROFILE", "smoke"),
        choices=("smoke", "full"),
    )
    parser.add_argument("--student-model", default=STUDENT_MODEL)
    parser.add_argument("--teacher-model", default=TEACHER_MODEL)
    parser.add_argument("--max-train-prompts", type=int, default=0, help="0 表示使用当前分片全部 prompt")
    parser.add_argument("--steps", type=int, default=0, help="0 表示使用 profile 的默认步数")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=0, help="0 表示使用 profile 默认值")
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--kl-penalty-coef", type=float, default=1.0)
    parser.add_argument("--loss-fn", choices=("importance_sampling", "ppo", "cispo"), default="importance_sampling")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=0, help=">0 时按 step 保存中间 adapter")
    return parser.parse_args()


def find_parquet(root: Path, pattern: str) -> Path:
    if root.is_file():
        return root
    direct = root / pattern
    if direct.exists():
        return direct
    candidates = sorted(root.rglob("*.parquet"))
    if not candidates:
        raise trainer.DataUnavailable(f"No parquet under {root}")
    return candidates[0]


def load_prompts(root: Path, max_prompts: int, seed: int) -> list[str]:
    dataset_dir = trainer.download_modelscope_dataset(DATASET_ID, root, DATASET_PATTERN)
    parquet = find_parquet(dataset_dir, DATASET_PATTERN)
    rows = trainer.load_parquet_rows(parquet)
    questions = [
        str(row["question"]).strip()
        for row in rows
        if str(row.get("question", "")).strip()
    ]
    if not questions:
        raise trainer.DataUnavailable("DeepMath question column is empty")
    random.Random(seed).shuffle(questions)
    if max_prompts > 0:
        questions = questions[:max_prompts]
    return questions


def build_prompt_ids(tokenizer: Any, question: str) -> list[int]:
    return local_opd.build_prompt_ids(
        tokenizer,
        [
            {
                "role": "user",
                "content": f"{question.strip()}\n\n{QUESTION_SUFFIX}",
            }
        ],
        enable_thinking=False,
    )


def main() -> int:
    args = parse_args()
    root, run_dir, run_id, config = trainer.prepare_run(5, args)
    steps = args.steps or config["steps"]
    max_new_tokens = args.max_new_tokens or config["max_new_tokens"]
    summary: dict[str, Any] = {
        "chapter": 5,
        "student_model": args.student_model,
        "teacher_model": args.teacher_model,
        "dataset": DATASET_ID,
        "profile": args.profile,
        "steps": steps,
        "batch_size": args.batch_size,
        "group_size": args.group_size,
        "max_new_tokens": max_new_tokens,
        "external_api_calls": 0,
        "study_order": [
            "experiments/general-opd/readme.md",
            "experiments/general-opd/01-demo-sync.py",
            "experiments/general-opd/local_gpu.py",
        ],
    }
    trainer.save_run_config(run_dir, summary)
    try:
        print("Downloading DeepMath prompts from ModelScope...", flush=True)
        prompts = load_prompts(root, args.max_train_prompts, args.seed)
        if not prompts:
            raise trainer.DataUnavailable("DeepMath prompt pool is empty")
        print(f"prompts={len(prompts)}", flush=True)
        print("Downloading Qwen3 student and teacher models...", flush=True)
        student_dir = trainer.download_modelscope_model(args.student_model, root)
        teacher_dir = trainer.download_modelscope_model(args.teacher_model, root)

        import torch

        device = trainer.check_gpu()
        student_model, tokenizer, _ = trainer.build_qlora_model(
            student_dir,
            rank=args.lora_rank,
            vision=False,
        )
        teacher_model = trainer.build_quantized_model(teacher_dir)
        teacher_tokenizer = trainer.load_text_tokenizer(teacher_dir)
        local_opd.assert_compatible_tokenizers(tokenizer, teacher_tokenizer)
        print(
            "tokenizer compatible: "
            f"{local_opd.tokenizer_fingerprint(tokenizer)}",
            flush=True,
        )

        student_model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in student_model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
        )
        history: list[dict[str, Any]] = []
        for step in range(steps):
            prompt_batch = [
                prompts[(step * args.batch_size + index) % len(prompts)]
                for index in range(args.batch_size)
            ]
            prompt_ids = [
                build_prompt_ids(tokenizer, question) for question in prompt_batch
            ]
            metrics, records = local_opd.run_opd_update(
                student_model=student_model,
                teacher_model=teacher_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                prompt_ids_list=prompt_ids,
                group_size=args.group_size,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                top_p=1.0,
                kl_penalty_coef=args.kl_penalty_coef,
                loss_fn=args.loss_fn,
                torch_module=torch,
                device=device,
                seed=args.seed + step * 1000,
            )
            metric = {"step": step, **metrics}
            history.append(metric)
            lib.append_metrics(root, 5, run_id, metric)
            print(json.dumps(metric, ensure_ascii=False), flush=True)
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(
                        json.dumps(
                            {"step": step, **record},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            if args.save_every > 0 and (step + 1) % args.save_every == 0:
                trainer.save_adapter(student_model, run_dir / f"adapter-step-{step + 1}")

        adapter = trainer.save_adapter(student_model, run_dir)
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
            5,
            run_id,
            "completed",
            {
                "adapter": str(adapter),
                "steps": len(history),
                "student_model": args.student_model,
                "teacher_model": args.teacher_model,
                "summary": str(run_dir / "summary.json"),
            },
        )
        print(f"chapter 05 completed: {run_dir}", flush=True)
        return 0
    except trainer.DataUnavailable as exc:
        (run_dir / "data_unavailable.txt").write_text(str(exc) + "\n", encoding="utf-8")
        trainer.update_training_state(
            root,
            5,
            run_id,
            "data_unavailable",
            {"dataset": DATASET_ID, "reason": str(exc), "note": "no fake success"},
        )
        print(f"chapter 05 data_unavailable: {exc}", flush=True)
        return 0
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        trainer.update_training_state(
            root,
            5,
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
