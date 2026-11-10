"""02 GRPO: real single-GPU QLoRA loop placed beside the original PyTRIO chapter.

This script reuses the original boxed-answer reward and GSM8K answer parser.
It never imports PyTRIO or calls an external judge/teacher/search API.

阅读主线（配合 experiments/grpo/readme.md）：
    下载模型/数据 -> 4-bit QLoRA -> 逐步采样一组回答 -> 规则打分
    -> 组内归一化得 advantage -> policy loss -> backward -> 保存记录

相比第 01 章，唯一的变化是 advantage 不再手写，而是由真实模型产生：
    A_i = reward_i - mean(reward_1..reward_G)
loss 里用的 ratio = exp(current_logprob - old_logprob) 仍是第 01 章那套。

smoke: steps=2, group_size=2, max_new_tokens=128（验证链路）
full:  steps=10, group_size=4, max_new_tokens=512（更容易出现组内方差）
"""

from __future__ import annotations

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


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"  # 4-bit 加载后约 4-5GB 显存
DATASET_ID = "AI-ModelScope/gsm8k"        # GSM8K 的 ModelScope 镜像
DATASET_PATTERN = "main/train-00000-of-00001.parquet"  # 只取这一个 parquet


def build_messages(original: object, question: str) -> list[dict[str, str]]:
    """把 few-shot 示例和当前题目拼成 chat messages。

    FEWSHOT_PREFIX 是原版 03-grpo.py 里的示例问答，QUESTION_SUFFIX 要求
    答案写进 \\boxed{}。固定模板是为了让同组的回答可以公平比较。
    """
    return [
        *original.FEWSHOT_PREFIX,
        {
            "role": "user",
            "content": question + original.QUESTION_SUFFIX,
        },
    ]


def parse_args() -> object:
    import argparse

    parser = argparse.ArgumentParser(description="本地 GPU 版 GRPO / GSM8K")
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
    """主入口：准备 -> 下载 -> 建模 -> GRPO 循环 -> 落盘。"""
    # prepare_run 负责建 runs/02-grpo/<run_id>/、写 state/environment/
    # study_notes/api_budget，并把章节状态置为 running。
    args = parse_args()
    root, run_dir, run_id, config = trainer.prepare_run(2, args)
    steps = args.steps or config["steps"]
    group_size = args.group_size or config["group_size"]
    max_new_tokens = args.max_new_tokens or config["max_new_tokens"]
    max_attempts = args.max_attempts or (steps * 4 if args.effective_only else steps)
    if steps < 1 or group_size < 2 or max_new_tokens < 1 or max_attempts < steps:
        raise ValueError(
            "steps >= 1, group-size >= 2, max-new-tokens >= 1 and "
            "max-attempts >= steps are required"
        )

    # 只从原版文件里抽出纯常量和纯函数（AST 加载），从而不 import pytrio、
    # swanlab 等云端依赖。required 让缺失的常量在下载模型前就报错。
    original = common.load_original_module(
        "grpo_03",
        REPO_ROOT / "experiments" / "grpo" / "03-grpo.py",
        required=("FEWSHOT_PREFIX", "QUESTION_SUFFIX", "extract_gsm8k_answer", "grade_answer"),
    )
    # config 里是 steps / group_size / max_new_tokens 等，先落盘便于复现实验。
    summary: dict[str, object] = {
        "chapter": 2,
        "model": MODEL_ID,
        "dataset": DATASET_ID,
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
            "experiments/grpo/readme.md",
            "experiments/grpo/03-grpo.py",
            "experiments/grpo/local_gpu.py",
        ],
    }
    trainer.save_run_config(run_dir, summary)
    try:
        # 模型和数据集都缓存到 /mnt/workspace，重启实例不用重新下载。
        print("Downloading model and GSM8K from ModelScope...", flush=True)
        model_dir = trainer.download_modelscope_model(MODEL_ID, root)
        dataset_dir = trainer.download_modelscope_dataset(DATASET_ID, root, DATASET_PATTERN)
        # 某些版本的 ModelScope 会把 parquet 放进子目录，做个兜底。
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

        # 明确要求 CUDA + 24GB 级显存，失败就报错而不是偷偷退化到 CPU。
        device = trainer.check_gpu()
        print(f"model_dir={model_dir} dataset_rows={len(rows)}", flush=True)

        # 4-bit NF4 量化 + gradient checkpointing + LoRA(all-linear)。
        # 只有 requires_grad=True 的 LoRA 参数会进优化器。
        model, tokenizer, _ = trainer.build_qlora_model(model_dir, vision=False)
        model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=4e-5,
        )
        history: list[dict[str, float]] = []
        tracker = common.EffectiveUpdateTracker(steps, max_attempts)
        while tracker.can_attempt:
            attempts = tracker.attempts
            # A. 取题：GSM8K 标准答案在 #### 之后。
            row = rows[(args.start_index + attempts) % len(rows)]
            question = str(row["question"])
            ground_truth = original.extract_gsm8k_answer(str(row["answer"]))
            # chat template 会插入 <|im_start|> / <|im_end|> 等 special tokens。
            inputs = trainer.apply_chat_template(
                tokenizer, build_messages(original, question), vision=False
            )
            inputs = trainer.move_inputs(inputs, device, torch)
            # B. 采 G 个回答。generate_group 会临时切 eval()，因为训练模式下
            # gradient checkpointing 与 KV cache 互斥，会生成乱码；采完切回 train()。
            # output.scores 经 log_softmax 得到采样那一刻的 old_logprobs。
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
            # C. 规则打分：最后一个 \boxed{...} 与标准答案一致就得 1，否则 0。
            # 这就是 RLVR，不需要额外的 reward model。
            rewards = [original.grade_answer(text, ground_truth) for text in texts]
            # D. 组内均值当 baseline。论文还会除以 std，这里为对齐原版省略了。
            advantages = common.group_advantages(rewards)
            effective = common.is_effective_group(rewards, group_size)
            if args.effective_only and not effective:
                metric = {
                    "step": attempts,
                    "effective_step": tracker.effective_updates,
                    "reward": sum(rewards) / len(rewards),
                    "degenerate": 1.0,
                    "train_samples": 0,
                    "skipped_degenerate_group": 1.0,
                    "avg_completion_tokens": sum(len(item) for item in completions) / len(completions),
                    "loss": 0.0,
                }
                history.append(metric)
                lib.append_metrics(root, 2, run_id, metric)
                print(json.dumps(metric, ensure_ascii=False), flush=True)
                with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": attempts,
                                "effective_step": tracker.effective_updates,
                                "question": question,
                                "ground_truth": ground_truth,
                                "answers": texts,
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
            # E. 组装训练样本。Sample 里 advantages 被复制到每个 completion token，
            # 因为 reward 是整条回答级别的，没有逐 token 打分。
            samples: list[trainer.Sample] = []
            prompt_tokens = inputs["input_ids"][0].tolist()
            for completion, logprobs, reward, advantage in zip(
                completions, old_logprobs, rewards, advantages, strict=True
            ):
                # 全对或全错时 advantage 全为 0（退化组），没有相对信号，
                # 这一步就完全跳过 backward。真实运行里 10 步有 9 步是这样。
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
            # F. 逐样本 backward 累积梯度，最后统一 clip + step。
            # sequence_policy_loss 只取 completion 区间的 logprob：
            # logits[i] 预测 input_ids[i+1]，所以起点是 prompt_length - 1。
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
            # G. 指标。loss 接近 0 不代表没学：组内 advantage 均值为 0，
            # 但每条梯度方向不同。判断有没有更新要看 train_samples。
            metric: dict[str, float] = {
                "step": attempts,
                "effective_step": tracker.effective_updates,
                "reward": sum(rewards) / len(rewards),
                "degenerate": 0.0 if samples else 1.0,
                "train_samples": len(samples),
                "avg_completion_tokens": sum(len(item) for item in completions) / len(completions),
                "loss": loss_value,
            }
            history.append(metric)
            lib.append_metrics(root, 2, run_id, metric)
            print(json.dumps(metric, ensure_ascii=False), flush=True)
            # H. 一行一个 step，保存原始回答文本供事后分析（例如 step 4
            # 两错两对，能看到模型把 "6 pages/week" 误读成 312 的系统性错误）。
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": attempts,
                            "effective_step": tracker.effective_updates,
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
            tracker.record(bool(samples))
            del samples, completions, old_logprobs, inputs
            torch.cuda.empty_cache()
        tracker.require_complete()
        # adapter 只存 LoRA 权重，留在 /mnt/workspace，不进 Git。
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
            {
                "adapter": str(adapter),
                "steps": tracker.effective_updates,
                "attempts": tracker.attempts,
                "skipped_groups": tracker.skipped_groups,
                "summary": str(run_dir / "summary.json"),
            },
            replace_details=True,
        )
        print(f"chapter 02 completed: {run_dir}", flush=True)
        return 0
    except KeyboardInterrupt:
        trainer.update_training_state(root, 2, run_id, "failed", {"reason": "interrupted"})
        raise
    except Exception as exc:
        # 完整 traceback 写入 run 目录，同时打到被 runner 捕获的 stderr，
        # 这样 chapter-02.log 里也能直接看到失败原因。
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        trainer.update_training_state(
            root,
            2,
            run_id,
            "failed",
            {"error": f"{type(exc).__name__}: {exc}", "error_log": str(run_dir / "error.txt")},
        )
        # The runner captures stderr in the chapter log, so the traceback must reach it.
        traceback.print_exc()
        print(f"chapter 02 failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
