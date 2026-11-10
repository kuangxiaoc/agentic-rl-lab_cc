"""04 PyTrio-Jev: local-GPU masked decision SFT.

This chapter is deliberately not an RL loop. A decision model makes one forward
pass over ``state + question schema + JSON skeleton`` and is supervised only at
the position immediately before each ``decision`` marker. The answer symbol is
then read back with a candidate-restricted softmax.

The original chapter uses PyTRIO and Qwen3.5. The local chapter uses a 4-bit
Qwen3-4B-Instruct-2507 QLoRA model on the same ModelScope A10 as chapters
01-03. Typed-decisions is available on ModelScope; ToolACE is optional because
the public mirror may be unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_gpu_trainer as trainer  # noqa: E402
import modelscope_lib as lib  # noqa: E402
import local_gpu_data as jev_data  # noqa: E402


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_ID = "LocalLLaMA/typed-decisions"
TRAIN_PATTERN = "all/train-00000-of-00001.parquet"
TEST_PATTERN = "all/test-00000-of-00001.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 GPU 版 PyTrio-Jev")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--profile",
        default=os.environ.get("RUN_PROFILE", "smoke"),
        choices=("smoke", "full"),
    )
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 表示使用全部训练集")
    parser.add_argument("--steps", type=int, default=0, help="0 表示使用 profile 的默认步数")
    parser.add_argument("--eval-limit", type=int, default=50, help="评测样本上限；0 表示全量")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
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


def load_records(root: Path, pattern: str) -> list[dict[str, Any]]:
    dataset_dir = trainer.download_modelscope_dataset(DATASET_ID, root, pattern)
    parquet = find_parquet(dataset_dir, pattern)
    rows = trainer.load_parquet_rows(parquet)
    return [jev_data.convert_typed_decisions_row(row) for row in rows]


def encode_record(
    tokenizer: Any,
    record: dict[str, Any],
    *,
    include_targets: bool = True,
) -> dict[str, Any]:
    compiled = jev_data.compile_record(record, include_targets=include_targets)
    messages = compiled["messages"]
    ids = tokenizer.encode(
        jev_data.build_chat_text(messages),
        add_special_tokens=False,
    )
    prefix_ids = tokenizer.encode(
        jev_data.build_chat_text(messages[:-1], add_generation_prompt=True),
        add_special_tokens=False,
    )
    if ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("prefix/skeleton boundary tokenization is unstable")
    skeleton_ids = ids[len(prefix_ids) :]
    marker_ids = tokenizer.encode(jev_data.MARKER, add_special_tokens=False)
    if len(marker_ids) != 1:
        raise ValueError("the decision marker must be exactly one token")
    marker_id = marker_ids[0]
    hits = [index for index, token_id in enumerate(skeleton_ids) if token_id == marker_id]
    markers: dict[str, int] = {}
    cursor = 0
    for field in compiled["fields"]:
        cursor += tokenizer.encode(field, add_special_tokens=False).count(marker_id)
        markers[field] = len(prefix_ids) + hits[cursor]
        cursor += 1
    if cursor != len(hits):
        raise ValueError("skeleton marker count does not match the number of fields")
    symbol_ids = {
        symbol: tokenizer.encode(symbol, add_special_tokens=False)[0]
        for field in compiled["options"].values()
        for symbol, _, _ in field
    }
    return {
        **compiled,
        "ids": ids,
        "markers": markers,
        "symbol_ids": symbol_ids,
        "skeleton_ids": skeleton_ids,
    }


def build_batch(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    torch_module: Any,
    device: Any,
) -> tuple[Any, Any, Any]:
    pad_id = trainer.pad_token_id(tokenizer) or 0
    max_length = max(len(example["ids"]) for example in examples)
    input_ids = torch_module.full(
        (len(examples), max_length), pad_id, device=device, dtype=torch_module.long
    )
    attention_mask = torch_module.zeros_like(input_ids)
    labels = torch_module.full(
        (len(examples), max_length), -100, device=device, dtype=torch_module.long
    )
    for row, example in enumerate(examples):
        length = len(example["ids"])
        input_ids[row, :length] = torch_module.tensor(
            example["ids"], device=device, dtype=torch_module.long
        )
        attention_mask[row, :length] = 1
        for field, symbol in example["targets"].items():
            position = example["markers"][field] - 1
            if position < 0:
                raise ValueError(f"field {field!r} has no predictable marker position")
            labels[row, position] = example["symbol_ids"][symbol]
    return input_ids, attention_mask, labels


def masked_ce_loss(logits: Any, labels: Any, torch_module: Any) -> tuple[Any, float]:
    vocab_size = logits.shape[-1]
    loss = torch_module.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
    mask = labels != -100
    if mask.any():
        predictions = logits.argmax(dim=-1)[mask]
        accuracy = float((predictions == labels[mask]).float().mean().item())
    else:
        accuracy = 0.0
    return loss, accuracy


def evaluate_records(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    torch_module: Any,
    device: Any,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    selected = records[:limit] if limit > 0 else records
    model.eval()
    results: list[dict[str, Any]] = []
    correct = total = 0
    with torch_module.no_grad():
        for record in selected:
            encoded = encode_record(tokenizer, record, include_targets=False)
            input_ids = torch_module.tensor(
                [encoded["ids"]], device=device, dtype=torch_module.long
            )
            attention_mask = torch_module.ones_like(input_ids)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[0]
            fields: dict[str, Any] = {}
            for field, options in encoded["options"].items():
                marker = encoded["markers"][field]
                candidate_ids = [encoded["symbol_ids"][symbol] for symbol, _, _ in options]
                candidate_logits = logits[marker - 1].index_select(
                    0,
                    torch_module.tensor(candidate_ids, device=device),
                )
                probabilities = torch_module.softmax(candidate_logits.float(), dim=-1)
                best = int(probabilities.argmax().item())
                predicted_symbol = options[best][0]
                gold_label = jev_data.normalize_label(
                    record["questions"][field],
                    record["targets"][field]["label"],
                )
                gold_symbol = next(
                    symbol for symbol, value, _ in options if value == gold_label
                )
                hit = predicted_symbol == gold_symbol
                correct += int(hit)
                total += 1
                fields[field] = {
                    "prediction": options[best][1],
                    "gold": gold_label,
                    "correct": hit,
                    "confidence": float(probabilities[best].item()),
                }
            results.append({"id": record["id"], "fields": fields})
    model.train()
    return results, {
        "records": len(selected),
        "decisions": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }


def main() -> int:
    args = parse_args()
    root, run_dir, run_id, config = trainer.prepare_run(4, args)
    steps = args.steps or config["steps"]
    summary: dict[str, Any] = {
        "chapter": 4,
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "profile": args.profile,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "steps": steps,
        "external_api_calls": 0,
        "study_order": [
            "experiments/pytrio-jev/readme.md",
            "experiments/pytrio-jev/train.py",
            "experiments/pytrio-jev/local_gpu.py",
        ],
    }
    trainer.save_run_config(run_dir, summary)
    try:
        print("Downloading Qwen3-4B and typed-decisions from ModelScope...", flush=True)
        model_dir = trainer.download_modelscope_model(MODEL_ID, root)
        train_records = load_records(root, TRAIN_PATTERN)
        test_records = load_records(root, TEST_PATTERN)
        if not train_records:
            raise trainer.DataUnavailable("typed-decisions train split is empty")
        if not test_records:
            raise trainer.DataUnavailable("typed-decisions test split is empty")
        random.Random(args.seed).shuffle(train_records)
        if args.max_train_samples > 0:
            train_records = train_records[: args.max_train_samples]
        print(f"train_records={len(train_records)} test_records={len(test_records)}", flush=True)

        import torch

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

        history: list[dict[str, float]] = []
        for step in range(steps):
            batch_records = [
                train_records[(step * args.batch_size + index) % len(train_records)]
                for index in range(args.batch_size)
            ]
            examples = [
                encode_record(tokenizer, record, include_targets=True)
                for record in batch_records
            ]
            if any(len(example["ids"]) > args.max_length for example in examples):
                examples = [
                    example
                    for example in examples
                    if len(example["ids"]) <= args.max_length
                ]
                if not examples:
                    continue
            input_ids, attention_mask, labels = build_batch(
                examples, tokenizer, torch, device
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            loss, accuracy = masked_ce_loss(logits, labels, torch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                1.0,
            )
            optimizer.step()
            metric = {
                "step": step,
                "loss": float(loss.detach().item()),
                "marker_accuracy": accuracy,
                "records": len(examples),
            }
            history.append(metric)
            lib.append_metrics(root, 4, run_id, metric)
            print(json.dumps(metric, ensure_ascii=False), flush=True)

        adapter = trainer.save_adapter(model, run_dir)
        results, metrics = evaluate_records(
            model,
            tokenizer,
            test_records,
            torch,
            device,
            args.eval_limit,
        )
        save_jsonl(run_dir / "eval.jsonl", results)
        (run_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "eval_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary.update(
            {
                "history": history,
                "eval": metrics,
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
            4,
            run_id,
            "completed",
            {
                "adapter": str(adapter),
                "steps": len(history),
                "eval_accuracy": metrics["accuracy"],
                "summary": str(run_dir / "summary.json"),
            },
        )
        print(f"chapter 04 completed: {run_dir}", flush=True)
        return 0
    except trainer.DataUnavailable as exc:
        (run_dir / "data_unavailable.txt").write_text(str(exc) + "\n", encoding="utf-8")
        trainer.update_training_state(
            root,
            4,
            run_id,
            "data_unavailable",
            {"dataset": DATASET_ID, "reason": str(exc), "note": "no fake success"},
        )
        print(f"chapter 04 data_unavailable: {exc}", flush=True)
        return 0
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        trainer.update_training_state(
            root,
            4,
            run_id,
            "failed",
            {"error": f"{type(exc).__name__}: {exc}", "error_log": str(run_dir / "error.txt")},
        )
        traceback.print_exc()
        return 1


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
