"""06 Medical OPD: local-GPU medical SFT + OPD + optional dual-teacher IDT.

The original chapter uses PyTRIO and a 4B student/teacher. The local chapter
uses the smallest compatible Qwen3 pair that fits comfortably on one A10:

    student        Qwen/Qwen3-1.7B
    medical teacher Qwen/Qwen3-4B-Instruct-2507 + Medical SFT LoRA
    anchor teacher  Qwen/Qwen3-1.7B (the student's frozen base)

The tokenizer compatibility check is mandatory. Qwen3.5 is not interchangeable
with Qwen3 for token-level OPD.

Stages:
    sar (default): medical SFT -> Medical OPD -> Base-anchor OPD -> eval
    idt:           medical SFT -> alternating Medical/C-Eval OPD -> eval
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import os
import random
import shutil
import sys
import traceback
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import local_gpu_trainer as trainer  # noqa: E402
import local_opd  # noqa: E402
import modelscope_lib as lib  # noqa: E402


STUDENT_MODEL = "Qwen/Qwen3-1.7B"
MEDICAL_TEACHER_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MEDICAL_SFT_REPO = "AI-ModelScope/medical-o1-reasoning-SFT"
MEDICAL_SFT_FILE = "medical_o1_sft_Chinese.json"
MEDQA_REPO = "AI-ModelScope/med_qa"
MEDQA_FILE = "data_clean.zip"
CEVAL_URL = "https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/c-eval/ceval-exam.zip"

MEDICAL_SYSTEM_MESSAGE = (
    "你是一个中文医疗问答助手。请根据题目给出严谨的医学推理和最终答案。"
)
CEVAL_SUBJECTS = [
    "computer_network",
    "college_programming",
    "advanced_mathematics",
    "discrete_mathematics",
    "college_physics",
    "logic",
    "chinese_language_and_literature",
    "college_economics",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 GPU 版 Medical OPD")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--profile",
        default=os.environ.get("RUN_PROFILE", "smoke"),
        choices=("smoke", "full"),
    )
    parser.add_argument("--mode", choices=("sar", "idt"), default="sar")
    parser.add_argument("--student-model", default=STUDENT_MODEL)
    parser.add_argument("--medical-teacher-model", default=MEDICAL_TEACHER_MODEL)
    parser.add_argument("--teacher-adapter", type=Path, default=None, help="复用已有 Medical SFT adapter；不传则先训练")
    parser.add_argument("--medical-sft-limit", type=int, default=2000)
    parser.add_argument("--sft-steps", type=int, default=0)
    parser.add_argument("--medical-opd-steps", type=int, default=0)
    parser.add_argument("--anchor-opd-steps", type=int, default=0)
    parser.add_argument("--idt-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate-sft", type=float, default=1e-4)
    parser.add_argument("--learning-rate-medical", type=float, default=4e-5)
    parser.add_argument("--learning-rate-anchor", type=float, default=5e-6)
    parser.add_argument("--learning-rate-idt", type=float, default=1e-5)
    parser.add_argument("--kl-penalty-coef", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--medqa-eval-limit", type=int, default=0)
    parser.add_argument("--ceval-eval-limit", type=int, default=0)
    parser.add_argument("--eval-max-new-tokens", type=int, default=32)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def cache_dir(root: Path) -> Path:
    path = root / "cache" / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_url(url: str, path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"Downloading {url}\n  -> {path}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "agentic-rl-lab-local-gpu"})
    with urllib.request.urlopen(request, timeout=300) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(path)
    return path


def modelscope_dataset_file(repo: str, file_path: str) -> str:
    namespace, name = repo.split("/", 1)
    query = urllib.parse.urlencode(
        {
            "Revision": "master",
            "FilePath": file_path,
        }
    )
    return f"https://www.modelscope.cn/api/v1/datasets/{namespace}/{name}/repo?{query}"


def first_text(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def option_dict(row: dict[str, Any]) -> dict[str, str]:
    options = row.get("options")
    if isinstance(options, dict):
        return {str(key).upper(): str(value).strip() for key, value in options.items()}
    return {
        key: str(row[key]).strip()
        for key in ("A", "B", "C", "D")
        if row.get(key) is not None and str(row[key]).strip()
    }


def normalize_medqa_row(row: dict[str, Any]) -> dict[str, Any] | None:
    question = first_text(row, ("question", "Question"))
    options = option_dict(row)
    answer = first_text(row, ("answer", "Answer"))
    answer_idx = str(row.get("answer_idx") or row.get("answer_index") or "").strip().upper()
    if answer_idx not in options:
        answer_idx = next(
            (key for key, value in options.items() if value == answer),
            "",
        )
    if not question or len(options) < 4 or answer_idx not in options:
        return None
    return {
        "row_id": str(row.get("id", "")),
        "question": question,
        "options": {key: options[key] for key in ("A", "B", "C", "D")},
        "answer_idx": answer_idx,
    }


def normalize_medical_sft(row: dict[str, Any]) -> dict[str, str] | None:
    question = first_text(row, ("Question", "question"))
    cot = first_text(row, ("Complex_CoT", "complex_cot"))
    response = first_text(row, ("Response", "response"))
    completion = build_medical_completion(cot, response)
    if not question or not completion:
        return None
    return {"question": question, "completion": completion}


def build_medical_completion(cot: str, response: str) -> str:
    """Match the original Medical SFT target: think block, then answer."""
    cot = cot.strip()
    response = response.strip()
    if cot and response:
        return f"<think>\n{cot}\n</think>\n\n{response}"
    if cot:
        return f"<think>\n{cot}\n</think>"
    return response


def normalize_ceval_row(
    row: dict[str, Any],
    subject: str,
    split: str,
    row_id: str,
) -> dict[str, Any] | None:
    question = first_text(row, ("question", "Question"))
    options = option_dict(row)
    answer = first_text(row, ("answer", "Answer")).upper()
    if not question or len(options) < 4 or answer not in options:
        return None
    return {
        "row_id": f"{subject}:{split}:{row_id}",
        "subject": subject,
        "question": question,
        "options": {key: options[key] for key in ("A", "B", "C", "D")},
        "answer_idx": answer,
    }


def read_zip_csv(archive: zipfile.ZipFile, name: str) -> list[dict[str, Any]]:
    with archive.open(name) as handle:
        text = io.TextIOWrapper(handle, encoding="utf-8-sig")
        return [dict(row) for row in csv.DictReader(text)]


def prepare_datasets(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    cache = cache_dir(root)
    medical_path = download_url(
        modelscope_dataset_file(MEDICAL_SFT_REPO, MEDICAL_SFT_FILE),
        cache / "medical_o1_sft_Chinese.json",
    )
    medqa_zip_path = download_url(
        modelscope_dataset_file(MEDQA_REPO, MEDQA_FILE),
        cache / "medqa_data_clean.zip",
    )
    ceval_zip_path = download_url(CEVAL_URL, cache / "ceval-exam.zip")

    medical_rows = json.loads(medical_path.read_text(encoding="utf-8"))
    medical = [
        normalized
        for row in medical_rows
        if (normalized := normalize_medical_sft(dict(row))) is not None
    ]
    random.Random(args.seed).shuffle(medical)
    if args.medical_sft_limit > 0:
        medical = medical[: args.medical_sft_limit]

    medqa_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(medqa_zip_path) as archive:
        target_suffix = "data_clean/questions/Mainland/4_options/test.jsonl"
        matches = [name for name in archive.namelist() if name.endswith(target_suffix)]
        if not matches:
            raise FileNotFoundError(f"Cannot find {target_suffix} in {medqa_zip_path}")
        with archive.open(matches[0]) as handle:
            for line in handle:
                row = normalize_medqa_row(json.loads(line.decode("utf-8")))
                if row is not None:
                    medqa_rows.append(row)
    random.Random(args.seed).shuffle(medqa_rows)

    ceval_train: list[dict[str, Any]] = []
    ceval_eval: list[dict[str, Any]] = []
    with zipfile.ZipFile(ceval_zip_path) as archive:
        for subject in CEVAL_SUBJECTS:
            val_name = f"val/{subject}_val.csv"
            dev_name = f"dev/{subject}_dev.csv"
            if val_name in archive.namelist():
                for index, row in enumerate(read_zip_csv(archive, val_name)):
                    normalized = normalize_ceval_row(row, subject, "val", str(index))
                    if normalized is not None:
                        ceval_train.append(normalized)
            if dev_name in archive.namelist():
                for index, row in enumerate(read_zip_csv(archive, dev_name)):
                    normalized = normalize_ceval_row(row, subject, "dev", str(index))
                    if normalized is not None:
                        ceval_eval.append(normalized)
    random.Random(args.seed + 1).shuffle(ceval_train)
    random.Random(args.seed + 2).shuffle(ceval_eval)

    out_dir = root / "data" / "06-medical-opd"
    out_dir.mkdir(parents=True, exist_ok=True)
    local_opd.save_jsonl(out_dir / "medical_sft.jsonl", medical)
    local_opd.save_jsonl(out_dir / "medical_opd_prompts.jsonl", medical)
    local_opd.save_jsonl(out_dir / "medqa_eval.jsonl", medqa_rows)
    local_opd.save_jsonl(out_dir / "ceval_opd_prompts.jsonl", ceval_train)
    local_opd.save_jsonl(out_dir / "ceval_eval.jsonl", ceval_eval)
    manifest = {
        "medical_sft_rows": len(medical),
        "medical_opd_prompts": len(medical),
        "medqa_eval_rows": len(medqa_rows),
        "ceval_opd_prompts": len(ceval_train),
        "ceval_eval_rows": len(ceval_eval),
        "ceval_subjects": CEVAL_SUBJECTS,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "medical": medical,
        "medqa": medqa_rows,
        "ceval_train": ceval_train,
        "ceval_eval": ceval_eval,
        "manifest": manifest,
    }


def build_medical_prompt_ids(tokenizer: Any, question: str) -> list[int]:
    return local_opd.build_prompt_ids(
        tokenizer,
        [
            {"role": "system", "content": MEDICAL_SYSTEM_MESSAGE},
            {"role": "user", "content": question},
        ],
        enable_thinking=True,
    )


def build_ceval_prompt_ids(tokenizer: Any, row: dict[str, Any]) -> list[int]:
    return local_opd.build_prompt_ids(
        tokenizer,
        [
            {"role": "system", "content": local_opd.CHOICE_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": local_opd.build_choice_prompt(
                    str(row["question"]),
                    dict(row["options"]),
                ),
            },
        ],
        enable_thinking=False,
    )


def build_sft_example(
    tokenizer: Any,
    row: dict[str, str],
    max_length: int,
) -> tuple[list[int], list[int]] | None:
    prompt_ids = build_medical_prompt_ids(tokenizer, row["question"])
    completion_ids = tokenizer.encode(row["completion"], add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        completion_ids.append(int(tokenizer.eos_token_id))
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    if not any(label != -100 for label in labels):
        return None
    return input_ids, labels


def build_sft_batch(
    examples: list[tuple[list[int], list[int]]],
    tokenizer: Any,
    torch_module: Any,
    device: Any,
) -> tuple[Any, Any, Any]:
    pad_id = trainer.pad_token_id(tokenizer) or 0
    max_length = max(len(input_ids) for input_ids, _ in examples)
    input_ids = torch_module.full(
        (len(examples), max_length), pad_id, device=device, dtype=torch_module.long
    )
    attention_mask = torch_module.zeros_like(input_ids)
    labels = torch_module.full(
        (len(examples), max_length), -100, device=device, dtype=torch_module.long
    )
    for row, (ids, row_labels) in enumerate(examples):
        length = len(ids)
        input_ids[row, :length] = torch_module.tensor(
            ids, device=device, dtype=torch_module.long
        )
        attention_mask[row, :length] = 1
        labels[row, :length] = torch_module.tensor(
            row_labels, device=device, dtype=torch_module.long
        )
    return input_ids, attention_mask, labels


def masked_ce_loss(logits: Any, labels: Any, torch_module: Any) -> tuple[Any, float]:
    vocab_size = logits.shape[-1]
    loss = torch_module.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )
    mask = labels != -100
    accuracy = (
        float((logits.argmax(dim=-1)[mask] == labels[mask]).float().mean().item())
        if mask.any()
        else 0.0
    )
    return loss, accuracy


def run_sft(
    root: Path,
    run_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    steps: int,
) -> Path:
    import torch

    device = trainer.check_gpu()
    teacher_dir = trainer.download_modelscope_model(args.medical_teacher_model, root)
    model, tokenizer, _ = trainer.build_qlora_model(
        teacher_dir,
        rank=args.lora_rank,
        vision=False,
    )
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate_sft,
    )
    history: list[dict[str, Any]] = []
    for step in range(steps):
        batch_rows = [
            rows[(step * args.batch_size + index) % len(rows)]
            for index in range(args.batch_size)
        ]
        examples = [
            example
            for row in batch_rows
            if (example := build_sft_example(tokenizer, row, args.max_length)) is not None
        ]
        if not examples:
            continue
        input_ids, attention_mask, labels = build_sft_batch(
            examples, tokenizer, torch, device
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss, accuracy = masked_ce_loss(logits, labels, torch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            1.0,
        )
        optimizer.step()
        metric = {
            "stage": "medical_sft",
            "step": step,
            "loss": float(loss.detach().item()),
            "token_accuracy": accuracy,
            "samples": len(examples),
        }
        history.append(metric)
        lib.append_metrics(root, 6, run_id, metric)
        print(json.dumps(metric, ensure_ascii=False), flush=True)

    adapter = trainer.save_adapter(model, run_dir / "medical_teacher_adapter")
    sft_dir = run_dir / "sft"
    sft_dir.mkdir(parents=True, exist_ok=True)
    (sft_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return adapter


def run_medical_opd(
    root: Path,
    run_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    steps: int,
    max_new_tokens: int,
    teacher_adapter: Path,
) -> Path:
    import torch

    device = trainer.check_gpu()
    student_dir = trainer.download_modelscope_model(args.student_model, root)
    teacher_dir = trainer.download_modelscope_model(args.medical_teacher_model, root)
    student_model, tokenizer, _ = trainer.build_qlora_model(
        student_dir,
        rank=args.lora_rank,
        vision=False,
    )
    teacher_model = trainer.attach_adapter(
        trainer.build_quantized_model(teacher_dir),
        teacher_adapter,
    )
    teacher_tokenizer = trainer.load_text_tokenizer(teacher_dir)
    local_opd.assert_compatible_tokenizers(tokenizer, teacher_tokenizer)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in student_model.parameters() if parameter.requires_grad),
        lr=args.learning_rate_medical,
    )
    history: list[dict[str, Any]] = []
    for step in range(steps):
        batch_rows = [
            rows[(step * args.batch_size + index) % len(rows)]
            for index in range(args.batch_size)
        ]
        prompt_ids = [
            build_medical_prompt_ids(tokenizer, row["question"])
            for row in batch_rows
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
            loss_fn="ppo",
            torch_module=torch,
            device=device,
            seed=args.seed + step * 1000,
        )
        metric = {"stage": "medical_opd", "step": step, **metrics}
        history.append(metric)
        lib.append_metrics(root, 6, run_id, metric)
        print(json.dumps(metric, ensure_ascii=False), flush=True)
        with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        {"stage": "medical_opd", "step": step, **record},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    adapter = trainer.save_adapter(student_model, run_dir / "student_medical_adapter")
    (run_dir / "medical_opd_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del student_model, teacher_model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return adapter


def run_anchor_opd(
    root: Path,
    run_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    steps: int,
    max_new_tokens: int,
    student_adapter: Path,
) -> Path:
    import torch

    device = trainer.check_gpu()
    student_dir = trainer.download_modelscope_model(args.student_model, root)
    student_model, tokenizer, _ = trainer.build_qlora_from_adapter(
        student_dir,
        student_adapter,
    )
    anchor_model = trainer.build_quantized_model(student_dir)
    anchor_tokenizer = trainer.load_text_tokenizer(student_dir)
    local_opd.assert_compatible_tokenizers(tokenizer, anchor_tokenizer)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in student_model.parameters() if parameter.requires_grad),
        lr=args.learning_rate_anchor,
    )
    history: list[dict[str, Any]] = []
    for step in range(steps):
        batch_rows = [
            rows[(step * args.batch_size + index) % len(rows)]
            for index in range(args.batch_size)
        ]
        prompt_ids = [
            build_ceval_prompt_ids(tokenizer, row)
            for row in batch_rows
        ]
        metrics, records = local_opd.run_opd_update(
            student_model=student_model,
            teacher_model=anchor_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            prompt_ids_list=prompt_ids,
            group_size=args.group_size,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_p=1.0,
            kl_penalty_coef=args.kl_penalty_coef,
            loss_fn="ppo",
            torch_module=torch,
            device=device,
            seed=args.seed + 10_000 + step * 1000,
        )
        metric = {"stage": "anchor_opd", "step": step, **metrics}
        history.append(metric)
        lib.append_metrics(root, 6, run_id, metric)
        print(json.dumps(metric, ensure_ascii=False), flush=True)
        with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        {"stage": "anchor_opd", "step": step, **record},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    adapter = trainer.save_adapter(student_model, run_dir / "student_anchor_adapter")
    (run_dir / "anchor_opd_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del student_model, anchor_model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return adapter


def run_idt(
    root: Path,
    run_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    medical_rows: list[dict[str, str]],
    ceval_rows: list[dict[str, Any]],
    steps: int,
    max_new_tokens: int,
    teacher_adapter: Path,
) -> Path:
    import torch

    device = trainer.check_gpu()
    student_dir = trainer.download_modelscope_model(args.student_model, root)
    teacher_dir = trainer.download_modelscope_model(args.medical_teacher_model, root)
    student_model, tokenizer, _ = trainer.build_qlora_model(
        student_dir,
        rank=args.lora_rank,
        vision=False,
    )
    medical_teacher = trainer.attach_adapter(
        trainer.build_quantized_model(teacher_dir),
        teacher_adapter,
    )
    anchor_teacher = trainer.build_quantized_model(student_dir)
    local_opd.assert_compatible_tokenizers(
        tokenizer,
        trainer.load_text_tokenizer(teacher_dir),
    )
    local_opd.assert_compatible_tokenizers(
        tokenizer,
        trainer.load_text_tokenizer(student_dir),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in student_model.parameters() if parameter.requires_grad),
        lr=args.learning_rate_idt,
    )
    history: list[dict[str, Any]] = []
    for step in range(steps):
        task = "medical" if step % 2 == 0 else "ceval"
        if task == "medical":
            row = medical_rows[(step // 2) % len(medical_rows)]
            prompt_ids = [build_medical_prompt_ids(tokenizer, row["question"])]
            teacher_model = medical_teacher
        else:
            row = ceval_rows[(step // 2) % len(ceval_rows)]
            prompt_ids = [build_ceval_prompt_ids(tokenizer, row)]
            teacher_model = anchor_teacher
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
            loss_fn="ppo",
            torch_module=torch,
            device=device,
            seed=args.seed + 20_000 + step * 1000,
        )
        metric = {"stage": "idt", "task": task, "step": step, **metrics}
        history.append(metric)
        lib.append_metrics(root, 6, run_id, metric)
        print(json.dumps(metric, ensure_ascii=False), flush=True)
        with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        {"stage": "idt", "task": task, "step": step, **record},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    adapter = trainer.save_adapter(student_model, run_dir / "student_idt_adapter")
    (run_dir / "idt_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del student_model, medical_teacher, anchor_teacher, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return adapter


def run_eval(
    root: Path,
    run_dir: Path,
    args: argparse.Namespace,
    medqa_rows: list[dict[str, Any]],
    ceval_rows: list[dict[str, Any]],
    adapter: Path,
) -> dict[str, Any]:
    import torch

    device = trainer.check_gpu()
    student_dir = trainer.download_modelscope_model(args.student_model, root)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    medqa_limit = args.medqa_eval_limit or (20 if args.profile == "smoke" else 100)
    ceval_limit = args.ceval_eval_limit or (20 if args.profile == "smoke" else 80)

    base_model = trainer.build_quantized_model(student_dir)
    base_tokenizer = trainer.load_text_tokenizer(student_dir)
    base_model.config.use_cache = True
    medqa_base_rows, medqa_base = local_opd.evaluate_choice_rows(
        base_model,
        base_tokenizer,
        medqa_rows,
        torch_module=torch,
        device=device,
        max_new_tokens=args.eval_max_new_tokens,
        limit=medqa_limit,
    )
    ceval_base_rows, ceval_base = local_opd.evaluate_choice_rows(
        base_model,
        base_tokenizer,
        ceval_rows,
        torch_module=torch,
        device=device,
        max_new_tokens=args.eval_max_new_tokens,
        limit=ceval_limit,
    )
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    final_model = trainer.attach_adapter(
        trainer.build_quantized_model(student_dir),
        adapter,
    )
    final_model.config.use_cache = True
    medqa_final_rows, medqa_final = local_opd.evaluate_choice_rows(
        final_model,
        base_tokenizer,
        medqa_rows,
        torch_module=torch,
        device=device,
        max_new_tokens=args.eval_max_new_tokens,
        limit=medqa_limit,
    )
    ceval_final_rows, ceval_final = local_opd.evaluate_choice_rows(
        final_model,
        base_tokenizer,
        ceval_rows,
        torch_module=torch,
        device=device,
        max_new_tokens=args.eval_max_new_tokens,
        limit=ceval_limit,
    )
    local_opd.save_jsonl(eval_dir / "medqa_base.jsonl", medqa_base_rows)
    local_opd.save_jsonl(eval_dir / "medqa_final.jsonl", medqa_final_rows)
    local_opd.save_jsonl(eval_dir / "ceval_base.jsonl", ceval_base_rows)
    local_opd.save_jsonl(eval_dir / "ceval_final.jsonl", ceval_final_rows)
    metrics = {
        "medqa_base": medqa_base,
        "medqa_final": medqa_final,
        "ceval_base": ceval_base,
        "ceval_final": ceval_final,
    }
    (eval_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    del final_model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def main() -> int:
    args = parse_args()
    root, run_dir, run_id, config = trainer.prepare_run(6, args)
    steps = config["steps"]
    max_new_tokens = args.max_new_tokens or config["max_new_tokens"]
    sft_steps = args.sft_steps or max(2, steps * 5)
    medical_steps = args.medical_opd_steps or steps
    anchor_steps = args.anchor_opd_steps or steps
    idt_steps = args.idt_steps or steps * 2
    summary: dict[str, Any] = {
        "chapter": 6,
        "mode": args.mode,
        "student_model": args.student_model,
        "medical_teacher_model": args.medical_teacher_model,
        "profile": args.profile,
        "sft_steps": sft_steps,
        "medical_opd_steps": medical_steps,
        "anchor_opd_steps": anchor_steps,
        "idt_steps": idt_steps,
        "external_api_calls": 0,
        "study_order": [
            "experiments/medical-opd/readme.md",
            "experiments/medical-opd/02-medical-sft.py",
            "experiments/medical-opd/03-medical-opd-async.py",
            "experiments/medical-opd/04-ceval-opd-async.py",
            "experiments/medical-opd/local_gpu.py",
        ],
    }
    trainer.save_run_config(run_dir, summary)
    try:
        print("Preparing medical SFT, MedQA and C-Eval data...", flush=True)
        data = prepare_datasets(root, args)
        print(json.dumps(data["manifest"], ensure_ascii=False), flush=True)
        for name, rows in (
            ("medical", data["medical"]),
            ("ceval_opd", data["ceval_train"]),
            ("medqa_eval", data["medqa"]),
            ("ceval_eval", data["ceval_eval"]),
        ):
            if not rows:
                raise trainer.DataUnavailable(f"Chapter 06 dataset is empty: {name}")
        teacher_adapter = args.teacher_adapter
        if teacher_adapter is None:
            print(f"Training Medical SFT teacher for {sft_steps} steps...", flush=True)
            teacher_adapter = run_sft(
                root,
                run_dir,
                run_id,
                args,
                data["medical"],
                sft_steps,
            )
        else:
            teacher_adapter = Path(teacher_adapter).resolve()
            if not teacher_adapter.exists():
                raise FileNotFoundError(f"Medical SFT adapter not found: {teacher_adapter}")

        if args.mode == "sar":
            print(f"Running Medical OPD for {medical_steps} steps...", flush=True)
            medical_adapter = run_medical_opd(
                root,
                run_dir,
                run_id,
                args,
                data["medical"],
                medical_steps,
                max_new_tokens,
                teacher_adapter,
            )
            print(f"Running Base-anchor OPD for {anchor_steps} steps...", flush=True)
            final_adapter = run_anchor_opd(
                root,
                run_dir,
                run_id,
                args,
                data["ceval_train"],
                anchor_steps,
                max_new_tokens,
                medical_adapter,
            )
        else:
            print(f"Running IDT OPD for {idt_steps} global steps...", flush=True)
            final_adapter = run_idt(
                root,
                run_dir,
                run_id,
                args,
                data["medical"],
                data["ceval_train"],
                idt_steps,
                max_new_tokens,
                teacher_adapter,
            )

        eval_metrics: dict[str, Any] | None = None
        if not args.skip_eval:
            print("Running local MedQA / C-Eval evaluation...", flush=True)
            eval_metrics = run_eval(
                root,
                run_dir,
                args,
                data["medqa"],
                data["ceval_eval"],
                final_adapter,
            )
            print(json.dumps(eval_metrics, ensure_ascii=False, indent=2), flush=True)

        summary.update(
            {
                "teacher_adapter": str(teacher_adapter),
                "final_adapter": str(final_adapter),
                "eval": eval_metrics,
            }
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        details = {
            "mode": args.mode,
            "teacher_adapter": str(teacher_adapter),
            "final_adapter": str(final_adapter),
            "sft_steps": sft_steps,
        }
        if eval_metrics:
            details.update(
                {
                    "medqa_base_accuracy": eval_metrics["medqa_base"]["accuracy"],
                    "medqa_final_accuracy": eval_metrics["medqa_final"]["accuracy"],
                    "ceval_base_accuracy": eval_metrics["ceval_base"]["accuracy"],
                    "ceval_final_accuracy": eval_metrics["ceval_final"]["accuracy"],
                }
            )
        trainer.update_training_state(root, 6, run_id, "completed", details)
        print(f"chapter 06 completed: {run_dir}", flush=True)
        return 0
    except trainer.DataUnavailable as exc:
        (run_dir / "data_unavailable.txt").write_text(str(exc) + "\n", encoding="utf-8")
        trainer.update_training_state(
            root,
            6,
            run_id,
            "data_unavailable",
            {"reason": str(exc), "note": "no fake success"},
        )
        print(f"chapter 06 data_unavailable: {exc}", flush=True)
        return 0
    except Exception as exc:
        (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        trainer.update_training_state(
            root,
            6,
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
