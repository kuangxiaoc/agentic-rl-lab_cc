"""Shared local-GPU OPD primitives for chapters 05 and 06.

The original OPD chapters use PyTRIO's ``sample`` and ``compute_logprobs``.
This module keeps the same contract on one local GPU:

    student samples a completion
    teacher scores the exact prompt + completion token sequence
    reverse_kl = student_logprob - teacher_logprob
    advantage  = -beta * reverse_kl
    PPO / importance-sampling updates the student LoRA

Only standard-library code is imported at module import time, so unit tests do
not need torch, transformers or a GPU.
"""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any, Iterable

import local_gpu_trainer as trainer


CHOICE_SYSTEM_MESSAGE = (
    "你是中文单项选择题作答助手。请在内部完成必要推理，"
    "但最终回答只能包含 A、B、C、D 中的一个大写字母，"
    "不要输出推理过程、解释、标点或其他文字。"
)


def tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    """A small, deterministic probe used to reject incompatible OPD teachers."""
    probe = "Qwen3 OPD tokenizer check: <|im_start|>assistant\nA B C D 1 2 3 医疗"
    return {
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "probe_ids": tokenizer.encode(probe, add_special_tokens=False)[:96],
    }


def assert_compatible_tokenizers(student_tokenizer: Any, teacher_tokenizer: Any) -> None:
    """OPD compares token-level logprobs, so the two tokenizers must agree exactly."""
    student = tokenizer_fingerprint(student_tokenizer)
    teacher = tokenizer_fingerprint(teacher_tokenizer)
    if student != teacher:
        raise ValueError(
            "Student and teacher tokenizers are incompatible. "
            f"student={student}, teacher={teacher}"
        )


def build_prompt_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = False,
) -> list[int]:
    """Render chat messages with the model's own template and return token ids."""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if trainer.chat_template_mentions(tokenizer, "enable_thinking"):
        kwargs["enable_thinking"] = enable_thinking
    prompt = tokenizer.apply_chat_template(messages, **kwargs)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        raise ValueError("Prompt tokenization returned no tokens")
    return ids


def reverse_kl_advantages(
    student_logprobs: Iterable[float],
    teacher_logprobs: Iterable[float],
    kl_penalty_coef: float = 1.0,
) -> list[float]:
    """Turn ``student_logprob - teacher_logprob`` into OPD advantages."""
    student = [float(value) for value in student_logprobs]
    teacher = [float(value) for value in teacher_logprobs]
    if len(student) != len(teacher):
        raise ValueError(
            f"OPD logprob lengths differ: student={len(student)} teacher={len(teacher)}"
        )
    if not student:
        raise ValueError("OPD requires at least one completion token")
    return [
        -float(kl_penalty_coef) * (student_value - teacher_value)
        for student_value, teacher_value in zip(student, teacher, strict=True)
    ]


def completion_teacher_logprobs(
    teacher_model: Any,
    prompt_ids: list[int],
    completion_ids: list[int],
    *,
    torch_module: Any,
    device: Any,
) -> list[float]:
    """Score the exact student trajectory with the frozen local teacher."""
    if not completion_ids:
        return []
    full_ids = prompt_ids + completion_ids
    input_ids = torch_module.tensor([full_ids], device=device, dtype=torch_module.long)
    attention_mask = torch_module.ones_like(input_ids)
    teacher_model.eval()
    with torch_module.no_grad():
        logits = teacher_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[0].float()
        # logits[i] predicts full_ids[i + 1]. The first completion token is at
        # full index len(prompt_ids), so its predictor is prompt_length - 1.
        start = len(prompt_ids) - 1
        positions = torch_module.arange(
            start,
            start + len(completion_ids),
            device=device,
        )
        targets = torch_module.tensor(
            completion_ids,
            device=device,
            dtype=torch_module.long,
        )
        selected = logits.index_select(0, positions)
        values = torch_module.log_softmax(selected, dim=-1).gather(
            1, targets.unsqueeze(1)
        )
    return [float(value) for value in values.squeeze(1).cpu().tolist()]


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance**0.5


def run_opd_update(
    *,
    student_model: Any,
    teacher_model: Any,
    tokenizer: Any,
    optimizer: Any,
    prompt_ids_list: list[list[int]],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    kl_penalty_coef: float,
    loss_fn: str,
    torch_module: Any,
    device: Any,
    seed: int,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    """Run one local on-policy OPD update over a list of prompts."""
    samples: list[trainer.Sample] = []
    records: list[dict[str, Any]] = []
    reverse_kl_values: list[float] = []
    completion_lengths: list[int] = []

    for prompt_index, prompt_ids in enumerate(prompt_ids_list):
        inputs = {
            "input_ids": torch_module.tensor(
                [prompt_ids], device=device, dtype=torch_module.long
            )
        }
        completions, old_logprobs, texts, _ = trainer.generate_group(
            student_model,
            tokenizer,
            inputs,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed + prompt_index,
            torch_module=torch_module,
            device=device,
        )
        for completion, student_logprobs, text in zip(
            completions, old_logprobs, texts, strict=True
        ):
            if not completion:
                continue
            teacher_logprobs = completion_teacher_logprobs(
                teacher_model,
                prompt_ids,
                completion,
                torch_module=torch_module,
                device=device,
            )
            if len(teacher_logprobs) != len(completion):
                raise ValueError(
                    "Teacher returned the wrong number of completion logprobs: "
                    f"completion={len(completion)} teacher={len(teacher_logprobs)}"
                )
            advantages = reverse_kl_advantages(
                student_logprobs,
                teacher_logprobs,
                kl_penalty_coef,
            )
            samples.append(
                trainer.Sample(
                    input_ids=prompt_ids + completion,
                    completion_ids=completion,
                    old_logprobs=student_logprobs,
                    advantages=advantages,
                    reward=0.0,
                    text=text,
                )
            )
            reverse_kl_values.extend(
                student_value - teacher_value
                for student_value, teacher_value in zip(
                    student_logprobs, teacher_logprobs, strict=True
                )
            )
            completion_lengths.append(len(completion))
            records.append(
                {
                    "prompt_index": prompt_index,
                    "completion": text,
                    "completion_tokens": len(completion),
                    "student_logprob_sum": sum(student_logprobs),
                    "teacher_logprob_sum": sum(teacher_logprobs),
                    "reverse_kl_sum": sum(
                        student_value - teacher_value
                        for student_value, teacher_value in zip(
                            student_logprobs, teacher_logprobs, strict=True
                        )
                    ),
                }
            )

    loss_value = 0.0
    if samples:
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for sample in samples:
            loss = trainer.sequence_policy_loss(
                student_model,
                sample,
                torch_module,
                device,
                loss_fn,
            )
            losses.append(loss.detach())
            loss.backward()
        torch_module.nn.utils.clip_grad_norm_(
            (parameter for parameter in student_model.parameters() if parameter.requires_grad),
            1.0,
        )
        optimizer.step()
        loss_value = float(torch_module.stack(losses).mean().item())

    reverse_kl_mean, reverse_kl_std = _mean_std(reverse_kl_values)
    metrics: dict[str, float | int] = {
        "prompts": len(prompt_ids_list),
        "train_samples": len(samples),
        "completion_tokens_mean": (
            sum(completion_lengths) / len(completion_lengths)
            if completion_lengths
            else 0.0
        ),
        "reverse_kl_mean": reverse_kl_mean,
        "reverse_kl_std": reverse_kl_std,
        "loss": loss_value,
    }
    del samples
    gc.collect()
    torch_module.cuda.empty_cache()
    return metrics, records


def build_choice_prompt(question: str, options: dict[str, str]) -> str:
    """Render a four-option multiple-choice prompt without exposing the answer."""
    return "\n".join(
        [
            "以下是中国考试中的单项选择题。请仔细思考，并只输出最终答案选项字母。",
            "",
            f"题目：{question.strip()}",
            *[
                f"{key}. {options[key]}"
                for key in ("A", "B", "C", "D")
                if key in options
            ],
            "",
            "答案：",
        ]
    )


def parse_choice(text: str) -> str | None:
    """Parse a generated choice letter, including optional thinking text."""
    raw = text.strip()
    if "</think>" in raw:
        raw = raw.rsplit("</think>", 1)[-1]
    normalized = raw.strip().upper()
    if not normalized:
        return None
    exact = re.fullmatch(r"[^A-Z]*([ABCD])[^A-Z]*", normalized)
    if exact:
        return exact.group(1)
    patterns = [
        r"(?:FINAL\s*(?:ANSWER|OUTPUT)|ANSWER|OUTPUT|最终答案|正确答案|答案|选项|选择|应选|故选)\s*(?:IS|是|为|:|：|->)?\s*[\(\[【]?\s*([ABCD])",
        r"[\(\[【]\s*([ABCD])\s*[\)\]】]",
    ]
    candidates: list[str] = []
    for pattern in patterns:
        candidates.extend(re.findall(pattern, normalized))
    if candidates:
        return candidates[-1]
    for line in reversed(normalized.splitlines()):
        match = re.fullmatch(r"\s*([ABCD])\s*[\.。:：、]?\s*", line)
        if match:
            return match.group(1)
    fallback = re.findall(r"(?<![A-Z])([ABCD])(?![A-Z])", normalized[-1000:])
    return fallback[-1] if fallback else None


def generate_choice_prediction(
    model: Any,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    torch_module: Any,
    device: Any,
    max_new_tokens: int = 32,
) -> tuple[str, str | None]:
    """Greedy local generation for MedQA / C-Eval style rows."""
    prompt_text = build_choice_prompt(
        str(row.get("question", "")),
        dict(row.get("options") or {}),
    )
    prompt_ids = build_prompt_ids(
        tokenizer,
        [
            {"role": "system", "content": CHOICE_SYSTEM_MESSAGE},
            {"role": "user", "content": prompt_text},
        ],
        enable_thinking=False,
    )
    input_ids = torch_module.tensor([prompt_ids], device=device, dtype=torch_module.long)
    attention_mask = torch_module.ones_like(input_ids)
    model.eval()
    with torch_module.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=trainer.pad_token_id(tokenizer),
        )
    text = tokenizer.decode(
        output[0, len(prompt_ids) :].tolist(),
        skip_special_tokens=True,
    )
    return text, parse_choice(text)


def evaluate_choice_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    torch_module: Any,
    device: Any,
    max_new_tokens: int = 32,
    limit: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate local greedy accuracy on rows with ``question/options/answer_idx``."""
    selected = rows[:limit] if limit > 0 else rows
    results: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        text, prediction = generate_choice_prediction(
            model,
            tokenizer,
            row,
            torch_module=torch_module,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        answer = str(row.get("answer_idx", "")).strip().upper()
        results.append(
            {
                "index": index,
                "row_id": row.get("row_id"),
                "subject": row.get("subject"),
                "question": row.get("question"),
                "answer_idx": answer,
                "prediction": prediction,
                "correct": prediction == answer,
                "output": text,
            }
        )
    correct = sum(1 for row in results if row["correct"])
    metrics = {
        "samples": len(results),
        "correct": correct,
        "accuracy": correct / len(results) if results else 0.0,
    }
    return results, metrics


def save_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
