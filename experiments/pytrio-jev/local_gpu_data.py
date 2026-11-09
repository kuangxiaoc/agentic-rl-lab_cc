"""Pure data helpers for the local-GPU PyTrio-Jev chapter.

The original chapter compiles typed decisions into a JSON skeleton where every
field value is the single token ``decision``. The local version keeps the same
idea but renders the skeleton with a minimal Qwen chat format, so a local Qwen3
checkpoint does not depend on the original Qwen3.5-specific Jinja file.
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "You are a careful decision assistant. Use the state and decision schema in "
    "the user message to make the requested decisions. For every field, choose "
    "exactly one answer symbol (e.g. A, B, C, ...) from its listed options and "
    "return one valid JSON object mapping each field name to its chosen symbol. "
    "Use the field names and symbols exactly as given. Do not include "
    "explanations, Markdown, or extra text."
)
ANSWER_SYMBOLS = string.ascii_uppercase + string.ascii_lowercase + string.digits
MARKER = "decision"
NOUL_YES = "The answer is yes (affirmative, or align with the claim)."
NOUL_NO = "The answer is no (negative, or disagree with the claim)."


def question_options(question: dict[str, Any]) -> list[tuple[str, str]]:
    """Expand a typed question into ordered ``(value, description)`` options."""
    kind = question["type"]
    criteria = question.get("criteria")
    if kind == "choice":
        return [(str(key), str(value)) for key, value in criteria.items()]
    if kind == "score":
        if isinstance(criteria, list):
            return [(str(index), str(value)) for index, value in enumerate(criteria)]
        return [(str(key), str(value)) for key, value in criteria.items()]
    if kind == "noul":
        descriptions = criteria or {}
        yes = next(
            (
                str(value)
                for key, value in descriptions.items()
                if str(key).lower() in {"yes", "true", "1"}
            ),
            NOUL_YES,
        )
        no = next(
            (
                str(value)
                for key, value in descriptions.items()
                if str(key).lower() in {"no", "false", "0"}
            ),
            NOUL_NO,
        )
        return [("no", no), ("yes", yes)]
    raise ValueError(f"unsupported question type: {kind!r}")


def normalize_label(question: dict[str, Any], label: object) -> str:
    if question["type"] == "noul":
        return "yes" if str(label).lower() in {"yes", "true", "1"} else "no"
    return str(label)


def compile_record(
    record: dict[str, Any],
    include_targets: bool = True,
) -> dict[str, Any]:
    """Compile one canonical record into messages, options and target symbols."""
    questions = record["questions"]
    fields: list[str] = []
    options: dict[str, list[tuple[str, str, str]]] = {}
    targets: dict[str, str] = {}
    schema_lines: list[str] = []
    for raw_field, question in questions.items():
        field = str(raw_field)
        expanded = question_options(question)
        symbols = ANSWER_SYMBOLS[: len(expanded)]
        fields.append(field)
        options[field] = [
            (symbol, value, description)
            for symbol, (value, description) in zip(symbols, expanded, strict=True)
        ]
        schema_lines.append(f"{field}: {question['instructions']}")
        for symbol, value, description in options[field]:
            schema_lines.append(f"    {symbol} = {value}: {description}")
        if include_targets:
            label = normalize_label(question, record["targets"][field]["label"])
            values = [value for _, value, _ in options[field]]
            targets[field] = symbols[values.index(label)]

    state = json.dumps(record.get("state"), ensure_ascii=False, indent=2)
    user_text = (
        "Return one answer for every field using the supplied answer symbols.\n\n"
        f"## State\n{state}\n"
        "## Decision schema\n" + "\n".join(schema_lines)
    )
    skeleton = json.dumps(dict.fromkeys(fields, MARKER), ensure_ascii=False, indent=4)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": skeleton},
    ]
    return {
        "messages": messages,
        "fields": fields,
        "options": options,
        "targets": targets if include_targets else None,
    }


def build_chat_text(
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = False,
) -> str:
    """Render the minimal Qwen chat format without a model-specific Jinja file."""
    parts = [
        f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        for message in messages
    ]
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def convert_typed_decisions_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one typed-decisions parquet row into a canonical record."""
    questions = json.loads(row["questions"])
    gold = json.loads(row["gold"])
    return {
        "id": row["id"],
        "state": json.loads(row["state"]),
        "questions": questions,
        "targets": {
            field: {"label": gold[field]["label"]}
            for field in questions
        },
    }
