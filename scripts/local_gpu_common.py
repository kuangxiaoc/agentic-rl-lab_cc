"""Dependency-free helpers shared by the local GPU chapter scripts."""

from __future__ import annotations

import ast
import math
import re
import types
from pathlib import Path
from typing import Any, Iterable


_MISSING = object()


def scalar_losses(
    old_logprob: float,
    current_logprob: float,
    advantage: float,
    loss_fn: str = "all",
    clip_high: float = 1.2,
) -> dict[str, float]:
    """Return the three scalar objectives from the original Loss Functions chapter."""
    ratio = math.exp(current_logprob - old_logprob)
    clipped = min(max(ratio, 1.0 / clip_high), clip_high)
    objectives = {
        "importance_sampling": -(ratio * advantage),
        "ppo": -min(ratio * advantage, clip_high * advantage),
        "cispo": -(clipped * current_logprob * advantage),
    }
    if loss_fn == "all":
        return objectives
    if loss_fn not in objectives:
        raise ValueError(f"Unknown loss function: {loss_fn}")
    return {loss_fn: objectives[loss_fn]}


def group_advantages(rewards: Iterable[float]) -> list[float]:
    """Mean-center rewards inside one GRPO group, matching the original scripts."""
    values = [float(reward) for reward in rewards]
    if not values:
        raise ValueError("A GRPO group must contain at least one reward")
    mean = sum(values) / len(values)
    return [reward - mean for reward in values]


def completion_valid_length(tokens: Iterable[int], eos_ids: set[int] | frozenset[int]) -> int:
    """Include the first EOS token; ignore any tokens generated after it."""
    values = list(tokens)
    for index, token in enumerate(values):
        if token in eos_ids:
            return index + 1
    return len(values)


def _ast_literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return _MISSING


def load_original_module(name: str, path: str | Path) -> types.ModuleType:
    """Load an original chapter's pure constants/functions without its PyTRIO imports."""
    source_path = Path(path).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    keep: list[ast.stmt] = []
    wanted_functions = {
        "extract_boxed",
        "normalize_answer",
        "grade_answer",
        "extract_gsm8k_answer",
        "extract_choice",
        "format_question",
    }
    wanted_constants = {
        "QUESTION_SUFFIX",
        "FEWSHOT_PREFIX",
        "LOSS_FNS",
        "CHOICE_LETTERS",
        "DEFAULT_MODEL",
        "IMAGE_PAD_TOKEN",
        "BOXED_CHOICE_PATTERN",
    }
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted_functions:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if names and all(name.isupper() for name in names) and any(
                name in wanted_constants for name in names
            ) and _ast_literal(node.value) is not _MISSING:
                keep.append(node)
    module = types.ModuleType(name)
    module.__file__ = str(source_path)
    module.__dict__.update({"re": re, "Path": Path})
    if "BOXED_CHOICE_PATTERN" in wanted_constants:
        module.__dict__["BOXED_CHOICE_PATTERN"] = re.compile(
            r"\\boxed\s*\{\s*([A-D])\s*\}", re.IGNORECASE
        )
    tree_out = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(tree_out)
    compiled = compile(tree_out, str(source_path), "exec")
    exec(compiled, module.__dict__)
    return module
