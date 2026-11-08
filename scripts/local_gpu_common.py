"""Dependency-free helpers shared by the local GPU chapter scripts."""

from __future__ import annotations

import ast
import math
import re
import types
from pathlib import Path
from typing import Any, Iterable


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


def _safe_constant(node: ast.AST, known_names: set[str]) -> bool:
    """Allow the constant shapes the original chapters use, and nothing else.

    Containers are joined with ``+`` to other module constants - FEWSHOT_PREFIX
    embeds QUESTION_SUFFIX - and some constants are compiled regexes, so a plain
    ``ast.literal_eval`` gate silently drops them and the chapter then fails with
    a confusing AttributeError in the middle of training.
    """
    if isinstance(node, ast.Attribute):
        # Flags such as re.IGNORECASE are module attributes, not literals.
        return isinstance(node.value, ast.Name) and node.value.id in {"re", "math"}
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_safe_constant(item, known_names) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (key is None or _safe_constant(key, known_names))
            and _safe_constant(value, known_names)
            for key, value in zip(node.keys, node.values, strict=True)
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _safe_constant(node.left, known_names) and _safe_constant(
            node.right, known_names
        )
    if isinstance(node, ast.Name):
        return node.id in known_names
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "compile"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "re"
    ):
        return all(_safe_constant(arg, known_names) for arg in node.args) and all(
            _safe_constant(keyword.value, known_names) for keyword in node.keywords
        )
    return False


def load_original_module(
    name: str, path: str | Path, required: tuple[str, ...] = ()
) -> types.ModuleType:
    """Load an original chapter's pure constants/functions without its PyTRIO imports.

    ``required`` is the chapter's real dependency list; anything the original file
    no longer exposes must fail here rather than after a 12GB model download.
    """
    source_path = Path(path).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    keep: list[ast.stmt] = []
    kept_names: set[str] = set()
    assigned_names = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
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
            kept_names.add(node.name)
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if (
                names
                and all(name.isupper() for name in names)
                and any(name in wanted_constants for name in names)
                and _safe_constant(
                    node.value, kept_names | (wanted_constants & assigned_names)
                )
            ):
                keep.append(node)
                kept_names.update(names)
    module = types.ModuleType(name)
    module.__file__ = str(source_path)
    module.__dict__.update({"re": re, "Path": Path})
    tree_out = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(tree_out)
    compiled = compile(tree_out, str(source_path), "exec")
    exec(compiled, module.__dict__)
    missing = [attr for attr in required if not hasattr(module, attr)]
    if missing:
        raise AttributeError(
            f"{source_path.name} did not expose {sorted(set(missing))}; "
            "extend load_original_module's wanted constants/functions"
        )
    return module
