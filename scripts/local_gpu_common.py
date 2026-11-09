"""Dependency-free helpers shared by the local GPU chapter scripts."""

from __future__ import annotations

import ast
import math
import re
import sys
import types
from dataclasses import dataclass
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


def normalized_group_advantages(
    rewards: Iterable[float], epsilon: float = 1e-4
) -> list[float]:
    """DAPO's standardised group advantage: (r - mean) / (std + eps), N-1 denominator.

    GRPO mean-centres only. DAPO divides by the sample standard deviation so every
    group contributes comparable gradient magnitude regardless of reward spread.
    A degenerate group (all rewards equal) yields zeros, which is the intended
    no-signal behaviour rather than a division by zero.
    """
    values = [float(reward) for reward in rewards]
    if not values:
        raise ValueError("A DAPO group must contain at least one reward")
    mean = sum(values) / len(values)
    if len(values) == 1:
        return [0.0]
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    if std == 0.0:
        return [0.0] * len(values)
    denominator = std + epsilon
    return [(value - mean) / denominator for value in values]


def is_effective_group(
    rewards: Iterable[float], expected_group_size: int
) -> bool:
    """DAPO Dynamic Sampling: keep only groups with both a right and a wrong answer.

    All-correct and all-wrong groups have zero variance, so they carry no learning
    signal and are dropped before the backward pass.
    """
    correct = sum(1 for reward in rewards if reward > 0)
    return 0 < correct < expected_group_size


def clipped_objectives(
    ratios: Iterable[float],
    advantages: Iterable[float],
    *,
    clip_low: float = 0.8,
    clip_high: float = 1.28,
) -> list[float]:
    """Per-token DAPO/PPO surrogate before reduction.

    objective_t = min(ratio_t * A_t, clip(ratio_t, low, high) * A_t)

    ``clip_high`` defaults to 1.28 ("Clip-Higher") so positive-advantage tokens may
    grow faster than PPO's symmetric 1.2 bound, which is DAPO's main stability fix
    for the entropy collapse of standard GRPO.
    """
    ratio_values = [float(value) for value in ratios]
    advantage_values = [float(value) for value in advantages]
    if len(ratio_values) != len(advantage_values):
        raise ValueError("ratios and advantages must have the same length")
    if not ratio_values:
        raise ValueError("clipped_objectives requires at least one token")
    return [
        min(ratio * advantage, min(max(ratio, clip_low), clip_high) * advantage)
        for ratio, advantage in zip(ratio_values, advantage_values, strict=True)
    ]


def dapo_loss(
    sequences: Iterable[tuple[Iterable[float], Iterable[float]]],
    *,
    clip_low: float = 0.8,
    clip_high: float = 1.28,
    reduction: str = "token",
) -> float:
    """Reduce a batch of ``(ratios, advantages)`` sequences to one scalar loss.

    ``reduction="token"`` (DAPO) pools every completion token and weights tokens
    equally, so long answers dominate. ``reduction="sample"`` (GRPO) averages each
    answer first, so every answer contributes equally. The difference is the whole
    point of DAPO's token-level policy-gradient loss.
    """
    if reduction not in {"token", "sample"}:
        raise ValueError(f"Unknown reduction: {reduction}")
    per_sequence: list[list[float]] = []
    for ratios, advantages in sequences:
        per_sequence.append(
            clipped_objectives(ratios, advantages, clip_low=clip_low, clip_high=clip_high)
        )
    if not per_sequence:
        raise ValueError("dapo_loss requires at least one sequence")
    if reduction == "token":
        pooled = [value for sequence in per_sequence for value in sequence]
        return -sum(pooled) / len(pooled)
    sequence_means = [sum(sequence) / len(sequence) for sequence in per_sequence]
    return -sum(sequence_means) / len(sequence_means)


def sequence_ratio(
    sampling_logprobs: Iterable[float], current_logprobs: Iterable[float]
) -> float:
    """GSPO's sequence-level importance ratio: the geometric mean of token ratios.

    log_ratio = mean_t(current_t - sampling_t)
    ratio     = exp(log_ratio)

    Averaging in log space keeps the ratio scale-free with respect to completion
    length, which is the whole point: a 20-token and a 200-token answer are
    compared on the same scale instead of the longer one being penalised.
    """
    sampling = [float(value) for value in sampling_logprobs]
    current = [float(value) for value in current_logprobs]
    if not sampling or len(sampling) != len(current):
        raise ValueError("sampling and current logprobs must be non-empty and equal length")
    log_ratio = sum(
        now - old for now, old in zip(current, sampling, strict=True)
    ) / len(sampling)
    return math.exp(log_ratio)


def gspo_loss(
    sequences: Iterable[tuple[Iterable[float], Iterable[float], float]],
    *,
    clip_low: float = 3e-4,
    clip_high: float = 4e-4,
) -> tuple[float, list[float]]:
    """GSPO's clipped objective over whole sequences.

    Each item is ``(sampling_logprobs, current_logprobs, advantage)``. Every
    sequence gets one ratio and one clipped objective; the loss averages over
    sequences, so a long answer cannot dominate merely by having more tokens.

    Returns ``(loss, ratios)`` so callers can log the ratio distribution.
    """
    objectives: list[float] = []
    ratios: list[float] = []
    for sampling, current, advantage in sequences:
        ratio = sequence_ratio(sampling, current)
        clipped = min(max(ratio, 1.0 - clip_low), 1.0 + clip_high)
        unclipped_objective = ratio * advantage
        clipped_objective = clipped * advantage
        objectives.append(min(unclipped_objective, clipped_objective))
        ratios.append(ratio)
    if not objectives:
        raise ValueError("gspo_loss requires at least one sequence")
    return -sum(objectives) / len(objectives), ratios


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


def _is_dataclass_decorator(node: ast.AST) -> bool:
    """Accept ``@dataclass`` and ``@dataclass(...)``; reject anything else."""
    if isinstance(node, ast.Name):
        return node.id == "dataclass"
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dataclass"
    )


def load_original_module(
    name: str,
    path: str | Path,
    required: tuple[str, ...] = (),
    namespace: dict[str, Any] | None = None,
) -> types.ModuleType:
    """Load an original chapter's pure constants/functions without its PyTRIO imports.

    ``required`` is the chapter's real dependency list; anything the original file
    no longer exposes must fail here rather than after a 12GB model download.
    ``namespace`` supplies stand-ins for heavy imports the kept functions use, so a
    chapter's reward logic can be reused without installing its full dependency set.
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
        # DAPO
        "extract_last_boxed",
        "answers_equivalent",
        "soft_overlong_penalty",
        "score_answer",
    }
    wanted_classes = {"RewardResult"}
    wanted_constants = {
        "QUESTION_SUFFIX",
        "FEWSHOT_PREFIX",
        "LOSS_FNS",
        "CHOICE_LETTERS",
        "DEFAULT_MODEL",
        "IMAGE_PAD_TOKEN",
        "BOXED_CHOICE_PATTERN",
        "ANSWER_WINDOW_CHARS",
    }
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted_functions:
            keep.append(node)
            kept_names.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name in wanted_classes:
            # Dataclasses such as DAPO's RewardResult are pure structure: keep them
            # only when every decorator is one of the safe stdlib ones.
            if all(_is_dataclass_decorator(item) for item in node.decorator_list):
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
    module.__dict__.update({"re": re, "Path": Path, "dataclass": dataclass, "math": math})
    if namespace:
        module.__dict__.update(namespace)
    # dataclass() resolves annotations through sys.modules[cls.__module__].
    sys.modules[name] = module
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
