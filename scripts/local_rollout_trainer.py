"""Helpers for training multi-turn rollouts on one local GPU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import local_gpu_trainer as trainer


@dataclass(frozen=True)
class TurnLike:
    """The subset of a rollout turn needed to build a training sample."""

    prompt_tokens: list[int]
    completion_tokens: list[int]
    logprobs: list[float]
    forced_prefix_tokens: int = 0


def build_multiturn_sample(
    turns: Iterable[TurnLike],
    advantage: float,
    *,
    reward: float = 0.0,
    text: str = "",
) -> trainer.Sample | None:
    """Concatenate prefix-extended turns and mask observation tokens.

    Only assistant completion tokens receive the trajectory advantage. Tool
    observations remain in the causal context, but their old logprob and
    advantage are zero, so they do not contribute a policy-gradient target.
    """
    turns = list(turns)
    if not turns:
        return None
    full_tokens: list[int] = []
    old_by_token: list[float] = []
    advantage_by_token: list[float] = []
    initial_prompt_length = len(turns[0].prompt_tokens)

    for index, turn in enumerate(turns):
        if len(turn.completion_tokens) != len(turn.logprobs):
            raise ValueError(
                f"turn {index} has {len(turn.completion_tokens)} tokens "
                f"but {len(turn.logprobs)} logprobs"
            )
        if index == 0:
            delta = turn.prompt_tokens
        elif turn.prompt_tokens[: len(full_tokens)] == full_tokens:
            delta = turn.prompt_tokens[len(full_tokens) :]
        else:
            raise ValueError(f"turn {index} prompt is not a prefix extension")
        full_tokens.extend(delta)
        old_by_token.extend([0.0] * len(delta))
        advantage_by_token.extend([0.0] * len(delta))
        full_tokens.extend(turn.completion_tokens)
        old_by_token.extend(turn.logprobs)
        forced = min(int(turn.forced_prefix_tokens), len(turn.completion_tokens))
        advantage_by_token.extend([0.0] * forced)
        advantage_by_token.extend(
            [float(advantage)] * (len(turn.completion_tokens) - forced)
        )

    if len(full_tokens) <= initial_prompt_length:
        return None
    return trainer.Sample(
        input_ids=full_tokens,
        completion_ids=full_tokens[initial_prompt_length:],
        old_logprobs=old_by_token[initial_prompt_length:],
        advantages=advantage_by_token[initial_prompt_length:],
        reward=reward,
        text=text,
    )


def train_trajectories(
    model: Any,
    optimizer: Any,
    trajectories: Iterable[Any],
    *,
    loss_fn: str,
    clip_low: float,
    clip_high: float,
    torch_module: Any,
    device: Any,
) -> tuple[list[float], int]:
    """Run one local optimizer step over nonzero-advantage trajectories."""
    samples: list[trainer.Sample] = []
    for trajectory in trajectories:
        if abs(float(trajectory.advantage)) <= 1e-12:
            continue
        sample = build_multiturn_sample(
            trajectory.turns,
            float(trajectory.advantage),
            reward=float(trajectory.reward),
            text=str(getattr(trajectory, "final_text", "")),
        )
        if sample is not None:
            samples.append(sample)
    if not samples:
        return [], 0

    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for sample in samples:
        loss = trainer.sequence_policy_loss(
            model,
            sample,
            torch_module,
            device,
            loss_fn=loss_fn,
            clip_low=clip_low,
            clip_high=clip_high,
        )
        (loss / len(samples)).backward()
        losses.append(float(loss.detach().item()))
    optimizer.step()
    torch_module.cuda.empty_cache()
    return losses, len(samples)
