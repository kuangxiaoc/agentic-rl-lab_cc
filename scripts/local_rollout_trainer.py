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


def build_turn_advantage_sample(trajectory: Any) -> trainer.Sample | None:
    """Build one PPO sample with a different advantage for each action span.

    AgentOPSD assigns credit per turn, not per trajectory. Assistant action
    tokens get their turn's advantage; prompt, tool, and observation tokens
    stay in the causal context with advantage zero.
    """
    steps = list(trajectory.steps)
    credits = list(trajectory.turn_credits)
    if not steps or len(steps) != len(credits):
        return None

    full_tokens = list(trajectory.full_student_tokens)
    advantages = [0.0] * len(full_tokens)
    old_logprobs = [0.0] * len(full_tokens)
    for step, credit in zip(steps, credits):
        start = int(step.action_start)
        end = int(step.action_end)
        if end - start != len(step.logprobs):
            raise ValueError(f"turn {step.index} action span/logprob mismatch")
        old_logprobs[start:end] = step.logprobs
        advantages[start:end] = [float(credit.advantage)] * (end - start)

    initial_length = int(trajectory.student_initial_length)
    if initial_length < 1 or initial_length >= len(full_tokens):
        return None
    return trainer.Sample(
        input_ids=full_tokens,
        completion_ids=full_tokens[initial_length:],
        old_logprobs=old_logprobs[initial_length:],
        advantages=advantages[initial_length:],
        reward=float(trajectory.reward),
        text="",
    )


def train_turn_advantage_trajectories(
    model: Any,
    optimizer: Any,
    trajectories: Iterable[Any],
    *,
    clip_low: float,
    clip_high: float,
    torch_module: Any,
    device: Any,
) -> tuple[list[float], int]:
    """Run one PPO optimizer step over AgentOPSD turn-level credits."""
    samples: list[trainer.Sample] = []
    for trajectory in trajectories:
        if abs(float(getattr(trajectory, "sequence_advantage", 0.0))) <= 1e-12:
            continue
        sample = build_turn_advantage_sample(trajectory)
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
            loss_fn="ppo",
            clip_low=clip_low,
            clip_high=clip_high,
        )
        (loss / len(samples)).backward()
        losses.append(float(loss.detach().item()))
    optimizer.step()
    torch_module.cuda.empty_cache()
    return losses, len(samples)


def build_step_sample(trajectory: Any) -> trainer.Sample | None:
    """Build one trajectory-level PPO sample from original step records."""
    turns = [
        TurnLike(
            prompt_tokens=list(step.prompt_tokens),
            completion_tokens=list(step.completion_tokens),
            logprobs=list(step.logprobs),
        )
        for step in trajectory.steps
        if step.completion_tokens
    ]
    if not turns:
        return None
    return build_multiturn_sample(
        turns,
        float(trajectory.advantage),
        reward=float(trajectory.reward),
        text="",
    )


def train_step_trajectories(
    model: Any,
    optimizer: Any,
    trajectories: Iterable[Any],
    *,
    clip_low: float,
    clip_high: float,
    torch_module: Any,
    device: Any,
) -> tuple[list[float], int]:
    """Run one PPO step over nonzero-advantage step-record trajectories."""
    samples: list[trainer.Sample] = []
    for trajectory in trajectories:
        if abs(float(trajectory.advantage)) <= 1e-12:
            continue
        sample = build_step_sample(trajectory)
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
            loss_fn="ppo",
            clip_low=clip_low,
            clip_high=clip_high,
        )
        (loss / len(samples)).backward()
        losses.append(float(loss.detach().item()))
    optimizer.step()
    torch_module.cuda.empty_cache()
    return losses, len(samples)


def build_segment_sample(start_state: Any, trajectory: Any) -> trainer.Sample | None:
    """Build a TEMPO actor sample for one macro-step branch."""
    steps = [step for step in trajectory.steps if step.completion_tokens]
    if not steps:
        return None

    base = list(start_state.token_prefix or [])
    full_tokens: list[int] = []
    old_by_token: list[float] = []
    advantage_by_token: list[float] = []

    for index, step in enumerate(steps):
        if len(step.completion_tokens) != len(step.logprobs):
            raise ValueError(
                f"turn {index} has {len(step.completion_tokens)} tokens "
                f"but {len(step.logprobs)} logprobs"
            )
        if not full_tokens:
            if base:
                if step.prompt_tokens[: len(base)] != base:
                    raise ValueError("first turn does not extend the saved prefix")
                full_tokens.extend(base)
                old_by_token.extend([0.0] * len(base))
                advantage_by_token.extend([0.0] * len(base))
                delta = step.prompt_tokens[len(base) :]
            else:
                delta = step.prompt_tokens
        elif step.prompt_tokens[: len(full_tokens)] == full_tokens:
            delta = step.prompt_tokens[len(full_tokens) :]
        else:
            raise ValueError(f"turn {index} prompt is not a prefix extension")

        full_tokens.extend(delta)
        old_by_token.extend([0.0] * len(delta))
        advantage_by_token.extend([0.0] * len(delta))
        full_tokens.extend(step.completion_tokens)
        old_by_token.extend(step.logprobs)
        advantage_by_token.extend(
            [float(trajectory.advantage)] * len(step.completion_tokens)
        )

    if trajectory.next_prompt_tokens is not None:
        if trajectory.next_prompt_tokens[: len(full_tokens)] != full_tokens:
            raise ValueError("final observation does not extend the branch")
        delta = trajectory.next_prompt_tokens[len(full_tokens) :]
        full_tokens.extend(delta)
        old_by_token.extend([0.0] * len(delta))
        advantage_by_token.extend([0.0] * len(delta))

    initial_length = len(base) if base else len(steps[0].prompt_tokens)
    if initial_length >= len(full_tokens):
        return None
    return trainer.Sample(
        input_ids=full_tokens,
        completion_ids=full_tokens[initial_length:],
        old_logprobs=old_by_token[initial_length:],
        advantages=advantage_by_token[initial_length:],
        reward=float(trajectory.segment_reward),
        text="",
    )


def build_critic_sample(
    sample: Any,
    advantage: float,
    *,
    reward: float,
) -> trainer.Sample | None:
    """Build a TEMPO critic sample from one generated value estimate."""
    completion = list(sample.completion_tokens)
    if not completion:
        return None
    if len(completion) != len(sample.logprobs):
        raise ValueError("critic token/logprob length mismatch")
    prompt = list(sample.prompt_tokens)
    full_tokens = [*prompt, *completion]
    old_by_token = [0.0] * len(prompt) + list(sample.logprobs)
    advantage_by_token = [0.0] * len(prompt) + [float(advantage)] * len(completion)
    return trainer.Sample(
        input_ids=full_tokens,
        completion_ids=completion,
        old_logprobs=old_by_token[len(prompt) :],
        advantages=advantage_by_token[len(prompt) :],
        reward=float(reward),
        text=str(sample.text),
    )


def train_mixed_samples(
    model: Any,
    optimizer: Any,
    samples: Iterable[trainer.Sample],
    *,
    clip_low: float,
    clip_high: float,
    torch_module: Any,
    device: Any,
) -> tuple[list[float], int]:
    """Accumulate actor and critic PPO gradients into one optimizer step."""
    samples = list(samples)
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
            loss_fn="ppo",
            clip_low=clip_low,
            clip_high=clip_high,
        )
        (loss / len(samples)).backward()
        losses.append(float(loss.detach().item()))
    optimizer.step()
    torch_module.cuda.empty_cache()
    return losses, len(samples)
