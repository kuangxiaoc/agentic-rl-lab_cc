"""16 Tournament-GRPO: local single-GPU training with Tavily search and a local judge.

The original chapter uses PyTRIO for policy training, DeepSeek Search for tools,
and Jev as the pairwise judge. This local version keeps the original XML tool
protocol, format reward, tournament reward, group normalization, and PPO
objective, replacing only those three external services:

    local Qwen policy -> Tavily search -> local Qwen judge -> tournament reward
    -> one local PPO optimizer step

The local judge is the same frozen base model with the policy adapter disabled.
It is a mechanism-preserving fallback, not the paper's Qwen2.5-72B/Jev judge.
Every Tavily request is recorded in the persistent API budget ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import local_gpu_trainer as trainer  # noqa: E402
import modelscope_lib as lib  # noqa: E402

sys.modules["pytrio"] = trainer.local_trio_compat()
sys.path.insert(0, str(EXPERIMENT_DIR))

from data import load_examples  # noqa: E402
from judge import build_chat_prompt, build_judge_context, parse_winner  # noqa: E402
from protocol import build_prompt, parse_tool_call, tool_observation, transcript_text  # noqa: E402
from reward import score_group  # noqa: E402
from rollout import RolloutConfig, Trajectory  # noqa: E402
from search import ToolResult, truncate_tool_text  # noqa: E402


MODEL_ID = "Qwen/Qwen3.5-4B"
DATASET_PATH = "experiments/tournament-grpo/datasets/train.jsonl"
TAVILY_URL = "https://api.tavily.com/search"


class TavilyResearchTools:
    """Original XML tool surface, backed by Tavily search and optional Jina browse."""

    def __init__(
        self,
        root: Path,
        *,
        api_key: str,
        search_count: int = 3,
        timeout: float = 30.0,
    ) -> None:
        self.root = root
        self.api_key = api_key
        self.search_count = search_count
        self.timeout = timeout
        self.search_calls = 0

    @classmethod
    def from_env(cls, root: Path, search_count: int) -> "TavilyResearchTools":
        key = os.environ.get("TAVILY_API_KEY", "").strip()
        if not key:
            raise trainer.DataUnavailable(
                "TAVILY_API_KEY is missing; add it to the server .env before chapter 16"
            )
        return cls(root, api_key=key, search_count=search_count)

    def search(self, query: str) -> ToolResult:
        import httpx

        query = query.strip()[:400]
        if not query:
            return ToolResult("Error: query is required.", False)
        self.search_calls += 1
        lib.record_external_usage(
            self.root,
            "tavily",
            "tavily-search",
            0,
            0,
            estimated_cost_cny=0.0,
            allow_unlisted_model=True,
        )
        try:
            response = httpx.post(
                TAVILY_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "query": query,
                    "search_depth": "basic",
                    "max_results": self.search_count,
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout=self.timeout,
                trust_env=False,
            )
            response.raise_for_status()
            blocks = []
            for item in response.json().get("results", [])[: self.search_count]:
                title = str(item.get("title") or "").strip()
                url = str(item.get("url") or "").strip()
                snippet = str(item.get("content") or "").strip()
                if not (title or url or snippet):
                    continue
                blocks.append(
                    "<snippet>\n"
                    f"Title: {title}\nURL: {url}\nSearch Snippet: {snippet}\n"
                    "</snippet>"
                )
            content = "\n".join(blocks) or "<snippet>\nNo search results found.\n</snippet>"
            return ToolResult(f"<tool_output>\n{content}\n</tool_output>", True)
        except Exception as exc:  # noqa: BLE001 - the model receives tool errors as observations
            return ToolResult(f"Error performing google_search: {exc}", False)

    def browse(self, url: str) -> ToolResult:
        """Use Jina Reader when configured; otherwise return a clear tool error."""
        import httpx

        url = url.strip()
        if not url:
            return ToolResult("Error: url is required.", False)
        base = os.environ.get("JINA_READER_BASE_URL", "https://r.jina.ai").rstrip("/")
        headers = {}
        if os.environ.get("JINA_API_KEY"):
            headers["Authorization"] = f"Bearer {os.environ['JINA_API_KEY']}"
        try:
            response = httpx.get(
                f"{base}/{url}",
                headers=headers,
                timeout=self.timeout,
                follow_redirects=True,
                trust_env=False,
            )
            response.raise_for_status()
            text = response.text.strip()
            if not text:
                raise ValueError("empty webpage response")
            return ToolResult(
                "<tool_output>\n<webpage>\n"
                f"URL: {url}\nContent: {truncate_tool_text(text)}\n"
                "</webpage>\n</tool_output>",
                True,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(f"Error performing browse_webpage: {exc}", False)

    def execute(self, call: Any) -> ToolResult:
        try:
            if call.name == "google_search":
                result = self.search(call.arguments.get("query", ""))
            elif call.name == "browse_webpage":
                result = self.browse(call.arguments.get("url", ""))
            else:
                raise KeyError(call.name)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(f"Error when executing tool: {exc}", False)
        return ToolResult(truncate_tool_text(result.text), result.ok)


class LocalJudge:
    """Pairwise judge backed by the same frozen base model with adapters disabled."""

    backend = "local-qwen-base"
    model = MODEL_ID
    model_name = MODEL_ID
    long_context = False

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        device: Any,
        *,
        seed: int = 0,
        max_new_tokens: int = 96,
    ) -> None:
        self.policy = model
        self.tokenizer = tokenizer
        self.torch = torch_module
        self.device = device
        self.seed = seed
        self.max_new_tokens = max_new_tokens

    def judge(
        self,
        query: str,
        rubrics: list[dict[str, str]],
        answers: list[str],
    ) -> tuple[int, dict[str, Any]]:
        prompt = build_chat_prompt(build_judge_context(query, rubrics, answers))
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict deep-research tournament judge. "
                    "Return only valid JSON without markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in dict(inputs).items()
        }
        was_training = self.policy.training
        self.policy.eval()
        self.torch.manual_seed(self.seed)
        try:
            with self.torch.no_grad():
                if hasattr(self.policy, "disable_adapter"):
                    context = self.policy.disable_adapter()
                else:
                    context = _nullcontext()
                with context:
                    output = self.policy.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=trainer.pad_token_id(self.tokenizer),
                    )
            generated = output[0, inputs["input_ids"].shape[1] :]
            text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        finally:
            if was_training:
                self.policy.train()
            self.torch.cuda.empty_cache()
        return parse_winner(text, len(answers)), {"raw": text}


class _nullcontext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc_info: Any) -> bool:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 GPU 版 Tournament-GRPO")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--profile",
        default=os.environ.get("RUN_PROFILE", "smoke"),
        choices=("smoke", "full"),
    )
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--questions-per-batch", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument("--max-assistant-turns", type=int, default=3)
    parser.add_argument("--max-tool-calls", type=int, default=1)
    parser.add_argument("--max-response-tokens", type=int, default=2048)
    parser.add_argument("--max-model-tokens", type=int, default=4096)
    parser.add_argument("--search-count", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--clip-low", type=float, default=0.8)
    parser.add_argument("--clip-high", type=float, default=1.2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--judge-max-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / DATASET_PATH)
    parser.add_argument(
        "--force-first-tool-call",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="教学 smoke：首轮预填非空 think + search 开始标签，只让模型生成 query",
    )
    return parser.parse_args()


def _build_prompt_with_forced_call(tokenizer: Any, query: str) -> tuple[list[int], list[int]]:
    prompt = build_prompt(tokenizer, query)
    prefix = (
        "I will search for relevant evidence before answering.\n"
        "</think>\n"
        '<call_tool name="google_search">\n'
    )
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    return prompt, prefix_tokens


def _sample_turn(
    client: trainer.LocalSamplingClient,
    tokens: list[int],
    config: RolloutConfig,
    num_samples: int,
    seed: int,
) -> Any:
    budget = min(
        config.max_response_tokens,
        config.max_model_tokens - len(tokens),
    )
    if budget <= 0:
        raise trainer.DataUnavailable("tournament prompt exhausted the model context")
    return client.sample_async(
        prompt=trainer.LocalTokenPrompt.from_ints(tokens),
        num_samples=num_samples,
        sampling_params=trainer.LocalSamplingParams(
            max_tokens=budget,
            stop=["</call_tool>"],
            temperature=config.temperature,
            top_p=config.top_p,
            seed=seed,
        ),
        return_text=False,
    )


async def _rollout_batch_local(
    client: trainer.LocalSamplingClient,
    tokenizer: Any,
    tools: TavilyResearchTools,
    examples: list[Any],
    config: RolloutConfig,
    *,
    force_first_tool_call: bool,
) -> list[Trajectory]:
    """Local counterpart of ``rollout_batch_async`` with token-based prompts."""
    prompt_parts = [
        (
            _build_prompt_with_forced_call(tokenizer, example.query)
            if force_first_tool_call
            else (build_prompt(tokenizer, example.query), [])
        )
        for example in examples
    ]
    prompts = [prompt + prefix for prompt, prefix in prompt_parts]
    first = await asyncio.gather(
        *(
            _sample_turn(client, prompt, config, config.group_size, config.seed + index)
            for index, prompt in enumerate(prompts)
        )
    )
    trajectories = [
        Trajectory(example, question_index, group_index, prompt)
        for question_index, (example, (prompt, prefix)) in enumerate(
            zip(examples, prompt_parts, strict=True)
        )
        for group_index in range(config.group_size)
    ]
    for question_index, (_, prefix) in enumerate(prompt_parts):
        for group_index in range(config.group_size):
            trajectory = trajectories[question_index * config.group_size + group_index]
            trajectory.response_tokens.extend(prefix)
            trajectory.response_mask.extend([0] * len(prefix))
            trajectory.logprobs.extend([0.0] * len(prefix))
    sequences = [sequence for response in first for sequence in response.sequences]

    for turn in range(1, config.max_assistant_turns + 1):
        pending = []
        for trajectory, sequence in zip(trajectories, sequences, strict=True):
            if sequence is None:
                continue
            trajectory.response_tokens.extend(sequence.tokens)
            trajectory.response_mask.extend([1] * len(sequence.tokens))
            trajectory.logprobs.extend(sequence.logprobs)
            trajectory.assistant_turns += 1
            if len(trajectory.response_tokens) >= config.max_response_tokens:
                trajectory.finish_reason = "max_response_tokens"
                continue
            if trajectory.assistant_turns >= config.max_assistant_turns:
                trajectory.finish_reason = "max_turns"
                continue
            call_text = (
                trajectory.response_tokens
                if turn == 1 and force_first_tool_call
                else sequence.tokens
            )
            call = parse_tool_call(tokenizer.decode(call_text))
            if call is None:
                trajectory.finish_reason = "no_tool_call"
                continue
            if len(trajectory.tool_calls) >= config.max_tool_calls:
                trajectory.finish_reason = "max_tool_calls"
                continue
            pending.append((trajectory, call))

        if pending:
            loop = asyncio.get_running_loop()
            results = await asyncio.gather(
                *(
                    loop.run_in_executor(None, tools.execute, call)
                    for _, call in pending
                )
            )
            for (trajectory, call), result in zip(pending, results, strict=True):
                trajectory.tool_calls.append(call)
                trajectory.tool_errors += int(not result.ok)
                text = tool_observation(
                    result.text,
                    tools_exhausted=len(trajectory.tool_calls) >= config.max_tool_calls,
                )
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if len(trajectory.response_tokens) + len(tokens) >= config.max_response_tokens:
                    trajectory.finish_reason = "max_response_tokens"
                    continue
                trajectory.response_tokens.extend(tokens)
                trajectory.response_mask.extend([0] * len(tokens))
                trajectory.logprobs.extend([0.0] * len(tokens))

        active = [index for index, trajectory in enumerate(trajectories) if not trajectory.done]
        if not active:
            break
        responses = await asyncio.gather(
            *(
                _sample_turn(
                    client,
                    trajectories[index].prompt_tokens + trajectories[index].response_tokens,
                    config,
                    1,
                    config.seed + index + turn * 10000,
                )
                for index in active
            )
        )
        sequences = [None] * len(trajectories)
        for index, response in zip(active, responses, strict=True):
            sequences[index] = response.sequences[0]

    for trajectory in trajectories:
        trajectory.response_tokens = trajectory.response_tokens[: config.max_response_tokens]
        trajectory.response_mask = trajectory.response_mask[: config.max_response_tokens]
        trajectory.logprobs = trajectory.logprobs[: config.max_response_tokens]
    return trajectories


def _assign_rewards(
    trajectories: list[Trajectory],
    tokenizer: Any,
    judge: LocalJudge,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[int, list[int]] = {}
    for index, trajectory in enumerate(trajectories):
        groups.setdefault(trajectory.question_index, []).append(index)
    details: list[dict[str, Any]] = [{} for _ in trajectories]
    logs = []
    for question_index, indices in groups.items():
        group = [trajectories[index] for index in indices]
        solutions = [
            transcript_text(tokenizer, item.prompt_tokens, item.response_tokens)
            for item in group
        ]
        results, log = score_group(group[0].example, solutions, judge)
        rewards = _reward_scores(group, results)
        mean = sum(rewards) / len(rewards)
        variance = (
            sum((value - mean) ** 2 for value in rewards) / (len(rewards) - 1)
            if len(rewards) > 1
            else 0.0
        )
        std = variance**0.5
        for index, trajectory, result, reward in zip(
            indices, group, results, rewards, strict=True
        ):
            trajectory.reward = reward
            trajectory.advantage = (
                (reward - mean) / (std + 1e-6) if std > 0 else 0.0
            )
            details[index] = result
        logs.append(
            {
                "question_index": question_index,
                **log,
                "advantages": [trajectory.advantage for trajectory in group],
            }
        )
    return details, logs


def _build_sample(trajectory: Trajectory) -> trainer.Sample | None:
    if not trajectory.response_tokens:
        return None
    if len(trajectory.response_mask) != len(trajectory.response_tokens):
        raise ValueError("tournament response mask/token length mismatch")
    if len(trajectory.logprobs) != len(trajectory.response_tokens):
        raise ValueError("tournament logprob/token length mismatch")
    return trainer.Sample(
        input_ids=[*trajectory.prompt_tokens, *trajectory.response_tokens],
        completion_ids=list(trajectory.response_tokens),
        old_logprobs=list(trajectory.logprobs),
        advantages=[float(trajectory.advantage)] * len(trajectory.response_tokens),
        reward=float(trajectory.reward),
        text="",
        loss_mask=list(trajectory.response_mask),
    )


def _metrics(
    trajectories: list[Trajectory],
    details: list[dict[str, Any]],
    logs: list[dict[str, Any]],
    losses: list[float],
    updated: int,
    tools: TavilyResearchTools,
) -> dict[str, float]:
    count = max(len(trajectories), 1)
    matches = sum(len(log.get("matches", [])) for log in logs)
    return {
        "reward/mean": sum(item.reward for item in trajectories) / count,
        "reward/has_answer": sum(
            bool(detail.get("rollout_answer")) for detail in details
        )
        / count,
        "reward/format": sum(
            float(detail.get("format_reward", 0.0)) for detail in details
        )
        / count,
        "reward/tournament": sum(
            float(detail.get("tournament_reward", 0.0)) for detail in details
        )
        / count,
        "rollout/assistant_turns": sum(
            item.assistant_turns for item in trajectories
        )
        / count,
        "rollout/response_tokens": sum(
            len(item.response_tokens) for item in trajectories
        )
        / count,
        "rollout/tool_calls": sum(len(item.tool_calls) for item in trajectories)
        / count,
        "tools/error_rate": sum(item.tool_errors for item in trajectories)
        / max(sum(len(item.tool_calls) for item in trajectories), 1),
        "tavily/search_calls": float(tools.search_calls),
        "judge/comparisons": float(matches),
        "train/samples": float(updated),
        "train/loss_mean": sum(losses) / len(losses) if losses else 0.0,
    }


def _reward_scores(
    trajectories: list[Trajectory],
    details: list[dict[str, Any]],
) -> list[float]:
    """Keep tournament quality dominant and use format as a tie-breaker.

    The paper's unweighted sum can cancel when one candidate wins the
    tournament but fails format while the other loses the tournament but
    passes format. Weighting tournament 2x preserves its ordering while
    still giving the format check a non-degenerate signal.
    """
    return [
        2.0 * float(detail.get("tournament_reward", 0.0))
        + float(detail.get("format_reward", 0.0))
        for detail in details
    ]


def main() -> int:
    args = parse_args()
    root, run_dir, run_id, config = trainer.prepare_run(16, args)
    steps = args.steps or (1 if args.profile == "smoke" else config["steps"])
    group_size = args.group_size or (2 if args.profile == "smoke" else config["group_size"])
    if (
        steps < 1
        or args.questions_per_batch < 1
        or group_size < 2
        or args.max_assistant_turns < 1
        or args.max_tool_calls < 1
        or args.max_response_tokens < 256
        or args.max_model_tokens < 512
    ):
        raise ValueError("Tournament-GRPO step/group/context limits are invalid")

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    summary: dict[str, Any] = {
        "chapter": 16,
        "model": MODEL_ID,
        "dataset": str(args.data),
        "profile": args.profile,
        "steps": steps,
        "questions_per_batch": args.questions_per_batch,
        "group_size": group_size,
        "max_assistant_turns": args.max_assistant_turns,
        "max_tool_calls": args.max_tool_calls,
        "max_response_tokens": args.max_response_tokens,
        "search_backend": "tavily",
        "judge_backend": "local Qwen base with adapter disabled",
        "force_first_tool_call": args.force_first_tool_call,
        "external_api_calls": "tavily only; budget ledger enabled",
        "study_order": [
            "experiments/tournament-grpo/readme.md",
            "experiments/tournament-grpo/protocol.py",
            "experiments/tournament-grpo/rollout.py",
            "experiments/tournament-grpo/reward.py",
            "experiments/tournament-grpo/judge.py",
            "experiments/tournament-grpo/local_gpu.py",
        ],
    }
    trainer.save_run_config(run_dir, summary)

    try:
        if not args.data.exists():
            raise trainer.DataUnavailable(
                f"Tournament data missing: {args.data}; run prepare_data.py first"
            )
        examples = load_examples(args.data)
        if not examples:
            raise trainer.DataUnavailable("Tournament train.jsonl is empty")
        tools = TavilyResearchTools.from_env(root, args.search_count)
        model_dir = trainer.download_modelscope_model(MODEL_ID, root)
        import torch

        device = trainer.check_gpu()
        model, tokenizer, _ = trainer.build_qlora_model(
            model_dir, rank=args.lora_rank, vision=False
        )
        model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
        )
        client = trainer.LocalSamplingClient(
            model, tokenizer, torch, device, task_id=f"tournament-{run_id}"
        )
        judge = LocalJudge(
            model,
            tokenizer,
            torch,
            device,
            seed=args.seed,
            max_new_tokens=args.judge_max_tokens,
        )
        history: list[dict[str, Any]] = []
        offset = 0
        for step in range(steps):
            batch = [
                examples[(offset + index) % len(examples)]
                for index in range(args.questions_per_batch)
            ]
            offset += len(batch)
            rollout_config = RolloutConfig(
                group_size=group_size,
                max_response_tokens=args.max_response_tokens,
                max_model_tokens=args.max_model_tokens,
                max_assistant_turns=args.max_assistant_turns,
                max_tool_calls=args.max_tool_calls,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed + step * 1000,
            )
            trajectories = asyncio.run(
                _rollout_batch_local(
                    client,
                    tokenizer,
                    tools,
                    batch,
                    rollout_config,
                    force_first_tool_call=args.force_first_tool_call,
                )
            )
            details, logs = _assign_rewards(trajectories, tokenizer, judge)
            with (run_dir / "judge.jsonl").open("a", encoding="utf-8") as file:
                for log in logs:
                    file.write(json.dumps({"step": step, **log}, ensure_ascii=False) + "\n")
            samples = [
                sample
                for trajectory in trajectories
                if abs(trajectory.advantage) > 1e-12
                for sample in [_build_sample(trajectory)]
                if sample is not None
            ]
            losses, updated = rollout_train_samples(
                model,
                optimizer,
                samples,
                clip_low=args.clip_low,
                clip_high=args.clip_high,
                torch_module=torch,
                device=device,
            )
            metric = {
                "step": step,
                **_metrics(trajectories, details, logs, losses, updated, tools),
            }
            history.append(metric)
            lib.append_metrics(root, 16, run_id, metric)
            with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as file:
                for trajectory, detail in zip(trajectories, details, strict=True):
                    file.write(
                        json.dumps(
                            {
                                "step": step,
                                "id": trajectory.example.id,
                                "query": trajectory.example.query,
                                "response_text": tokenizer.decode(
                                    trajectory.response_tokens, skip_special_tokens=True
                                ),
                                "reward": trajectory.reward,
                                "advantage": trajectory.advantage,
                                "detail": detail,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            print(json.dumps(metric, ensure_ascii=False), flush=True)

        summary.update(
            {
                "history": history,
                "adapter": str(trainer.save_adapter(model, run_dir)),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "tavily_calls": tools.search_calls,
                "examples_available": len(examples),
            }
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        trainer.update_training_state(
            root,
            16,
            run_id,
            "completed",
            {
                "summary": str(run_dir / "summary.json"),
                "steps": steps,
                "tavily_calls": tools.search_calls,
            },
            replace_details=True,
        )
        print(f"chapter 16 completed: {run_dir}", flush=True)
        return 0
    except trainer.DataUnavailable as exc:
        (run_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        trainer.update_training_state(
            root, 16, run_id, "data_unavailable", {"error": str(exc)}
        )
        print(f"chapter 16 data_unavailable: {exc}", flush=True)
        return 0
    except Exception as exc:
        error = traceback.format_exc()
        (run_dir / "error.txt").write_text(error, encoding="utf-8")
        trainer.update_training_state(
            root, 16, run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"}
        )
        print(error, file=sys.stderr, flush=True)
        return 1
    finally:
        gc.collect()


def rollout_train_samples(
    model: Any,
    optimizer: Any,
    samples: list[trainer.Sample],
    *,
    clip_low: float,
    clip_high: float,
    torch_module: Any,
    device: Any,
) -> tuple[list[float], int]:
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


if __name__ == "__main__":
    raise SystemExit(main())
