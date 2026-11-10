"""15 Spec-o3: local Qwen3.5 SFT + multimodal tool GRPO on one GPU.

The original chapter uses PyTRIO for remote SFT/RL and keeps the interleaved
spectrum-tool loop in local Python. This version keeps the original action
protocol and spectral tool, but runs the vision-language model, SFT loss,
sampling logprobs, PPO loss, backward pass, and optimizer step on the local
ModelScope GPU.

Smoke mode deliberately uses one SFT row and one benchmark question with two
short trajectories. It proves the real image -> tool -> image -> answer loop
and a real optimizer update; it is not the paper-scale 2-epoch SFT/GRPO run.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(EXPERIMENT_DIR))

import local_gpu_common as common  # noqa: E402
import local_gpu_trainer as trainer  # noqa: E402
import modelscope_lib as lib  # noqa: E402

sys.modules["pytrio"] = trainer.local_trio_compat()

import prepare_data as spec_data  # noqa: E402
import protocol as spec_protocol  # noqa: E402
from tools import spectral_visualization_tool  # noqa: E402


MODEL_ID = "Qwen/Qwen3.5-4B"
IMAGE_TOKEN_ID = 248056
IMAGE_PAD = "<|image_pad|>"
FINAL_ANSWER_REMINDER = (
    "\n\nYou have used the available tool turn. Do not call the tool again. "
    "After thinking, output exactly one final answer in this format: "
    r"<answer>\boxed{YES}justification</answer> or "
    r"<answer>\boxed{NO}justification</answer>. "
    "Do not put any text outside the answer tags."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地 GPU 版 Spec-o3")
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--profile",
        default=os.environ.get("RUN_PROFILE", "smoke"),
        choices=("smoke", "full"),
    )
    parser.add_argument("--stage", choices=("sft", "grpo", "both"), default="both")
    parser.add_argument("--sft-steps", type=int, default=0)
    parser.add_argument("--rl-steps", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--rl-learning-rate", type=float, default=1e-6)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="默认 WORKSPACE_ROOT/datasets/spec-o3",
    )
    return parser.parse_args()


def _model_dir(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    for base in (root / "cache" / "modelscope-models", root / "cache" / "modelscope" / "models"):
        if not base.exists():
            continue
        for config in base.rglob("config.json"):
            if "qwen3.5-4b" in str(config.parent).lower():
                return config.parent.resolve()
    return trainer.download_modelscope_model(MODEL_ID, root)


def _data_root(root: Path, explicit: Path | None) -> Path:
    return explicit.expanduser().resolve() if explicit else root / "datasets" / "spec-o3"


def _expand_image_tokens(
    ids: list[int],
    masks: list[int],
    grids: Any,
    image_token_id: int,
    merge_size: int,
) -> tuple[list[int], list[int]]:
    """Expand tokenizer placeholders exactly as the vision processor does."""
    grid_rows = grids.tolist() if hasattr(grids, "tolist") else list(grids)
    expanded_ids: list[int] = []
    expanded_masks: list[int] = []
    image_index = 0
    for token, mask in zip(ids, masks, strict=True):
        if token != image_token_id:
            expanded_ids.append(token)
            expanded_masks.append(mask)
            continue
        if image_index >= len(grid_rows):
            raise ValueError("more image placeholders than image_grid_thw rows")
        grid = [int(value) for value in grid_rows[image_index]]
        count = (grid[0] * grid[1] * grid[2]) // (merge_size**2)
        expanded_ids.extend([image_token_id] * count)
        expanded_masks.extend([0] * count)
        image_index += 1
    if image_index != len(grid_rows):
        raise ValueError("image_grid_thw has unused rows")
    return expanded_ids, expanded_masks


def _supervised_inputs(
    messages: list[dict[str, Any]],
    image_paths: list[str],
    tokenizer: Any,
    processor: Any,
    *,
    generation: bool = False,
) -> dict[str, Any]:
    """Return processor tensors plus an assistant-only token mask."""
    token_result = tokenizer.apply_chat_template(
        messages,
        tools=spec_protocol.TOOLS,
        chat_template=spec_protocol.TEMPLATE,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        add_generation_prompt=generation,
        enable_thinking=True,
    )
    ids = list(token_result["input_ids"])
    masks = list(token_result["assistant_masks"])
    processed = processor.apply_chat_template(
        messages,
        tools=spec_protocol.TOOLS,
        chat_template=spec_protocol.TEMPLATE,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=generation,
        enable_thinking=True,
    )
    merge_size = int(
        getattr(processor.image_processor, "merge_size", 2)
    )
    expanded_ids, expanded_masks = _expand_image_tokens(
        ids,
        masks,
        processed["image_grid_thw"],
        IMAGE_TOKEN_ID,
        merge_size,
    )
    processor_ids = processed["input_ids"][0].tolist()
    if expanded_ids != processor_ids:
        raise ValueError(
            "tokenizer/processor input_ids differ after image expansion; "
            f"tokenizer={len(expanded_ids)} processor={len(processor_ids)}"
        )
    result = {key: value for key, value in dict(processed).items()}
    result["assistant_mask"] = expanded_masks
    return result


def _move(inputs: dict[str, Any], device: Any, torch: Any) -> dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(torch.bfloat16)
        moved[key] = value
    return moved


def _sft_loss(
    model: Any,
    inputs: dict[str, Any],
    torch: Any,
) -> tuple[Any, int]:
    mask = inputs.pop("assistant_mask")
    outputs = model(**inputs, use_cache=False)
    logits = outputs.logits[0, :-1].float()
    targets = inputs["input_ids"][0, 1:]
    token_mask = torch.tensor(mask[1:], device=targets.device, dtype=torch.bool)
    if token_mask.sum() == 0:
        raise ValueError("assistant mask is empty; refusing a fake SFT update")
    losses = torch.nn.functional.cross_entropy(
        logits,
        targets,
        reduction="none",
    )
    return (losses * token_mask).sum() / token_mask.sum(), int(token_mask.sum().item())


def _load_sft_rows(data_root: Path, limit: int) -> list[dict[str, Any]]:
    root = data_root / "sft-raw"
    manifest = root / "train_sharegpt.json"
    if not manifest.exists():
        raise trainer.DataUnavailable(f"Spec-o3 SFT manifest missing: {manifest}")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if not raw:
        raise trainer.DataUnavailable("Spec-o3 SFT manifest is empty")
    rows = [spec_data.convert_sft(sample, root.resolve()) for sample in raw[:limit]]
    if not rows:
        raise trainer.DataUnavailable("Spec-o3 SFT rows are empty")
    return rows


def _bench_rows(data_root: Path, limit: int) -> list[dict[str, Any]]:
    """Extract a few benchmark rows and their images/spectra on demand."""
    root = data_root / "bench-raw"
    parquet_files = sorted((root / "data").glob("train_*.parquet"))
    if not parquet_files:
        raise trainer.DataUnavailable(
            f"Spec-o3 bench parquet missing under {root / 'data'}"
        )
    image_dir = data_root / "prepared" / "images"
    spectrum_dir = data_root / "prepared" / "spectra"
    image_dir.mkdir(parents=True, exist_ok=True)
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    import numpy as np
    import pyarrow.parquet as pq

    for path in parquet_files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
            for sample in batch.to_pylist():
                if len(rows) >= limit:
                    return rows
                info = sample["extra_info"]
                name = f"{path.stem}-{info['index']}"
                images = []
                for index, image in enumerate(sample["images"]):
                    image_path = image_dir / f"{name}-{index}.png"
                    image_path.write_bytes(image["bytes"])
                    images.append(str(image_path.resolve()))
                image_paths = iter(images)
                messages = [
                    {
                        "role": message["role"],
                        "content": spec_data.multimodal_content(
                            message["content"], image_paths
                        ),
                    }
                    for message in sample["prompt"]
                ]
                spectrum = info["tools_kwargs"]["spectral_visualization_tool"][
                    "create_kwargs"
                ].copy()
                spectrum["redshift"] = (
                    0.0 if spectrum["redshift"] is None else spectrum["redshift"]
                )
                spectrum_path = spectrum_dir / f"{name}.npz"
                np.savez_compressed(spectrum_path, **spectrum)
                rows.append(
                    {
                        "id": name,
                        "task": path.stem.split("_", 1)[1],
                        "source_name": info["name"],
                        "label": sample["reward_model"]["ground_truth"],
                        "messages": messages,
                        "images": images,
                        "spectrum": str(spectrum_path.resolve()),
                    }
                )
    return rows


def _generate_turn(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, Any],
    torch: Any,
    device: Any,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    num_samples: int = 1,
) -> tuple[Any, list[list[int]], list[list[float]], list[str]]:
    prompt_len = inputs["input_ids"].shape[1]
    # assistant_mask is for the supervised loss, not a model/generate kwarg.
    generation_inputs = {
        key: value for key, value in inputs.items() if key != "assistant_mask"
    }
    was_training = model.training
    model.eval()
    torch.manual_seed(seed)
    try:
        with torch.no_grad():
            output = model.generate(
                **generation_inputs,
                do_sample=True,
                num_return_sequences=num_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=trainer.pad_token_id(tokenizer),
                return_dict_in_generate=True,
                output_scores=True,
            )
    finally:
        if was_training:
            model.train()
    completions, logprobs, texts = [], [], []
    eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
    for index, sequence in enumerate(output.sequences):
        tokens = sequence[prompt_len:].tolist()
        tokens = tokens[: common.completion_valid_length(tokens, eos_ids)]
        values = trainer._sampling_scores_to_logprobs(
            output.scores, index, tokens, torch
        )
        completions.append(tokens)
        logprobs.append(values)
        texts.append(tokenizer.decode(tokens, skip_special_tokens=False))
    return output, completions, logprobs, texts


def _append_completion_inputs(
    prompt_inputs: dict[str, Any],
    completion: list[int],
    torch: Any,
) -> dict[str, Any]:
    full_ids = torch.cat(
        [
            prompt_inputs["input_ids"],
            torch.tensor([completion], device=prompt_inputs["input_ids"].device),
        ],
        dim=1,
    )
    attention = torch.cat(
        [
            prompt_inputs["attention_mask"],
            torch.ones((1, len(completion)), device=prompt_inputs["attention_mask"].device),
        ],
        dim=1,
    )
    mm = prompt_inputs.get("mm_token_type_ids")
    if mm is not None:
        mm = torch.cat(
            [
                mm,
                torch.zeros((1, len(completion)), device=mm.device, dtype=mm.dtype),
            ],
            dim=1,
        )
    result = {
        key: value
        for key, value in prompt_inputs.items()
        if key not in {"input_ids", "attention_mask", "mm_token_type_ids"}
    }
    result.update(
        {
            "input_ids": full_ids,
            "attention_mask": attention,
        }
    )
    if mm is not None:
        result["mm_token_type_ids"] = mm
    return result


def _trajectory_reward(
    trajectory: dict[str, Any],
    format_penalty: float = 0.2,
) -> tuple[float, bool]:
    final = trajectory["turns"][-1]["action"] if trajectory["turns"] else {"kind": "invalid"}
    correct = final.get("kind") == "answer" and final.get("prediction") == trajectory["label"]
    reward = float(correct) - format_penalty * float(not trajectory["format_ok"])
    return reward, correct


def _append_assistant_turn(messages: list[dict[str, Any]], text: str) -> None:
    reasoning, _, content = text.partition("</think>")
    messages.append(
        {
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": content.removesuffix("<|im_end|>").strip(),
            "tool_calls": [],
        }
    )


def _append_repair_prompt(messages: list[dict[str, Any]]) -> None:
    messages.append({"role": "user", "content": FINAL_ANSWER_REMINDER.strip()})


def _rollout_one(
    row: dict[str, Any],
    model: Any,
    tokenizer: Any,
    processor: Any,
    torch: Any,
    device: Any,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    messages = deepcopy(row["messages"])
    image_paths = list(row["images"])
    turns = []
    prediction = None
    format_ok = False
    for turn in range(args.max_turns):
        turn_messages = messages
        if turn + 1 == args.max_turns and turn > 0:
            turn_messages = deepcopy(messages)
            turn_messages[-1] = {
                **turn_messages[-1],
                "content": (
                    f"{turn_messages[-1]['content']}{FINAL_ANSWER_REMINDER}"
                ),
            }
        prompt_inputs = _supervised_inputs(
            turn_messages,
            image_paths,
            tokenizer,
            processor,
            generation=True,
        )
        prompt_inputs = _move(prompt_inputs, device, torch)
        _, completions, logprobs, texts = _generate_turn(
            model,
            tokenizer,
            prompt_inputs,
            torch,
            device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=seed + turn,
        )
        completion = completions[0]
        text = texts[0]
        action = spec_protocol.parse_action(text)
        train_inputs = _append_completion_inputs(prompt_inputs, completion, torch)
        turns.append(
            {
                "text": text,
                "action": action,
                "completion": completion,
                "logprobs": logprobs[0],
                "train_inputs": train_inputs,
            }
        )
        if action["kind"] == "answer":
            prediction = action["prediction"]
            format_ok = True
            break
        if action["kind"] == "invalid":
            # Give the model one chance to repair malformed output instead of
            # silently ending every trajectory in the same no-signal state.
            if turn + 1 < args.max_turns:
                _append_assistant_turn(messages, text)
                _append_repair_prompt(messages)
                continue
            break
        if turn + 1 == args.max_turns:
            break
        with np.load(row["spectrum"]) as spectrum:
            wavelength = spectrum["wavelength"]
            flux = spectrum["flux"]
            redshift = float(spectrum["redshift"])
        output_dir = Path(args._run_dir) / "tool-images" / row["id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        observation = spectral_visualization_tool(
            wavelength,
            flux,
            redshift,
            action["wavelength_range"],
            action["label"],
            output_dir / f"turn-{turn}.png",
        )
        _append_assistant_turn(messages, text)
        messages.append({"role": "tool", "content": observation["content"]})
        image_paths.extend(observation["images"])
    return {
        "id": row["id"],
        "task": row["task"],
        "label": row["label"],
        "prediction": prediction,
        "format_ok": format_ok,
        "turns": turns,
    }


def _build_samples(
    trajectories: list[dict[str, Any]],
    advantages: list[float],
    torch: Any,
) -> list[trainer.Sample]:
    samples = []
    for trajectory, advantage in zip(trajectories, advantages, strict=True):
        if abs(advantage) <= 1e-12:
            continue
        for turn in trajectory["turns"]:
            if turn["action"].get("kind") not in {"tool", "answer"}:
                continue
            completion = turn["completion"]
            if not completion:
                continue
            inputs = turn["train_inputs"]
            extra = {
                key: value
                for key, value in inputs.items()
                if key not in {"input_ids", "attention_mask", "assistant_mask"}
            }
            samples.append(
                trainer.Sample(
                    input_ids=inputs["input_ids"][0].tolist(),
                    completion_ids=completion,
                    old_logprobs=turn["logprobs"],
                    advantages=[float(advantage)] * len(completion),
                    reward=float(trajectory["reward"]),
                    text=turn["text"],
                    extra_inputs=extra,
                )
            )
    return samples


def _train_samples(
    model: Any,
    optimizer: Any,
    samples: list[trainer.Sample],
    torch: Any,
    device: Any,
    *,
    clip_low: float = 0.8,
    clip_high: float = 1.2,
) -> tuple[list[float], int]:
    if not samples:
        return [], 0
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for sample in samples:
        loss = trainer.sequence_policy_loss(
            model,
            sample,
            torch,
            device,
            loss_fn="ppo",
            clip_low=clip_low,
            clip_high=clip_high,
        )
        (loss / len(samples)).backward()
        losses.append(float(loss.detach().item()))
    optimizer.step()
    torch.cuda.empty_cache()
    return losses, len(samples)


def _main() -> int:
    args = parse_args()
    root, run_dir, run_id, config = trainer.prepare_run(15, args)
    args._run_dir = str(run_dir)
    data_root = _data_root(root, args.data_root)
    sft_steps = args.sft_steps or (1 if args.profile == "smoke" else 2)
    rl_steps = args.rl_steps or (1 if args.profile == "smoke" else config["steps"])
    summary: dict[str, Any] = {
        "chapter": 15,
        "model": MODEL_ID,
        "dataset": "Spec-o3-ColdStartSFT + SpecVI-Bench",
        "data_root": str(data_root),
        "profile": args.profile,
        "stage": args.stage,
        "sft_steps": sft_steps,
        "rl_steps": rl_steps,
        "group_size": args.group_size,
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "max_seq_len": args.max_seq_len,
        "external_api_calls": 0,
        "study_order": [
            "experiments/spec-o3/readme.md",
            "experiments/spec-o3/protocol.py",
            "experiments/spec-o3/tools.py",
            "experiments/spec-o3/rollout.py",
            "experiments/spec-o3/local_gpu.py",
        ],
    }
    trainer.save_run_config(run_dir, summary)
    try:
        import torch

        device = trainer.check_gpu()
        model_dir = _model_dir(root, args.model_dir)
        model, tokenizer, processor = trainer.build_qlora_model(
            model_dir, rank=args.lora_rank, vision=True
        )
        model.train()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
        )
        history: list[dict[str, Any]] = []

        if args.stage in {"sft", "both"}:
            rows = _load_sft_rows(data_root, sft_steps)
            for step, row in enumerate(rows):
                inputs = _supervised_inputs(
                    row["messages"],
                    row["images"],
                    tokenizer,
                    processor,
                )
                inputs = _move(inputs, device, torch)
                loss, assistant_tokens = _sft_loss(model, inputs, torch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                metric = {
                    "stage": "sft",
                    "step": step,
                    "loss": float(loss.detach().item()),
                    "assistant_tokens": assistant_tokens,
                }
                history.append(metric)
                lib.append_metrics(root, 15, run_id, metric)
                print(json.dumps(metric, ensure_ascii=False), flush=True)
                torch.cuda.empty_cache()

        if args.stage in {"grpo", "both"}:
            rows = _bench_rows(data_root, max(1, rl_steps))
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=args.rl_learning_rate,
            )
            for step, row in enumerate(rows):
                trajectories = [
                    _rollout_one(
                        row,
                        model,
                        tokenizer,
                        processor,
                        torch,
                        device,
                        args,
                        seed=args.seed + step * 1000 + index,
                    )
                    for index in range(args.group_size)
                ]
                rewards = []
                for trajectory in trajectories:
                    reward, correct = _trajectory_reward(trajectory)
                    trajectory["reward"] = reward
                    trajectory["correct"] = correct
                    rewards.append(reward)
                mean = sum(rewards) / len(rewards)
                variance = (
                    sum((value - mean) ** 2 for value in rewards) / (len(rewards) - 1)
                    if len(rewards) > 1
                    else 0.0
                )
                std = variance**0.5
                advantages = [
                    (value - mean) / (std + 1e-6) if std > 0 else 0.0
                    for value in rewards
                ]
                samples = _build_samples(trajectories, advantages, torch)
                losses, updated = _train_samples(
                    model, optimizer, samples, torch, device
                )
                metric = {
                    "stage": "grpo",
                    "step": step,
                    "reward_mean": mean,
                    "success_rate": sum(item["correct"] for item in trajectories)
                    / len(trajectories),
                    "format_rate": sum(item["format_ok"] for item in trajectories)
                    / len(trajectories),
                    "tool_calls": sum(
                        sum(turn["action"].get("kind") == "tool" for turn in item["turns"])
                        for item in trajectories
                    ),
                    "trajectories": len(trajectories),
                    "train_samples": updated,
                    "loss_mean": sum(losses) / len(losses) if losses else 0.0,
                }
                history.append(metric)
                lib.append_metrics(root, 15, run_id, metric)
                with (run_dir / "samples.jsonl").open("a", encoding="utf-8") as file:
                    for trajectory, advantage in zip(trajectories, advantages, strict=True):
                        file.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "id": trajectory["id"],
                                    "task": trajectory["task"],
                                    "label": trajectory["label"],
                                    "prediction": trajectory["prediction"],
                                    "format_ok": trajectory["format_ok"],
                                    "reward": trajectory["reward"],
                                    "advantage": advantage,
                                    "turns": [
                                        {
                                            "text": turn["text"],
                                            "action": turn["action"],
                                            "completion_tokens": len(turn["completion"]),
                                        }
                                        for turn in trajectory["turns"]
                                    ],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                print(json.dumps(metric, ensure_ascii=False), flush=True)
                torch.cuda.empty_cache()

        summary.update(
            {
                "history": history,
                "adapter": str(trainer.save_adapter(model, run_dir)),
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
            15,
            run_id,
            "completed",
            {
                "summary": str(run_dir / "summary.json"),
                "sft_steps": sft_steps,
                "rl_steps": rl_steps,
            },
            replace_details=True,
        )
        print(f"chapter 15 completed: {run_dir}", flush=True)
        return 0
    except trainer.DataUnavailable as exc:
        (run_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        trainer.update_training_state(
            root, 15, run_id, "data_unavailable", {"error": str(exc)}
        )
        print(f"chapter 15 data_unavailable: {exc}", flush=True)
        return 0
    except Exception as exc:
        error = traceback.format_exc()
        (run_dir / "error.txt").write_text(error, encoding="utf-8")
        trainer.update_training_state(
            root, 15, run_id, "failed", {"error": f"{type(exc).__name__}: {exc}"}
        )
        print(error, file=sys.stderr, flush=True)
        return 1
    finally:
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(_main())
