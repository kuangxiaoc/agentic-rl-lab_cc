"""Shared local GPU primitives for 02/03. Imported only by those chapter scripts."""

from __future__ import annotations

import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import local_gpu_common as common  # noqa: E402
import modelscope_lib as lib  # noqa: E402


PROFILES = {
    "smoke": {"steps": 2, "batch_size": 1, "group_size": 2, "max_new_tokens": 128},
    "full": {"steps": 10, "batch_size": 2, "group_size": 4, "max_new_tokens": 512},
}


class DataUnavailable(RuntimeError):
    """Raised when a chapter's exact dataset cannot be prepared."""


def profile_config(profile: str | None = None) -> dict[str, int]:
    profile = profile or os.environ.get("RUN_PROFILE", "smoke")
    if profile not in PROFILES:
        raise ValueError(f"Unknown RUN_PROFILE: {profile}")
    return dict(PROFILES[profile])


def check_gpu() -> Any:
    """Require CUDA and BF16; never silently fall back to CPU or lower precision."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA GPU is required; no CPU fallback is configured")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    if properties.total_memory < 20 * 1024**3:
        raise RuntimeError(
            f"24GB VRAM class GPU required; detected {properties.total_memory / 1024**3:.1f} GiB"
        )
    return device


def download_modelscope_model(model_id: str, root: Path) -> Path:
    """Download a model from ModelScope into the persistent cache."""
    try:
        from modelscope import snapshot_download
    except ImportError:
        from modelscope.hub.snapshot_download import snapshot_download

    kwargs = {"cache_dir": str(root / "cache" / "modelscope" / "models")}
    signature = inspect.signature(snapshot_download)
    first = next(iter(signature.parameters), "model_id")
    if first in {"repo_id", "model_id"}:
        kwargs[first] = model_id
        path = snapshot_download(**kwargs)
    else:
        path = snapshot_download(model_id, **kwargs)
    return Path(path).resolve()


def download_modelscope_dataset(dataset_id: str, root: Path, pattern: str) -> Path:
    """Download only the known parquet files for one ModelScope dataset."""
    try:
        from modelscope import dataset_snapshot_download, snapshot_download
    except ImportError:
        from modelscope.hub.snapshot_download import (
            dataset_snapshot_download,
            snapshot_download,
        )

    cache = root / "cache" / "modelscope" / "datasets"
    try:
        path = dataset_snapshot_download(
            dataset_id=dataset_id,
            cache_dir=str(cache),
            allow_patterns=[pattern],
        )
    except (AttributeError, TypeError):
        signature = inspect.signature(snapshot_download)
        kwargs: dict[str, Any] = {
            "cache_dir": str(cache),
            "repo_type": "dataset",
        }
        if "allow_patterns" in signature.parameters:
            kwargs["allow_patterns"] = [pattern]
        elif "allow_file_pattern" in signature.parameters:
            kwargs["allow_file_pattern"] = [pattern]
        first = next(iter(signature.parameters), "model_id")
        if first in {"repo_id", "model_id"}:
            kwargs[first] = dataset_id
            path = snapshot_download(**kwargs)
        else:
            path = snapshot_download(dataset_id, **kwargs)
    return Path(path).resolve()


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise DataUnavailable(f"Parquet file does not exist: {path}")
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=str(path), split="train")
    return [dict(row) for row in dataset]


def download_geoqa(root: Path) -> list[dict[str, Any]]:
    """Preflight the original HF GeoQA source; mark unavailable, never substitute text data."""
    from huggingface_hub import snapshot_download

    target = root / "cache" / "geoqa"
    target.mkdir(parents=True, exist_ok=True)
    try:
        local_dir = snapshot_download(
            repo_id="hz2475/geoQA",
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=["data/*.parquet"],
        )
        parquet_files = sorted((Path(local_dir) / "data").glob("*.parquet"))
        if not parquet_files:
            raise DataUnavailable("No data/*.parquet files in HF hz2475/geoQA")
        from datasets import load_dataset

        dataset = load_dataset(
            "parquet",
            data_files={"train": [str(path) for path in parquet_files]},
            split="train",
        )
        rows = [
            dict(row)
            for row in dataset
            if row.get("original_split") == "train"
            and row.get("image") is not None
            and row.get("choices") is not None
        ]
        if not rows:
            raise DataUnavailable("GeoQA train rows with image/choices are empty")
        return rows
    except Exception as exc:
        raise DataUnavailable(f"HF GeoQA preflight failed: {type(exc).__name__}: {exc}") from exc


def build_qlora_model(model_dir: Path, rank: int = 16, vision: bool = False) -> tuple[Any, Any, Any]:
    """Load a 4-bit NF4 model, prepare QLoRA, and return model/tokenizer/processor."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Source: https://huggingface.co/docs/bitsandbytes/main/quickstart
    if vision:
        from transformers import Qwen3VLForConditionalGeneration

        model_class = Qwen3VLForConditionalGeneration
        task_type = TaskType.CAUSAL_LM
    else:
        model_class = AutoModelForCausalLM
        task_type = TaskType.CAUSAL_LM
    model = model_class.from_pretrained(
        str(model_dir),
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        bias="none",
        task_type=task_type,
    )
    model = get_peft_model(model, peft_config)
    tokenizer = AutoProcessor.from_pretrained(str(model_dir)) if vision else None
    from transformers import AutoTokenizer

    text_tokenizer = tokenizer if vision else AutoTokenizer.from_pretrained(str(model_dir))
    return model, text_tokenizer, tokenizer


def call_supported(function: Any, **kwargs: Any) -> Any:
    """Drop optional kwargs that the installed library version does not accept."""
    try:
        supported = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return function(**kwargs)
    accepts_extra = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in supported.values()
    )
    filtered = {
        key: value
        for key, value in kwargs.items()
        if key in supported or accepts_extra
    }
    return function(**filtered)


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]], vision: bool = False) -> Any:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if not vision:
        kwargs["enable_thinking"] = False
    result = call_supported(
        tokenizer.apply_chat_template, conversation=messages, **kwargs
    )
    if hasattr(result, "pop"):
        result.pop("token_type_ids", None)
    return result


def move_inputs(inputs: Any, device: Any, torch_module: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in dict(inputs).items():
        if hasattr(value, "to"):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(torch_module.bfloat16)
        output[key] = value
    return output


@dataclass
class Sample:
    input_ids: list[int]
    completion_ids: list[int]
    old_logprobs: list[float]
    advantages: list[float]
    reward: float
    text: str
    extra_inputs: dict[str, Any] | None = None


def generate_group(
    model: Any,
    tokenizer: Any,
    inputs: dict[str, Any],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    torch_module: Any,
    device: Any,
) -> tuple[list[list[int]], list[list[float]], list[str], int]:
    """Sample one GRPO group with the current LoRA policy and record old logprobs."""
    torch_module.manual_seed(seed)
    input_ids = inputs["input_ids"]
    prompt_length = input_ids.shape[1]
    expanded = {
        key: value.repeat(group_size, *([1] * (value.dim() - 1)))
        if hasattr(value, "repeat") and value.shape[0] == 1
        else value
        for key, value in inputs.items()
    }
    with torch_module.no_grad():
        # Include generation scores so the sampling policy's logprobs are retained.
        output = model.generate(
            **expanded,
            do_sample=True,
            num_return_sequences=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        sequences = output.sequences
        scores = output.scores
        eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id is not None else set()
        completions: list[list[int]] = []
        old_logprobs: list[list[float]] = []
        texts: list[str] = []
        for index, sequence in enumerate(sequences):
            tokens = sequence[prompt_length:].tolist()
            valid = common.completion_valid_length(tokens, eos_ids)
            tokens = tokens[:valid]
            completions.append(tokens)
            token_logprobs: list[float] = []
            for position, token in enumerate(tokens):
                if position >= len(scores):
                    break
                logits = scores[position][index].float()
                token_logprobs.append(torch_module.log_softmax(logits, dim=-1)[token].item())
            old_logprobs.append(token_logprobs)
            texts.append(tokenizer.decode(tokens, skip_special_tokens=True))
        valid_lengths = [len(tokens) for tokens in completions]
        del expanded, output, sequences, scores
    torch_module.cuda.empty_cache()
    if min(valid_lengths, default=0) < max(valid_lengths, default=0):
        print(
            "Warning: variable-length group completions; loss uses each sample's valid length",
            flush=True,
        )
    return completions, old_logprobs, texts, prompt_length


def sequence_policy_loss(
    model: Any,
    sample: Sample,
    torch_module: Any,
    device: Any,
    loss_fn: str = "importance_sampling",
) -> Any:
    """Compute one completion's local policy-gradient loss from current logprobs."""
    input_ids = torch_module.tensor(
        [sample.input_ids], device=device, dtype=torch_module.long
    )
    attention_mask = torch_module.ones_like(input_ids)
    kwargs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    if sample.extra_inputs:
        kwargs.update(sample.extra_inputs)
    outputs = model(**kwargs, use_cache=False)
    logits = outputs.logits[0, :-1].float()
    targets = input_ids[0, 1:]
    prompt_length = len(sample.input_ids) - len(sample.completion_ids)
    completion_positions = slice(prompt_length - 1, prompt_length - 1 + len(sample.completion_ids))
    current = torch_module.log_softmax(logits, dim=-1).gather(
        1, targets.unsqueeze(1)
    ).squeeze(1)[completion_positions]
    valid_length = min(len(current), len(sample.old_logprobs), len(sample.advantages))
    if valid_length == 0:
        raise ValueError("A training sample has no valid completion logprobs")
    current = current[:valid_length]
    old = torch_module.tensor(
        sample.old_logprobs[:valid_length], device=device, dtype=torch_module.float32
    )
    advantage = torch_module.tensor(
        sample.advantages[:valid_length], device=device, dtype=torch_module.float32
    )
    ratio = torch_module.exp(current - old)
    if loss_fn == "importance_sampling":
        return -(ratio * advantage).mean()
    if loss_fn == "ppo":
        clip_high = 1.2
        clipped = torch_module.clamp(ratio, 1.0 / clip_high, clip_high)
        return -torch_module.minimum(ratio * advantage, clipped * advantage).mean()
    if loss_fn == "cispo":
        clip_high = 1.2
        clipped = torch_module.clamp(ratio, 1.0 / clip_high, clip_high).detach()
        return -(clipped * current * advantage).mean()
    raise ValueError(f"Unknown loss function: {loss_fn}")


def save_run_config(run_dir: Path, payload: dict[str, Any]) -> None:
    (run_dir / "config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def save_adapter(model: Any, run_dir: Path) -> Path:
    path = run_dir / "adapter"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    return path


def update_training_state(
    root: Path,
    chapter: int,
    run_id: str,
    status: str,
    details: dict[str, Any],
) -> None:
    lib.update_chapter_state(root, chapter, status, run_id, details=details)
    lib.write_report(root)


def chapter_cli(description: str) -> Any:
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workspace-root", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--profile", default=os.environ.get("RUN_PROFILE", "smoke"))
    parser.add_argument("--loss-fn", choices=("importance_sampling", "ppo", "cispo"), default="importance_sampling")
    return parser.parse_args()


def prepare_run(chapter: int, args: Any) -> tuple[Path, Path, str, dict[str, int]]:
    root = lib.resolve_workspace_root(args.workspace_root)
    root.mkdir(parents=True, exist_ok=True)
    if os.environ.get("LOCAL_GPU_SKIP_DOWNLOADS") == "1" and chapter > 1:
        raise RuntimeError("LOCAL_GPU_SKIP_DOWNLOADS=1: diagnostic stop before model/data download")
    run_dir = lib.new_run_dir(root, chapter, args.run_id)
    lib.initialize_state(root)
    lib.init_budget(root)
    lib.write_environment(root, REPO_ROOT, args.profile)
    lib.write_study_notes(root)
    lib.update_chapter_state(root, chapter, "running", run_dir.name)
    config = profile_config(args.profile)
    return root, run_dir, run_dir.name, config
