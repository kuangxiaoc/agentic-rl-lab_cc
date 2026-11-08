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
    from modelscope.hub import snapshot_download as hub_module

    function = hub_module.snapshot_download
    first = next(iter(inspect.signature(function).parameters), "model_id")
    cache_dir = str(root / "cache" / "modelscope" / "models")
    if first in {"repo_id", "model_id"}:
        path = function(**supported_kwargs(function, cache_dir=cache_dir, **{first: model_id}))
    else:
        path = function(model_id, **supported_kwargs(function, cache_dir=cache_dir))
    return Path(path).resolve()


def supported_kwargs(function: Any, **kwargs: Any) -> dict[str, Any]:
    """Keep only the keyword arguments the installed library version declares."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def download_modelscope_dataset(dataset_id: str, root: Path, pattern: str) -> Path:
    """Download only the known parquet files for one ModelScope dataset.

    modelscope exposes the dataset helper in ``modelscope.hub.snapshot_download``;
    the top-level re-export has moved between versions, so look it up by attribute
    instead of relying on one import path plus an exception-type fallback.
    """
    from modelscope.hub import snapshot_download as hub_module

    cache = str(root / "cache" / "modelscope" / "datasets")
    attempts: list[tuple[Any, dict[str, Any]]] = []
    dataset_fn = getattr(hub_module, "dataset_snapshot_download", None)
    if callable(dataset_fn):
        attempts.append((dataset_fn, {"dataset_id": dataset_id}))
    generic_fn = getattr(hub_module, "snapshot_download", None)
    if callable(generic_fn):
        first = next(iter(inspect.signature(generic_fn).parameters), "model_id")
        attempts.append((generic_fn, {first: dataset_id, "repo_type": "dataset"}))

    errors: list[str] = []
    for function, id_kwargs in attempts:
        kwargs = supported_kwargs(
            function, cache_dir=cache, allow_patterns=[pattern], **id_kwargs
        )
        try:
            path = function(**kwargs)
        except Exception as exc:  # noqa: BLE001 - report every hub attempt
            errors.append(f"{function.__name__}{sorted(kwargs)}: {type(exc).__name__}: {exc}")
            continue
        if path:
            return Path(path).resolve()
        errors.append(f"{function.__name__} returned no local path")
    raise DataUnavailable(
        "Could not download dataset "
        + dataset_id
        + ("; attempts: " + " | ".join(errors) if errors else "")
    )


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


def chat_template_mentions(tokenizer: Any, variable: str) -> bool:
    """Check whether the model's own Jinja template reads one generation variable."""
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        return any(variable in (value or "") for value in template.values() if isinstance(value, str))
    return isinstance(template, str) and variable in template


def apply_chat_template(tokenizer: Any, messages: list[dict[str, Any]], vision: bool = False) -> Any:
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    # Passing the conversation positionally works for both the old `conversation=`
    # and the current `messages=` parameter name, so the rename cannot break 02/03.
    # `enable_thinking` only exists in thinking-capable templates, so it is passed
    # only when the model's own template reads it; Qwen3-*-Instruct has no such flag.
    if not vision and chat_template_mentions(tokenizer, "enable_thinking"):
        kwargs["enable_thinking"] = False
    result = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(result, "pop"):
        result.pop("token_type_ids", None)
    return result


def pad_token_id(tokenizer: Any) -> int | None:
    """Qwen chat models have no real pad token, so stop generation on EOS.

    A vision processor may not expose token ids at all; returning None then lets
    Transformers fall back to the model's own generation config.
    """
    for name in ("pad_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, int):
            return value
    return None


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
    with torch_module.no_grad():
        # Include generation scores so the sampling policy's logprobs are retained.
        # `num_return_sequences` expands the batch-1 prompt inside generate(), which
        # also repeats multimodal tensors; repeating them here first would sample
        # group_size ** 2 completions for one GRPO group.
        output = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id(tokenizer),
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
        del output, sequences, scores
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
