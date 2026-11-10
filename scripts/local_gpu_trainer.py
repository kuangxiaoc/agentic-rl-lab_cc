"""Shared local GPU primitives for the local chapter scripts."""

from __future__ import annotations

import inspect
import json
import os
import sys
import types
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


def hub_downloaders() -> tuple[Any, Any]:
    """Return (model, dataset) snapshot helpers from any recent modelscope.

    ``modelscope.hub.snapshot_download`` is a module, but the package ``__init__``
    rebinds that attribute to the function inside it, so ``from modelscope.hub
    import snapshot_download`` yields a function with no ``snapshot_download``
    attribute of its own. Import the module by name instead.
    """
    import importlib

    module = importlib.import_module("modelscope.hub.snapshot_download")
    return module.snapshot_download, getattr(module, "dataset_snapshot_download", None)


def download_modelscope_model(model_id: str, root: Path) -> Path:
    """Download a model from ModelScope into the persistent cache."""
    function, _ = hub_downloaders()
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

    The dataset helper has moved between modelscope releases, so look both helpers
    up by attribute on the hub module rather than trusting one import path.
    """
    generic_fn, dataset_fn = hub_downloaders()
    cache = str(root / "cache" / "modelscope" / "datasets")
    attempts: list[tuple[Any, dict[str, Any]]] = [(dataset_fn, {"dataset_id": dataset_id})]
    first = next(iter(inspect.signature(generic_fn).parameters), "model_id")
    attempts.append((generic_fn, {first: dataset_id, "repo_type": "dataset"}))

    errors: list[str] = []
    for function, id_kwargs in attempts:
        if function is None:
            continue
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


def load_parquet_rows(
    path: Path,
    *,
    limit: int | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise DataUnavailable(f"Parquet file does not exist: {path}")
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files=str(path), split="train")
    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    return [dict(row) for row in dataset]


def download_geoqa(root: Path) -> list[dict[str, Any]]:
    """Preflight the original HF GeoQA source; mark unavailable, never substitute text data.

    huggingface.co is unreachable from mainland GPU boxes; setting HF_ENDPOINT to a
    mirror (for example https://hf-mirror.com) routes both the hub call and the
    subsequent datasets download through it.
    """
    endpoint = os.environ.get("HF_ENDPOINT")
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        print(f"GeoQA: using HF endpoint {endpoint}", flush=True)
    else:
        print("GeoQA: HF_ENDPOINT is unset; trying huggingface.co directly", flush=True)
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


def build_quantized_model(model_dir: Path) -> Any:
    """Load a frozen 4-bit model for local teacher / evaluation inference."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        quantization_config=quantization,
        device_map={"": 0},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def load_text_tokenizer(model_dir: Path) -> Any:
    """Load only the text tokenizer, avoiding a processor for text-only chapters."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir))


def build_qlora_from_adapter(model_dir: Path, adapter_dir: Path) -> tuple[Any, Any, None]:
    """Resume a local QLoRA student from an adapter written by ``save_adapter``."""
    from peft import PeftModel, prepare_model_for_kbit_training

    model = build_quantized_model(model_dir)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    model.train()
    return model, load_text_tokenizer(model_dir), None


def attach_adapter(model: Any, adapter_dir: Path) -> Any:
    """Attach a frozen LoRA adapter to an already quantized model."""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)


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


def text_tokenizer(tokenizer: Any) -> Any:
    """Return whatever exposes token ids: a tokenizer, or a processor's inner one.

    ``Qwen3VLProcessor`` wraps the tokenizer and has no ``eos_token_id`` of its own,
    so callers must unwrap it before reading ids or decoding.
    """
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return tokenizer
    inner = getattr(tokenizer, "tokenizer", None)
    return inner if inner is not None else tokenizer


def pad_token_id(tokenizer: Any) -> int | None:
    """Qwen chat models have no real pad token, so stop generation on EOS.

    A vision processor may not expose token ids at all; returning None then lets
    Transformers fall back to the model's own generation config.
    """
    tokenizer = text_tokenizer(tokenizer)
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


@dataclass
class GeneratedSequence:
    """One completion returned by a local sampling client."""

    tokens: list[int]
    logprobs: list[float]
    text: str


@dataclass
class SampleResponse:
    """Minimal response shape shared by the original PyTRIO rollout code."""

    sequences: list[GeneratedSequence]


def _sampling_scores_to_logprobs(
    scores: Any,
    sequence_index: int,
    tokens: list[int],
    torch_module: Any,
) -> list[float]:
    """Read the sampling policy's logprob for each generated token."""
    logprobs: list[float] = []
    for position, token in enumerate(tokens):
        if position >= len(scores):
            break
        logits = scores[position][sequence_index].float()
        logprobs.append(
            torch_module.log_softmax(logits, dim=-1)[token].item()
        )
    return logprobs


class LocalSamplingClient:
    """Single-GPU adapter for the subset of the PyTRIO sampling API we reuse.

    The original chapters call ``sample_async`` with token prompts, stop
    strings, and per-call sampling parameters. Implementing that small surface
    lets the Search-R1 and ReTool rollout state machines stay unchanged while
    sampling happens on the local model.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        torch_module: Any,
        device: Any,
        *,
        task_id: str = "local-gpu",
    ) -> None:
        self.model = model
        self.tokenizer = text_tokenizer(tokenizer)
        self.torch = torch_module
        self.device = device
        self.task_id = task_id

    def sample(
        self,
        *,
        prompt_tokens: list[int],
        num_samples: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> SampleResponse:
        """Sample one prompt without gradients and retain sampling logprobs."""
        if num_samples < 1 or max_tokens < 1:
            raise ValueError("num_samples and max_tokens must be >= 1")
        self.torch.manual_seed(seed)
        input_ids = self.torch.tensor(
            [prompt_tokens], device=self.device, dtype=self.torch.long
        )
        attention_mask = self.torch.ones_like(input_ids)
        was_training = self.model.training
        self.model.eval()
        try:
            with self.torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    num_return_sequences=num_samples,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_token_id(self.tokenizer),
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                sequences: list[GeneratedSequence] = []
                eos_ids = (
                    {self.tokenizer.eos_token_id}
                    if self.tokenizer.eos_token_id is not None
                    else set()
                )
                for index, sequence in enumerate(output.sequences):
                    tokens = sequence[len(prompt_tokens) :].tolist()
                    valid = common.completion_valid_length(tokens, eos_ids)
                    tokens = tokens[:valid]
                    sequences.append(
                        GeneratedSequence(
                            tokens=tokens,
                            logprobs=_sampling_scores_to_logprobs(
                                output.scores, index, tokens, self.torch
                            ),
                            text=self.tokenizer.decode(
                                tokens, skip_special_tokens=True
                            ),
                        )
                    )
                del output
        finally:
            if was_training:
                self.model.train()
            self.torch.cuda.empty_cache()
        return SampleResponse(sequences)

    async def sample_async(
        self,
        *,
        prompt: Any,
        num_samples: int,
        sampling_params: Any,
        return_text: bool = True,
    ) -> SampleResponse:
        """Accept the PyTRIO request shape and run one synchronous GPU sample."""
        del return_text
        prompt_tokens = [int(token) for token in prompt.tokens]
        return self.sample(
            prompt_tokens=prompt_tokens,
            num_samples=num_samples,
            max_tokens=int(sampling_params.max_tokens),
            temperature=float(getattr(sampling_params, "temperature", 1.0)),
            top_p=float(getattr(sampling_params, "top_p", 1.0)),
            seed=int(getattr(sampling_params, "seed", 0)),
        )


class LocalTokenPrompt:
    """Token-only prompt matching ``trio.ModelInput.from_ints``."""

    def __init__(self, tokens: list[int]) -> None:
        self.tokens = list(tokens)

    @classmethod
    def from_ints(cls, tokens: list[int]) -> "LocalTokenPrompt":
        return cls(tokens)


@dataclass(frozen=True)
class LocalSamplingParams:
    """The sampling fields the reused rollout code reads."""

    max_tokens: int
    seed: int = 0
    stop: list[str] | tuple[str, ...] = ()
    temperature: float = 1.0
    top_p: float = 1.0


def local_trio_compat() -> types.SimpleNamespace:
    """Return the tiny PyTRIO surface used by multi-turn rollout modules."""
    return types.SimpleNamespace(
        ModelInput=LocalTokenPrompt,
        SamplingParams=LocalSamplingParams,
    )


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
    text_tok = text_tokenizer(tokenizer)
    # Gradient checkpointing is incompatible with the KV cache, so generate() would
    # drop past_key_values every step and sample from a single-token context, which
    # decodes as gibberish. Sample in eval mode, then hand the model back in train mode.
    was_training = model.training
    model.eval()
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
        eos_ids = {text_tok.eos_token_id} if text_tok.eos_token_id is not None else set()
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
            texts.append(text_tok.decode(tokens, skip_special_tokens=True))
        valid_lengths = [len(tokens) for tokens in completions]
        del output, sequences, scores
    if was_training:
        model.train()
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
    clip_low: float = 1.0 / 1.2,
    clip_high: float = 1.2,
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
        clipped = torch_module.clamp(ratio, clip_low, clip_high)
        return -torch_module.minimum(ratio * advantage, clipped * advantage).mean()
    if loss_fn == "cispo":
        clipped = torch_module.clamp(ratio, clip_low, clip_high).detach()
        return -(clipped * current * advantage).mean()
    raise ValueError(f"Unknown loss function: {loss_fn}")


def sequence_policy_loss_gspo(
    model: Any,
    sample: Sample,
    torch_module: Any,
    device: Any,
    clip_low: float = 3e-4,
    clip_high: float = 4e-4,
) -> Any:
    """GSPO: one ratio per answer, from the geometric mean of its token ratios.

    Chapter 02/07 compute a per-token ratio and clip each token. GSPO instead
    averages the log-ratio over the completion, so every answer contributes a
    single clipped objective and length no longer changes the ratio's scale.
    """
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
    # One advantage per answer; GSPO scores whole sequences, not tokens.
    advantage = float(sample.advantages[0])
    log_ratio = (current - old).mean()
    ratio = torch_module.exp(log_ratio)
    clipped = torch_module.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)
    objective = torch_module.minimum(ratio * advantage, clipped * advantage)
    return -objective


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
    *,
    replace_details: bool = False,
) -> None:
    lib.update_chapter_state(
        root,
        chapter,
        status,
        run_id,
        details=details,
        replace_details=replace_details,
    )
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
