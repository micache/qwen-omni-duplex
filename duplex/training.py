"""Compact Session 08 LoRA/QLoRA training support."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from .dataset import (
    DATASET_VIEW,
    ConversationRecord,
    DuplexCollator,
    Speaker,
    Split,
    TimelineSample,
    WindowKind,
    WindowMetadata,
    WordSpan,
    assign_split,
    load_conversation,
    read_manifest,
    sample_window_metadata,
)
from .model import (
    FUSION,
    MODEL_COMPONENT,
    MODEL_ID,
    OUTPUT,
    NextEventLossWeights,
    QwenDuplexThinker,
)
from .timeline import InterruptionConfig


PROJECTION_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
GROUPS = ("text", "idle", "start", "stop")
MODEL_INPUTS = {
    "text_ids",
    "text_mask",
    "control_ids",
    "control_mask",
    "attention_mask",
    "input_features",
    "feature_attention_mask",
    "preconv_feature_lengths",
    "labels",
    "position_ids",
    "past_key_values",
    "use_cache",
    "last_position_only",
}
DEPENDENCIES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "datasets",
    "PyYAML",
    "tensorboard",
)
PINNED_REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a YAML mapping.")
    return value


def _required(mapping: Mapping[str, Any], name: str, path: str) -> Any:
    if name not in mapping:
        raise ValueError(f"Training config is missing {path}.{name}.")
    return mapping[name]


def load_training_config(path: str | Path) -> dict[str, Any]:
    """Read and validate the single Session 08 YAML configuration."""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    config = dict(_mapping(loaded, str(path)))
    for section in ("model", "timeline", "task", "data", "training", "lora", "logging"):
        _mapping(_required(config, section, "config"), section)

    model = config["model"]
    timeline = config["timeline"]
    task = config["task"]
    data = config["data"]
    training = config["training"]
    lora = config["lora"]
    logging = config["logging"]
    if model.get("name") != MODEL_ID or model.get("component") != MODEL_COMPONENT:
        raise ValueError("Session 08 supports only Qwen/Qwen2.5-Omni-3B Thinker.")
    revision = model.get("revision")
    if revision != PINNED_REVISION:
        raise ValueError(f"model.revision must be the pinned commit {PINNED_REVISION}.")
    if model.get("local_files_only") is not True:
        raise ValueError("Session 08 model loading must remain local_files_only.")
    if timeline.get("chunk_seconds") != 2.0 or timeline.get("frame_rate_hz") != 25:
        raise ValueError("Session 08 requires fixed 2 s chunks at 25 Hz.")
    if timeline.get("control_events") != ["IDLE", "START", "STOP"]:
        raise ValueError("timeline.control_events must be [IDLE, START, STOP].")
    expected_task = {
        "fusion": FUSION,
        "objective": "weighted_next_event",
        "output": OUTPUT,
    }
    if any(task.get(key) != value for key, value in expected_task.items()):
        raise ValueError(f"task must retain the Session 08 scope: {expected_task}.")
    if data.get("dataset") != DATASET_VIEW or data.get("synthetic_interruption") is not True:
        raise ValueError("Training data must be DailyTalkContiguous with synthetic interruption.")

    method = training.get("method")
    load_in_4bit = model.get("load_in_4bit", False)
    if method not in {"lora", "qlora"}:
        raise ValueError("training.method must be lora or qlora.")
    if (method == "qlora") != (load_in_4bit is True):
        raise ValueError("QLoRA requires explicit model.load_in_4bit: true; LoRA forbids it.")
    if lora.get("rank") != 16:
        raise ValueError("Session 08 starts at LoRA rank 16.")
    if lora.get("target_projections") != list(PROJECTION_SUFFIXES):
        raise ValueError(
            "lora.target_projections must contain only q/k/v/o and gate/up/down projections."
        )
    if training.get("bf16") is not True:
        raise ValueError("Session 08 training requires BF16 (including QLoRA compute).")
    if training.get("auto_find_batch_size", False):
        raise ValueError("Automatic OOM fallback is forbidden; select the QLoRA config explicitly.")
    for name in (
        "seed",
        "split_seed",
        "window_seed",
        "interruption_seed",
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
    ):
        value = training.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"training.{name} must be a non-negative integer.")
    if (
        training["max_steps"] < 1
        or training["batch_size"] < 1
        or training["gradient_accumulation_steps"] < 1
    ):
        raise ValueError(
            "training.max_steps, batch_size, and gradient_accumulation_steps "
            "must be positive."
        )
    backends = logging.get("backends", [])
    if not isinstance(backends, list) or not backends or not set(backends) <= {
        "jsonl",
        "tensorboard",
    }:
        raise ValueError("logging.backends must select jsonl and/or tensorboard.")
    output_dir = training.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("training.output_dir must be a non-empty path.")
    run_label = training.get("run_label")
    expected_label = "QLoRA" if method == "qlora" else "LoRA"
    if run_label != expected_label:
        raise ValueError(f"training.run_label must explicitly be {expected_label}.")
    return config


def seed_everything(seed: int) -> None:
    """Seed the process RNGs before loading data or constructing adapters."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_text_decoder_lora_targets(thinker: nn.Module) -> list[str]:
    """Return real eligible text-decoder paths, excluding same-named tower layers."""

    layers = getattr(getattr(thinker, "model", None), "layers", None)
    if not isinstance(layers, (nn.ModuleList, list, tuple)) or not layers:
        raise TypeError("Loaded Thinker text decoder has no model.layers sequence.")
    targets: list[str] = []
    for layer_index, layer in enumerate(layers):
        groups = (("self_attn", PROJECTION_SUFFIXES[:4]), ("mlp", PROJECTION_SUFFIXES[4:]))
        for group_name, suffixes in groups:
            group = getattr(layer, group_name, None)
            if group is None:
                continue
            for suffix in suffixes:
                module = getattr(group, suffix, None)
                if module is None:
                    continue
                if not hasattr(module, "weight"):
                    raise TypeError(
                        f"model.layers.{layer_index}.{group_name}.{suffix} has no weight."
                    )
                targets.append(f"model.layers.{layer_index}.{group_name}.{suffix}")
    if not targets:
        raise ValueError("No eligible Thinker text-decoder LoRA projections were discovered.")
    return targets


def _freeze_module(module: object | None) -> None:
    if isinstance(module, nn.Module):
        module.requires_grad_(False)


def freeze_base_and_excluded_modules(thinker: nn.Module) -> None:
    """Freeze the entire base and explicitly freeze every excluded component."""

    thinker.requires_grad_(False)
    for name in (
        "audio_tower",
        "visual",
        "vision_tower",
        "talker",
        "token2wav",
        "waveform_decoder",
    ):
        _freeze_module(getattr(thinker, name, None))


def assert_only_allowed_lora_trainable(
    model: nn.Module, target_module_names: Sequence[str]
) -> tuple[str, ...]:
    """Fail closed if PEFT exposes anything except adapters on discovered targets."""

    trainable = tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    if not trainable:
        raise RuntimeError("No LoRA parameters are trainable.")
    illegal = [
        name
        for name in trainable
        if not (".lora_A." in name or ".lora_B." in name)
        or not any(target in name for target in target_module_names)
    ]
    if illegal:
        raise RuntimeError(f"Trainable parameters outside the LoRA allowlist: {illegal[:8]}")
    missing = [
        target
        for target in target_module_names
        if sum(target in name for name in trainable) != 2
    ]
    if missing:
        raise RuntimeError(f"Discovered targets without exactly one LoRA A/B pair: {missing[:8]}")
    return trainable


def _remove_vision_tower(thinker: nn.Module) -> None:
    # The Session 06 direct Thinker route established that vision is unused and
    # safe to prune after it has been explicitly frozen.
    if hasattr(thinker, "visual"):
        del thinker.visual
        gc.collect()
        torch.cuda.empty_cache()


def build_training_model(config: Mapping[str, Any]) -> tuple[QwenDuplexThinker, object, list[str]]:
    """Load the pinned direct Thinker and inject adapters only into discovered text layers."""

    if not torch.cuda.is_available():
        raise RuntimeError("Session 08 BF16 LoRA/QLoRA training requires an available CUDA GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA GPU does not report BF16 support.")

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        BitsAndBytesConfig,
        Qwen2_5OmniConfig,
        Qwen2_5OmniProcessor,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    model_config = config["model"]
    training = config["training"]
    lora = config["lora"]
    local_only = bool(model_config.get("local_files_only", True))
    common = {
        "revision": model_config["revision"],
        "local_files_only": local_only,
    }
    root_config = Qwen2_5OmniConfig.from_pretrained(model_config["name"], **common)
    processor = Qwen2_5OmniProcessor.from_pretrained(model_config["name"], **common)
    load_kwargs: dict[str, Any] = {
        **common,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": "cuda:0"},
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    if training["method"] == "qlora":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_config["name"], **load_kwargs
    )
    freeze_base_and_excluded_modules(thinker)
    targets = discover_text_decoder_lora_targets(thinker)
    if training["method"] == "qlora":
        thinker = prepare_model_for_kbit_training(
            thinker,
            use_gradient_checkpointing=bool(training["gradient_checkpointing"]),
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    _remove_vision_tower(thinker)
    thinker.config.use_cache = False
    thinker.model.config.use_cache = False
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=targets,
        bias="none",
        modules_to_save=None,
        base_model_name_or_path=model_config["name"],
        revision=model_config["revision"],
    )
    thinker = get_peft_model(thinker, peft_config)
    duplex = QwenDuplexThinker(
        thinker,
        qwen_config=root_config,
        loss_weights=NextEventLossWeights(**config["loss_weights"]),
    )
    assert_only_allowed_lora_trainable(duplex, targets)
    return duplex, processor, targets


class _ListDataset(Dataset):
    def __init__(self, values: Sequence[object]) -> None:
        self.values = tuple(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> object:
        return self.values[index]


def _stable_item_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _synthetic_samples(count: int, *, split: Split, seed: int) -> list[TimelineSample]:
    values: list[TimelineSample] = []
    time = np.arange(32_000, dtype=np.float32) / 16_000
    for index in range(count):
        rng = np.random.default_rng(_stable_item_seed(seed, f"{split.value}-{index}"))
        frequency = 180.0 + 20.0 * index
        waveform = (0.05 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
        waveform += rng.normal(0.0, 0.001, waveform.shape).astype(np.float32)
        conversation_id = f"session08-{split.value}-{index}"
        record = ConversationRecord(
            conversation_id=conversation_id,
            duration_seconds=2.0,
            sample_rate_hz=16_000,
            source_sample_rate_hz=16_000,
            user_waveform=waveform,
            assistant_reference_path=Path("synthetic-not-on-disk.wav"),
            assistant_waveform=None,
            user_words=(WordSpan("interrupt", 1.40, 1.65, Speaker.USER, 1),),
            assistant_words=(WordSpan("hello", 0.28, 0.80, Speaker.ASSISTANT, 0),),
        )
        window = WindowMetadata(
            conversation_id,
            split,
            0.0,
            2.0,
            WindowKind.ASSISTANT_TO_USER,
            boundary_seconds=1.40,
        )
        values.append(TimelineSample(record, window))
    return values


def build_datasets_and_collator(
    config: Mapping[str, Any], processor: object, duplex: QwenDuplexThinker
) -> tuple[Dataset, Dataset | None, DuplexCollator]:
    """Build deterministic DailyTalkContiguous views or the explicit smoke fixture."""

    data = config["data"]
    training = config["training"]
    if data.get("synthetic_smoke", False):
        train_values = _synthetic_samples(4, split=Split.TRAIN, seed=training["window_seed"])
        eval_values = _synthetic_samples(
            2, split=Split.VALIDATION, seed=training["window_seed"]
        )
    else:
        root_value = data.get("root")
        if not isinstance(root_value, str) or not root_value:
            raise ValueError("data.root must point to an existing DailyTalkContiguous snapshot.")
        root = Path(root_value)
        manifest = root / data.get("manifest", "dailytalk.jsonl")
        entries = read_manifest(manifest)
        split_salt = f"DailyTalkContiguous-session08-{training['split_seed']}"
        train_values: list[TimelineSample] = []
        eval_values: list[TimelineSample] = []
        label_map = data.get("speaker_label_map")
        for entry in entries:
            split = assign_split(entry.conversation_id, salt=split_salt)
            if split is Split.TEST:
                continue
            record = load_conversation(
                entry,
                dataset_root=root,
                speaker_label_map=label_map,
            )
            item_seed = _stable_item_seed(training["window_seed"], entry.conversation_id)
            windows = sample_window_metadata(
                record,
                random_count=int(data.get("random_windows_per_conversation", 1)),
                seed=item_seed,
                include_boundaries=bool(data.get("include_boundary_windows", True)),
            )
            destination = train_values if split is Split.TRAIN else eval_values
            destination.extend(TimelineSample(record, replace(window, split=split)) for window in windows)
        max_train = data.get("max_train_samples")
        max_eval = data.get("max_eval_samples")
        if max_train is not None:
            train_values = train_values[: int(max_train)]
        if max_eval is not None:
            eval_values = eval_values[: int(max_eval)]
    if not train_values:
        raise ValueError("The configured training view contains no windows.")
    evaluate = config["training"].get("eval_strategy", "no") != "no"
    if evaluate and not eval_values:
        raise ValueError("Evaluation is enabled but the validation view contains no windows.")

    interruption = InterruptionConfig(
        probability=float(data.get("interruption_probability", 0.0)),
        min_assistant_frames=int(data.get("min_assistant_frames", 1)),
    )
    collator = DuplexCollator(
        audio_processor=processor,
        tokenizer=processor.tokenizer,
        control_tokens=duplex.control_tokens,
        thinker_bos_token_id=duplex.base_thinker.config.bos_token_id,
        frame_rate_hz=25,
        interruption_config=interruption,
        augmentation_seed=training["interruption_seed"],
    )
    return _ListDataset(train_values), (_ListDataset(eval_values) if evaluate else None), collator


class GradientAuditCallback(TrainerCallback):
    """Audit gradients immediately before each optimizer update."""

    def __init__(self, targets: Sequence[str]) -> None:
        self.targets = tuple(targets)
        self.nonzero_gradient_names: set[str] = set()
        self.checked_steps = 0

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            raise RuntimeError("Gradient audit received no model.")
        self.checked_steps += 1
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Non-finite gradient in {name}.")
            if torch.count_nonzero(gradient).item() == 0:
                continue
            if not parameter.requires_grad or not (
                (".lora_A." in name or ".lora_B." in name)
                and any(target in name for target in self.targets)
            ):
                raise RuntimeError(f"Non-LoRA parameter has a nonzero gradient: {name}")
            if ".audio_tower." in name or ".visual." in name:
                raise RuntimeError(f"Frozen tower has a nonzero gradient: {name}")
            self.nonzero_gradient_names.add(name)


class Session08Trainer(Trainer):
    """Trainer that consumes the wrapper loss and emits duplex diagnostics."""

    def __init__(
        self,
        *args,
        processor: object,
        raw_config: Mapping[str, Any],
        target_module_names: Sequence[str],
        **kwargs,
    ) -> None:
        self.processor_for_save = processor
        self.raw_config = dict(raw_config)
        self.target_module_names = tuple(target_module_names)
        self.metric_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.jsonl_path = Path(raw_config["training"]["output_dir"]) / "metrics.jsonl"
        super().__init__(*args, **kwargs)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ):
        del num_items_in_batch
        outputs = model(**{key: value for key, value in inputs.items() if key in MODEL_INPUTS})
        if outputs.loss is None:
            raise RuntimeError("The custom duplex model returned no training loss.")
        if not torch.isfinite(outputs.loss):
            raise FloatingPointError("Non-finite duplex loss.")
        if outputs.loss_weight_sum is None or outputs.loss_weight_sum.item() <= 0:
            raise RuntimeError("Empty weighted batch: no positive-weight supervised events.")
        mode = "train" if model.training else "eval"
        totals = self.metric_totals[mode]
        totals["batches"] += 1
        totals["total_loss"] += float(outputs.loss.detach())
        valid_count = 0
        for group in GROUPS:
            count = int(outputs.target_counts[group].detach())
            predicted = int(outputs.prediction_counts[group].detach())
            totals[f"label_count_{group}"] += count
            totals[f"predicted_count_{group}"] += predicted
            totals[f"loss_sum_{group}"] += float(outputs.group_losses[group].detach()) * count
            valid_count += count
        totals["frames_processed"] += valid_count
        totals["tokens_processed"] += int(outputs.target_counts["text"].detach())
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        del prediction_loss_only, ignore_keys
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def _consume_metrics(self, mode: str) -> dict[str, float]:
        totals = self.metric_totals.pop(mode, None)
        if not totals:
            return {}
        batches = max(totals["batches"], 1.0)
        valid = max(sum(totals[f"label_count_{group}"] for group in GROUPS), 1.0)
        result = {f"{mode}_total_loss": totals["total_loss"] / batches}
        for group in GROUPS:
            count = totals[f"label_count_{group}"]
            result[f"{mode}_{group}_loss"] = totals[f"loss_sum_{group}"] / max(count, 1.0)
            result[f"{mode}_{group}_label_count"] = totals[f"label_count_{group}"]
            result[f"{mode}_{group}_predicted_fraction"] = (
                totals[f"predicted_count_{group}"] / valid
            )
        self.state.num_input_tokens_seen += int(totals["frames_processed"])
        result[f"{mode}_frames_processed"] = totals["frames_processed"]
        result[f"{mode}_tokens_processed"] = totals["tokens_processed"]
        result["frames_processed_total"] = float(self.state.num_input_tokens_seen)
        return result

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "eval" if any(key.startswith("eval_") for key in logs) else "train"
        merged = {**logs, **self._consume_metrics(mode)}
        if torch.cuda.is_available():
            merged["peak_vram_allocated_bytes"] = float(torch.cuda.max_memory_allocated())
            merged["peak_vram_reserved_bytes"] = float(torch.cuda.max_memory_reserved())
        if any(not math.isfinite(float(value)) for value in merged.values()):
            raise FloatingPointError(f"Non-finite logged metric: {merged}")
        super().log(merged, start_time=start_time)
        if "jsonl" in self.raw_config["logging"]["backends"] and self.args.should_save:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                json.dump({"step": self.state.global_step, **merged}, handle, sort_keys=True)
                handle.write("\n")

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        del state_dict
        output = Path(output_dir or self.args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        model = self.accelerator.unwrap_model(self.model)
        assert_only_allowed_lora_trainable(model, self.target_module_names)
        model.thinker.save_pretrained(
            output, safe_serialization=True, save_embedding_layers=False
        )
        self.processor_for_save.save_pretrained(output)
        write_checkpoint_metadata(
            output, self.raw_config, model, self.target_module_names
        )

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

        destination = self.model if model is None else model
        adapter = self.accelerator.unwrap_model(destination).thinker
        weights = load_peft_weights(resume_from_checkpoint, device="cpu", local_files_only=True)
        result = set_peft_model_state_dict(adapter, weights, adapter_name="default")
        if getattr(result, "unexpected_keys", None):
            raise RuntimeError(f"Unexpected adapter keys while resuming: {result.unexpected_keys}")


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in DEPENDENCIES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def write_checkpoint_metadata(
    output: Path,
    config: Mapping[str, Any],
    model: QwenDuplexThinker,
    target_module_names: Sequence[str],
) -> None:
    """Write public reconstruction metadata without serializing base weights."""

    method = config["training"]["method"]
    public_config = {
        "schema_version": 1,
        "adapter_method": "QLoRA" if method == "qlora" else "LoRA",
        "base_model": {
            "id": config["model"]["name"],
            "revision": config["model"]["revision"],
            "component": MODEL_COMPONENT,
        },
        "timeline": {
            "chunk_seconds": model.timeline.chunk_seconds,
            "frame_rate_hz": model.timeline.frame_rate_hz,
            "frames_per_chunk": model.frames_per_chunk,
        },
        "control_ids": {
            "IDLE": model.control_tokens.idle,
            "START": model.control_tokens.start,
            "STOP": model.control_tokens.stop,
        },
        "task": dict(config["task"]),
        "loss_weights": dict(config["loss_weights"]),
        "lora": {
            "rank": config["lora"]["rank"],
            "alpha": config["lora"]["alpha"],
            "dropout": config["lora"]["dropout"],
            "exact_target_modules": list(target_module_names),
        },
    }
    with (output / "duplex_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(public_config, handle, sort_keys=False)
    with (output / "training_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
    with (output / "dependency_versions.json").open("w", encoding="utf-8") as handle:
        json.dump(_versions(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    forbidden = [
        path.name
        for path in output.iterdir()
        if path.name in {"model.safetensors", "pytorch_model.bin"}
        or (path.name.startswith("model-") and path.suffix == ".safetensors")
    ]
    if forbidden:
        raise RuntimeError(f"Refusing checkpoint containing full base weights: {forbidden}")


def make_training_arguments(config: Mapping[str, Any], *, max_steps: int | None = None) -> TrainingArguments:
    training = config["training"]
    logging = config["logging"]
    report_to = [name for name in logging["backends"] if name != "jsonl"]
    if "tensorboard" in report_to:
        try:
            importlib.metadata.version("tensorboard")
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                "TensorBoard logging was requested but tensorboard is not installed."
            ) from error
    return TrainingArguments(
        output_dir=training["output_dir"],
        per_device_train_batch_size=training["batch_size"],
        per_device_eval_batch_size=training.get("eval_batch_size", training["batch_size"]),
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=float(training["learning_rate"]),
        lr_scheduler_type=training.get("lr_scheduler_type", "linear"),
        max_steps=training["max_steps"] if max_steps is None else max_steps,
        warmup_steps=training.get("warmup_steps", 0),
        weight_decay=float(training.get("weight_decay", 0.0)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        bf16=True,
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy=training.get("eval_strategy", "no"),
        eval_steps=training.get("eval_steps"),
        save_strategy=training.get("save_strategy", "steps"),
        save_steps=training.get("save_steps", 100),
        save_total_limit=training.get("save_total_limit", 2),
        logging_strategy="steps",
        logging_steps=logging.get("steps", 1),
        logging_first_step=True,
        logging_nan_inf_filter=False,
        report_to=report_to or "none",
        run_name=training["run_label"],
        seed=training["seed"],
        data_seed=training["split_seed"],
        full_determinism=bool(training.get("full_determinism", False)),
        dataloader_num_workers=int(training.get("dataloader_num_workers", 0)),
        remove_unused_columns=False,
        prediction_loss_only=True,
        auto_find_batch_size=False,
        optim=training.get("optimizer", "adamw_torch"),
        disable_tqdm=bool(logging.get("disable_tqdm", False)),
        skip_memory_metrics=True,
    )


def make_trainer(
    config: Mapping[str, Any],
    model: QwenDuplexThinker,
    processor: object,
    targets: Sequence[str],
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    collator: DuplexCollator,
    *,
    max_steps: int | None = None,
) -> tuple[Session08Trainer, GradientAuditCallback]:
    audit = GradientAuditCallback(targets)
    trainer = Session08Trainer(
        model=model,
        args=make_training_arguments(config, max_steps=max_steps),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[audit],
        processor=processor,
        raw_config=config,
        target_module_names=targets,
    )
    return trainer, audit


def assert_optimizer_scope(trainer: Trainer) -> None:
    """Require optimizer groups/state to reference only trainable parameters."""

    if trainer.optimizer is None:
        raise RuntimeError("Trainer created no optimizer.")
    trainable = {parameter for parameter in trainer.model.parameters() if parameter.requires_grad}
    grouped = {parameter for group in trainer.optimizer.param_groups for parameter in group["params"]}
    state = set(trainer.optimizer.state)
    if grouped != trainable or not state <= trainable:
        raise RuntimeError("Optimizer groups or state include frozen base parameters.")


def train_from_config(config: Mapping[str, Any]) -> Session08Trainer:
    """Run the configured job; callers decide whether it is a smoke or real dataset run."""

    seed_everything(config["training"]["seed"])
    torch.cuda.reset_peak_memory_stats()
    model, processor, targets = build_training_model(config)
    train_dataset, eval_dataset, collator = build_datasets_and_collator(
        config, processor, model
    )
    trainer, audit = make_trainer(
        config, model, processor, targets, train_dataset, eval_dataset, collator
    )
    trainer.train(resume_from_checkpoint=config["training"].get("resume_from_checkpoint"))
    if not audit.nonzero_gradient_names:
        raise RuntimeError("No LoRA parameter received a nonzero gradient.")
    if any(parameter.grad is not None for parameter in model.base_thinker.audio_tower.parameters()):
        raise RuntimeError("Frozen audio tower retained gradients.")
    assert_optimizer_scope(trainer)
    trainer.save_model(Path(config["training"]["output_dir"]) / "final")
    return trainer


def _cpu_model_batch(collator: DuplexCollator, item: object) -> dict[str, torch.Tensor]:
    batch = collator([item])
    return {
        key: value.detach().cpu().clone()
        for key, value in batch.items()
        if key in MODEL_INPUTS and isinstance(value, torch.Tensor)
    }


def _reference_logits(model: nn.Module, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    device = next(parameter.device for parameter in model.parameters())
    inputs = {key: value.to(device) for key, value in batch.items()}
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**inputs).logits
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite logits during adapter reload verification.")
    return logits.detach().cpu()


def _load_adapter_weights(model: QwenDuplexThinker, checkpoint: Path) -> None:
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    weights = load_peft_weights(str(checkpoint), device="cpu", local_files_only=True)
    result = set_peft_model_state_dict(model.thinker, weights, adapter_name="default")
    if getattr(result, "unexpected_keys", None) or getattr(result, "mismatched_keys", None):
        raise RuntimeError(f"Adapter reload failed: {result}")


def _release_cuda_objects(*objects: object) -> None:
    del objects
    gc.collect()
    torch.cuda.empty_cache()


def run_smoke_from_config(config: Mapping[str, Any]) -> Session08Trainer:
    """Run two steps, reload the adapter, then resume for exactly one more step."""

    if not config["data"].get("synthetic_smoke", False):
        raise ValueError("Smoke verification requires data.synthetic_smoke: true.")
    if config["training"]["max_steps"] != 2:
        raise ValueError("The Session 08 smoke config must begin with exactly two steps.")
    seed_everything(config["training"]["seed"])
    torch.cuda.reset_peak_memory_stats()
    model, processor, targets = build_training_model(config)
    train_dataset, eval_dataset, collator = build_datasets_and_collator(
        config, processor, model
    )
    trainer, audit = make_trainer(
        config, model, processor, targets, train_dataset, eval_dataset, collator
    )
    trainer.train()
    if trainer.state.global_step != 2:
        raise RuntimeError(f"Smoke run stopped at step {trainer.state.global_step}, expected 2.")
    if not audit.nonzero_gradient_names or audit.checked_steps != 2:
        raise RuntimeError("Gradient audit did not observe both initial optimizer steps.")
    if any(parameter.grad is not None for parameter in model.base_thinker.audio_tower.parameters()):
        raise RuntimeError("Frozen audio tower retained gradients.")
    assert_optimizer_scope(trainer)

    checkpoint = Path(config["training"]["output_dir"]) / "checkpoint-2"
    if not checkpoint.is_dir():
        raise RuntimeError(f"Expected smoke checkpoint was not saved: {checkpoint}")
    fixed_batch = _cpu_model_batch(collator, train_dataset[0])
    reference = _reference_logits(model, fixed_batch)
    initial_nonzero = sorted(audit.nonzero_gradient_names)

    del trainer, audit, model, processor, train_dataset, eval_dataset, collator
    _release_cuda_objects()

    seed_everything(config["training"]["seed"])
    resumed_model, resumed_processor, resumed_targets = build_training_model(config)
    _load_adapter_weights(resumed_model, checkpoint)
    reloaded = _reference_logits(resumed_model, fixed_batch)
    max_abs_difference = float((reference.float() - reloaded.float()).abs().max())
    torch.testing.assert_close(reloaded, reference, rtol=1e-3, atol=1e-3)

    resumed_train, resumed_eval, resumed_collator = build_datasets_and_collator(
        config, resumed_processor, resumed_model
    )
    resumed_trainer, resumed_audit = make_trainer(
        config,
        resumed_model,
        resumed_processor,
        resumed_targets,
        resumed_train,
        resumed_eval,
        resumed_collator,
        max_steps=3,
    )
    resumed_trainer.train(resume_from_checkpoint=str(checkpoint))
    if resumed_trainer.state.global_step != 3 or resumed_audit.checked_steps != 1:
        raise RuntimeError(
            "Resume did not advance checkpoint-2 by exactly one optimizer step: "
            f"global_step={resumed_trainer.state.global_step}, "
            f"optimizer_updates={resumed_audit.checked_steps}."
        )
    if not resumed_audit.nonzero_gradient_names:
        raise RuntimeError("No LoRA parameter received a nonzero gradient after resume.")
    if any(
        parameter.grad is not None
        for parameter in resumed_model.base_thinker.audio_tower.parameters()
    ):
        raise RuntimeError("Frozen audio tower retained gradients after resume.")
    assert_optimizer_scope(resumed_trainer)
    final = Path(config["training"]["output_dir"]) / "final"
    resumed_trainer.save_model(final)

    report = {
        "status": "passed",
        "training_mode": config["training"]["run_label"],
        "initial_optimizer_steps": 2,
        "resumed_from_step": 2,
        "final_optimizer_step": 3,
        "discovered_target_count": len(resumed_targets),
        "trainable_lora_parameter_count": sum(
            parameter.numel()
            for parameter in resumed_model.parameters()
            if parameter.requires_grad
        ),
        "nonzero_lora_gradient_parameter_count": len(
            set(initial_nonzero) | resumed_audit.nonzero_gradient_names
        ),
        "audio_tower_gradient_parameter_count": 0,
        "optimizer_scope": "trainable_lora_parameters_only",
        "adapter_reload_max_abs_logit_difference": max_abs_difference,
        "adapter_reload_rtol": 1e-3,
        "adapter_reload_atol": 1e-3,
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    with (Path(config["training"]["output_dir"]) / "smoke_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return resumed_trainer
