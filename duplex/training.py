"""Configuration, model loading, and training for the duplex Thinker."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import math
import os
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

from .model import (
    FUSION,
    MODEL_COMPONENT,
    MODEL_ID,
    OUTPUT,
    NextEventLossWeights,
    QwenDuplexThinker,
)
from .instructs2s import DATASET_VIEW as INSTRUCTS2S_VIEW, InstructS2SFirstTurnDataset
from .timeline import ControlTokenIds
from .turn_packed import MusanNoiseDataset, NoiseAugmentationConfig, TurnPackedCollator


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
    "audio_chunk_counts",
    "audio_cache_keys",
    "context_ids",
    "bootstrap_mask",
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
    """Read and validate a training YAML configuration."""

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
        raise ValueError("Only Qwen/Qwen2.5-Omni-3B Thinker is supported.")
    revision = model.get("revision")
    if revision != PINNED_REVISION:
        raise ValueError(f"model.revision must be the pinned commit {PINNED_REVISION}.")
    if model.get("local_files_only") is not True:
        raise ValueError("model.local_files_only must be true for training.")
    if timeline.get("chunk_seconds") != 2.0 or timeline.get("frame_rate_hz") != 25:
        raise ValueError("The timeline requires fixed 2 s chunks at 25 Hz.")
    if timeline.get("control_events") != ["IDLE", "START", "STOP"]:
        raise ValueError("timeline.control_events must be [IDLE, START, STOP].")
    if timeline.get("control_token_source", "talker") not in {
        "talker",
        "thinker_native",
    }:
        raise ValueError("timeline.control_token_source must be talker or thinker_native.")
    expected_task = {
        "fusion": FUSION,
        "objective": "weighted_next_event",
        "output": OUTPUT,
    }
    if any(task.get(key) != value for key, value in expected_task.items()):
        raise ValueError(f"task must match {expected_task}.")
    audio_gain = float(task.get("audio_gain", 1.0))
    if not math.isfinite(audio_gain) or not 0 < audio_gain <= 1:
        raise ValueError("task.audio_gain must be finite and in (0, 1].")
    response_gain = float(task.get("response_audio_gain", 1.0))
    if not math.isfinite(response_gain) or not 0 < response_gain <= 1:
        raise ValueError("task.response_audio_gain must be finite and in (0, 1].")
    if data.get("dataset") != INSTRUCTS2S_VIEW:
        raise ValueError("Only InstructS2SFirstTurn training data is supported.")
    if not isinstance(data.get("root"), str) or not data["root"]:
        raise ValueError("data.root must point to local prepared InstructS2S data.")
    if data.get("synthetic_interruption") is not True:
        raise ValueError("Training data must enable synthetic interruption.")
    if data.get("complete_conversations") is not True:
        raise ValueError("InstructS2S training requires complete conversations.")
    noise = _mapping(data.get("noise", {}), "data.noise")
    NoiseAugmentationConfig(
        probability=float(noise.get("probability", 0.0)),
        min_snr_db=float(noise.get("min_snr_db", 5.0)),
        max_snr_db=float(noise.get("max_snr_db", 20.0)),
    )
    if float(noise.get("probability", 0.0)) > 0 and not noise.get("root"):
        raise ValueError("data.noise.root is required when noise is enabled.")

    method = training.get("method")
    load_in_4bit = model.get("load_in_4bit", False)
    if method not in {"lora", "qlora"}:
        raise ValueError("training.method must be lora or qlora.")
    if (method == "qlora") != (load_in_4bit is True):
        raise ValueError("QLoRA requires explicit model.load_in_4bit: true; LoRA forbids it.")
    if lora.get("rank") != 16:
        raise ValueError("lora.rank must be 16.")
    if lora.get("target_projections") != list(PROJECTION_SUFFIXES):
        raise ValueError(
            "lora.target_projections must contain only q/k/v/o and gate/up/down projections."
        )
    if training.get("bf16") is not True:
        raise ValueError("BF16 is required for LoRA and QLoRA training.")
    if training.get("auto_find_batch_size", False):
        raise ValueError("Automatic OOM fallback is forbidden; select the QLoRA config explicitly.")
    for name in (
        "seed",
        "split_seed",
        "interruption_seed",
        "batch_size",
        "gradient_accumulation_steps",
    ):
        value = training.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"training.{name} must be a non-negative integer.")
    if training.get("num_train_epochs") not in (1, 3):
        raise ValueError("Complete-conversation training requires one or three epochs.")
    if training.get("max_steps") is not None:
        raise ValueError("Complete-conversation training must not set max_steps.")
    if (
        training["batch_size"] < 1
        or training["gradient_accumulation_steps"] < 1
    ):
        raise ValueError(
            "training.batch_size and gradient_accumulation_steps must be positive."
        )
    frame_budget = training.get("max_batch_frames")
    if frame_budget is not None:
        if isinstance(frame_budget, bool) or not isinstance(frame_budget, int) or frame_budget < 1:
            raise ValueError("training.max_batch_frames must be a positive integer.")
        if training["gradient_accumulation_steps"] != 1 or not data.get("lengths_file"):
            raise ValueError("Frame-budget batches require accumulation one and data.lengths_file.")
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
    # Vision is unused here; drop its frozen weights to save memory.
    if hasattr(thinker, "visual"):
        del thinker.visual
        gc.collect()
        torch.cuda.empty_cache()


def build_training_model(config: Mapping[str, Any]) -> tuple[QwenDuplexThinker, object, list[str]]:
    """Load the pinned direct Thinker and inject adapters only into discovered text layers."""

    if not torch.cuda.is_available():
        raise RuntimeError("BF16 LoRA/QLoRA training requires an available CUDA GPU.")
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
        "device_map": {"": f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"},
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
    control_tokens = None
    if config["timeline"].get("control_token_source") == "thinker_native":
        thinker_config = root_config.thinker_config
        control_tokens = ControlTokenIds(
            idle=thinker_config.pad_token_id,
            start=thinker_config.bos_token_id,
            stop=thinker_config.eos_token_id,
            thinker_vocab_size=thinker_config.text_config.vocab_size,
        )
    duplex = QwenDuplexThinker(
        thinker,
        qwen_config=root_config,
        control_tokens=control_tokens,
        loss_weights=NextEventLossWeights(**config["loss_weights"]),
    )
    duplex.processor = processor
    duplex.audio_gain = float(config["task"].get("audio_gain", 1.0))
    duplex.response_audio_gain = float(config["task"].get("response_audio_gain", 1.0))
    # Apply the same FP32/TF32 policy before training and standalone inference.
    torch.backends.cuda.matmul.allow_tf32 = bool(config["training"].get("tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(config["training"].get("tf32", False))
    duplex.base_thinker.audio_tower.float()
    duplex.audio_encoder_fp32 = True
    duplex.pad_audio_to_full_chunks = True
    duplex.requires_empty_context = True
    duplex.default_max_silent_chunks = 16
    duplex.default_max_new_tokens = 768
    assert_only_allowed_lora_trainable(duplex, targets)
    return duplex, processor, targets


def build_datasets_and_collator(
    config: Mapping[str, Any], processor: object, duplex: QwenDuplexThinker
) -> tuple[Dataset, Dataset | None, TurnPackedCollator]:
    """Read prepared InstructS2S splits and apply training augmentation."""
    data = config["data"]
    training = config["training"]
    train_dataset = InstructS2SFirstTurnDataset(data["root"], split="train")
    eval_dataset = (
        InstructS2SFirstTurnDataset(data["root"], split="dev")
        if training.get("eval_strategy", "no") != "no" else None
    )
    if training.get("max_batch_frames") is not None:
        lengths = json.loads(Path(data["lengths_file"]).read_text())
        for split, dataset in (("train", train_dataset), ("dev", eval_dataset)):
            if dataset is None:
                continue
            rows = lengths[split]
            if [row["id"] for row in rows] != dataset.conversation_ids:
                raise ValueError(f"{split} length index differs from the local dataset.")
            dataset.frame_lengths = [row["frames"] for row in rows]
    noise_options = _mapping(data.get("noise", {}), "data.noise")
    noise_config = NoiseAugmentationConfig(
        probability=float(noise_options.get("probability", 0.0)),
        min_snr_db=float(noise_options.get("min_snr_db", 5.0)),
        max_snr_db=float(noise_options.get("max_snr_db", 20.0)),
    )
    noise_dataset = (
        MusanNoiseDataset(noise_options["root"])
        if noise_config.probability > 0.0
        else None
    )
    collator = TurnPackedCollator(
        audio_processor=processor,
        audio_tower=duplex.base_thinker.audio_tower,
        tokenizer=processor.tokenizer,
        control_tokens=duplex.control_tokens,
        thinker_bos_token_id=duplex.base_thinker.config.bos_token_id,
        interruption_probability=float(data.get("interruption_probability", 0.0)),
        min_assistant_frames=int(data.get("min_assistant_frames", 1)),
        noise_dataset=noise_dataset,
        noise_config=noise_config,
        augmentation_seed=int(training["interruption_seed"]),
    )
    return train_dataset, eval_dataset, collator


class GradientAuditCallback(TrainerCallback):
    """Audit gradients before selected optimizer updates (every update by default)."""

    def __init__(self, targets: Sequence[str], *, steps: int = 1) -> None:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("Gradient audit interval must be a positive integer.")
        self.steps = steps
        self.targets = tuple(targets)
        self.nonzero_gradient_names: set[str] = set()
        self.checked_steps = 0

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            raise RuntimeError("Gradient audit received no model.")
        update = state.global_step + 1
        if update != 1 and update != state.max_steps and update % self.steps:
            return
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


class DuplexTrainer(Trainer):
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
        if outputs.loss_weight_sum is None or (
            outputs.loss_weight_sum.item() <= 0 and not inputs.get("_empty_rank", False)
        ):
            raise RuntimeError("Empty weighted batch: no positive-weight supervised events.")
        loss = outputs.loss
        reported_loss = loss.detach()
        if self.accelerator.num_processes > 1:
            from .distributed import normalize_distributed_loss
            loss, reported_loss = normalize_distributed_loss(loss, outputs.loss_weight_sum)
            if not model.training:
                loss = reported_loss
        mode = "train" if model.training else "eval"
        totals = self.metric_totals[mode]
        totals["batches"] += 1
        totals["total_loss"] += float(reported_loss)
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
        return (loss, outputs) if return_outputs else loss

    def _prepare_frame_loader(self, loader):
        if self.accelerator.num_processes == 1:
            return self.accelerator.prepare(loader)
        from accelerate.data_loader import prepare_data_loader
        # The collator has already split each original batch. Do not shard the
        # sampler again: all ranks must retain the original 3,750 updates.
        return prepare_data_loader(
            loader, device=self.accelerator.device, num_processes=1,
            process_index=0, put_on_device=True, rng_types=[],
        )

    def _frame_collator(self):
        if self.accelerator.num_processes == 1:
            return self.data_collator
        from .distributed import SplitGlobalBatchCollator
        return SplitGlobalBatchCollator(
            self.data_collator, self.accelerator.process_index, self.accelerator.num_processes
        )

    def get_train_dataloader(self):
        budget = self.raw_config["training"].get("max_batch_frames")
        if budget is None:
            return super().get_train_dataloader()
        from .batching import FrameBudgetBatchSampler

        sampler = FrameBudgetBatchSampler(
            self.train_dataset.frame_lengths, max_frames=budget,
            max_batch_size=self.args.per_device_train_batch_size,
            seed=self.args.data_seed,
        )
        loader = DataLoader(
            self.train_dataset, batch_sampler=sampler, collate_fn=self._frame_collator(),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor,
        )
        return self._prepare_frame_loader(loader)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        del prediction_loss_only, ignore_keys
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def get_eval_dataloader(self, eval_dataset=None):
        budget = self.raw_config["training"].get("max_batch_frames")
        if budget is None:
            return super().get_eval_dataloader(eval_dataset)
        from .batching import FrameBudgetBatchSampler
        dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        sampler = FrameBudgetBatchSampler(
            dataset.frame_lengths, max_frames=budget,
            max_batch_size=self.args.per_device_eval_batch_size,
            seed=self.args.data_seed,
        )
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=self._frame_collator(),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers,
            prefetch_factor=self.args.dataloader_prefetch_factor)
        return self._prepare_frame_loader(loader)

    def _consume_metrics(self, mode: str) -> dict[str, float]:
        totals = self.metric_totals.pop(mode, None)
        if not totals:
            return {}
        if self.accelerator.num_processes > 1:
            keys = sorted(totals)
            values = torch.tensor([totals[key] for key in keys], dtype=torch.float64,
                                  device=self.accelerator.device)
            torch.distributed.all_reduce(values)
            totals = dict(zip(keys, values.tolist()))
        batches = max(totals["batches"], 1.0)
        valid = max(sum(totals[f"label_count_{group}"] for group in GROUPS), 1.0)
        result = {f"{mode}_total_loss": totals["total_loss"] / batches}
        control_count = sum(totals[f"label_count_{group}"] for group in ("idle", "start", "stop"))
        control_sum = sum(totals[f"loss_sum_{group}"] for group in ("idle", "start", "stop"))
        result[f"{mode}_control_loss"] = control_sum / max(control_count, 1.0)
        result[f"{mode}_lexical_loss"] = totals["loss_sum_text"] / max(totals["label_count_text"], 1.0)
        for group in GROUPS:
            count = totals[f"label_count_{group}"]
            result[f"{mode}_{group}_loss"] = totals[f"loss_sum_{group}"] / max(count, 1.0)
            result[f"{mode}_{group}_label_count"] = totals[f"label_count_{group}"]
            result[f"{mode}_{group}_predicted_count"] = totals[f"predicted_count_{group}"]
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
        "audio_input": {"pad_to_full_chunks": bool(getattr(model, "pad_audio_to_full_chunks", False)),
                        "fusion_gain": getattr(model, "audio_gain", 1.0),
                        "response_fusion_gain": getattr(model, "response_audio_gain", 1.0),
                        "encoder_precision": "float32" if getattr(model, "audio_encoder_fp32", False) else "bfloat16",
                        "context_token_ids": [], "dataset_view": config["data"]["dataset"]},
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
        max_steps=(-1 if training.get("max_steps") is None else training["max_steps"]) if max_steps is None else max_steps,
        num_train_epochs=float(training.get("num_train_epochs", 1)),
        warmup_steps=training.get("warmup_steps", 0),
        weight_decay=float(training.get("weight_decay", 0.0)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        bf16=True,
        tf32=training.get("tf32"),
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
        dataloader_persistent_workers=bool(training.get("dataloader_persistent_workers", False)),
        dataloader_prefetch_factor=training.get("dataloader_prefetch_factor"),
        accelerator_config={"even_batches": False} if training.get("max_batch_frames") else None,
        ddp_find_unused_parameters=False,
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
    collator: TurnPackedCollator,
    *,
    max_steps: int | None = None,
) -> tuple[DuplexTrainer, GradientAuditCallback]:
    audit = GradientAuditCallback(targets, steps=config["training"].get("gradient_audit_steps", 1))
    trainer = DuplexTrainer(
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


def train_from_config(config: Mapping[str, Any]) -> DuplexTrainer:
    """Train the Thinker adapters and save the final checkpoint."""
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    seed_everything(config["training"]["seed"])
    model, processor, targets = build_training_model(config)
    torch.cuda.reset_peak_memory_stats()
    train_dataset, eval_dataset, collator = build_datasets_and_collator(config, processor, model)
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
    trainer.save_state()
    return trainer


def _load_adapter_weights(model: QwenDuplexThinker, checkpoint: Path) -> None:
    metadata_path = checkpoint / "duplex_config.yaml"
    if metadata_path.exists():
        saved_audio = yaml.safe_load(metadata_path.read_text()).get("audio_input", {})
        if saved_audio.get("dataset_view") == INSTRUCTS2S_VIEW:
            expected_audio = {
                "pad_to_full_chunks": bool(getattr(model, "pad_audio_to_full_chunks", False)),
                "fusion_gain": getattr(model, "audio_gain", 1.0),
                "response_fusion_gain": getattr(model, "response_audio_gain", 1.0),
                "encoder_precision": "float32" if getattr(model, "audio_encoder_fp32", False) else "bfloat16",
            }
            defaults = {"fusion_gain": 1.0, "response_fusion_gain": 1.0}
            if (not getattr(model, "requires_empty_context", False)
                    or any(saved_audio.get(key, defaults.get(key)) != value
                           for key, value in expected_audio.items())):
                raise ValueError("Inference audio contract differs from the adapter's training configuration.")
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    weights = load_peft_weights(str(checkpoint), device="cpu", local_files_only=True)
    result = set_peft_model_state_dict(model.thinker, weights, adapter_name="default")
    if getattr(result, "unexpected_keys", None) or getattr(result, "mismatched_keys", None):
        raise RuntimeError(f"Adapter reload failed: {result}")
