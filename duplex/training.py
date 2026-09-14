"""Compact Session 08 LoRA/QLoRA training support."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
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
from .timeline import (
    ControlTokenIds,
    EventKind,
    InterruptionConfig,
    WindowTimeline,
    augment_synthetic_interruption,
    build_window_timeline,
    format_timeline_table,
)


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
        raise ValueError(f"task must retain the Session 08 scope: {expected_task}.")
    if data.get("dataset") != DATASET_VIEW or data.get("synthetic_interruption") is not True:
        raise ValueError("Training data must be DailyTalkContiguous with synthetic interruption.")

    selected_windows = data.get("selected_windows")
    if selected_windows is not None:
        if data.get("synthetic_smoke", False):
            raise ValueError("selected_windows cannot be combined with synthetic_smoke.")
        if not isinstance(selected_windows, list) or not 2 <= len(selected_windows) <= 8:
            raise ValueError("data.selected_windows must contain 2-8 window mappings.")
        seen_windows: set[tuple[str, float, float]] = set()
        interrupted = 0
        for index, raw_window in enumerate(selected_windows):
            window = _mapping(raw_window, f"data.selected_windows[{index}]")
            conversation_id = window.get("conversation_id")
            if not isinstance(conversation_id, str) or not conversation_id:
                raise ValueError(f"selected window {index} has no public conversation_id.")
            start = window.get("start_seconds")
            end = window.get("end_seconds")
            if (
                isinstance(start, bool)
                or not isinstance(start, (int, float))
                or isinstance(end, bool)
                or not isinstance(end, (int, float))
                or not math.isclose(float(end) - float(start), 2.0, abs_tol=1e-9)
            ):
                raise ValueError(f"selected window {index} must span exactly 2 seconds.")
            if window.get("kind") not in {kind.value for kind in WindowKind}:
                raise ValueError(f"selected window {index} has an invalid kind.")
            word_indices = window.get("assistant_word_indices")
            if (
                not isinstance(word_indices, list)
                or not word_indices
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in word_indices
                )
                or word_indices != sorted(set(word_indices))
            ):
                raise ValueError(
                    f"selected window {index} needs unique sorted assistant_word_indices."
                )
            key = (conversation_id, float(start), float(end))
            if key in seen_windows:
                raise ValueError(f"duplicate selected window metadata: {key}.")
            seen_windows.add(key)
            synthetic_audio = window.get("synthetic_user_audio")
            if synthetic_audio is not None:
                synthetic_audio = _mapping(
                    synthetic_audio,
                    f"data.selected_windows[{index}].synthetic_user_audio",
                )
                source_start = synthetic_audio.get("start_seconds")
                source_end = synthetic_audio.get("end_seconds")
                if (
                    isinstance(source_start, bool)
                    or not isinstance(source_start, (int, float))
                    or isinstance(source_end, bool)
                    or not isinstance(source_end, (int, float))
                    or not float(start) <= float(source_start) < float(source_end) <= float(end)
                ):
                    raise ValueError(
                        f"selected window {index} has invalid synthetic user-audio metadata."
                    )
                interrupted += 1
        if interrupted < 1:
            raise ValueError("At least one selected window must request synthetic interruption.")

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
    if selected_windows is not None and not 100 <= training["max_steps"] <= 300:
        raise ValueError("The Session 09 overfit budget must be 100-300 optimizer steps.")
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
    selected_windows = data.get("selected_windows")
    prebuilt_interruption = selected_windows is not None
    if selected_windows is not None:
        root = Path(data["root"])
        entries = {
            entry.conversation_id: entry
            for entry in read_manifest(root / data.get("manifest", "dailytalk.jsonl"))
        }
        split_salt = f"DailyTalkContiguous-session09-{training['split_seed']}"
        train_values = []
        eval_values = []
        for index, selected in enumerate(selected_windows):
            conversation_id = selected["conversation_id"]
            if conversation_id not in entries:
                raise ValueError(
                    f"selected window {index} references unknown public ID "
                    f"{conversation_id!r}."
                )
            split = assign_split(conversation_id, salt=split_salt)
            if split is not Split.TRAIN:
                raise ValueError(
                    f"selected window {conversation_id!r} is assigned to {split.value}, "
                    "not train."
                )
            record = load_conversation(entries[conversation_id], dataset_root=root)
            requested_indices = tuple(selected["assistant_word_indices"])
            assistant_words = tuple(
                word
                for word in record.assistant_words
                if word.source_index in requested_indices
            )
            found_indices = tuple(word.source_index for word in assistant_words)
            if found_indices != requested_indices:
                raise ValueError(
                    f"selected window {conversation_id!r} requested assistant words "
                    f"{requested_indices}, found {found_indices}."
                )
            synthetic_audio = selected.get("synthetic_user_audio")
            user_words = ()
            if synthetic_audio is not None:
                user_words = (
                    WordSpan(
                        "<synthetic-user-audio>",
                        float(synthetic_audio["start_seconds"]),
                        float(synthetic_audio["end_seconds"]),
                        Speaker.USER,
                    ),
                )
            record = replace(
                record,
                assistant_words=assistant_words,
                user_words=user_words,
            )
            window = WindowMetadata(
                conversation_id=conversation_id,
                split=split,
                start_seconds=float(selected["start_seconds"]),
                end_seconds=float(selected["end_seconds"]),
                kind=WindowKind(selected["kind"]),
            )
            timeline = build_window_timeline(
                record,
                window,
                tokenizer=processor.tokenizer,
                control_tokens=duplex.control_tokens,
                thinker_bos_token_id=duplex.base_thinker.config.bos_token_id,
                frame_rate_hz=25,
            )
            if synthetic_audio is not None:
                timeline = augment_synthetic_interruption(
                    timeline,
                    config=InterruptionConfig(
                        probability=1.0,
                        min_assistant_frames=int(data.get("min_assistant_frames", 1)),
                    ),
                    rng=random.Random(
                        _stable_item_seed(
                            training["interruption_seed"], timeline.sample_id
                        )
                    ),
                    control_tokens=duplex.control_tokens,
                    thinker_bos_token_id=duplex.base_thinker.config.bos_token_id,
                )
                if timeline.interruption is None:
                    raise ValueError(
                        f"selected window {timeline.sample_id!r} was not eligible for "
                        "the requested deterministic interruption."
                    )
            train_values.append(timeline)
    elif data.get("synthetic_smoke", False):
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
        probability=(
            0.0
            if prebuilt_interruption
            else float(data.get("interruption_probability", 0.0))
        ),
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


def _timeline_values(dataset: Dataset) -> tuple[WindowTimeline, ...]:
    values = getattr(dataset, "values", None)
    if values is None or not all(isinstance(value, WindowTimeline) for value in values):
        raise TypeError("Session 09 requires a prebuilt fixed WindowTimeline dataset.")
    return tuple(values)


def _cuda_batch(batch: Mapping[str, Any], model: nn.Module) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in MODEL_INPUTS and isinstance(value, torch.Tensor)
    }


def _prediction_kind(token_id: int, model: QwenDuplexThinker) -> str:
    if token_id == model.control_tokens.idle:
        return EventKind.IDLE.value
    if token_id == model.control_tokens.start:
        return EventKind.START.value
    if token_id == model.control_tokens.stop:
        return EventKind.STOP.value
    return EventKind.TEXT.value


def _sequence_summary(
    prediction_ids: Sequence[int],
    timeline: WindowTimeline,
    model: QwenDuplexThinker,
    tokenizer: object,
) -> dict[str, Any]:
    target_ids = timeline.causal.labels
    target_nonidle_frames = [
        index
        for index, event in enumerate(timeline.targets.events)
        if event.kind is not EventKind.IDLE
    ]
    predicted_at_target_frames = [prediction_ids[index] for index in target_nonidle_frames]
    expected_at_target_frames = [target_ids[index] for index in target_nonidle_frames]
    lexical_frames = [
        index
        for index, event in enumerate(timeline.targets.events)
        if event.kind is EventKind.TEXT
    ]
    predicted_lexical_ids = [prediction_ids[index] for index in lexical_frames]
    target_lexical_ids = [target_ids[index] for index in lexical_frames]
    return {
        "sample_id": timeline.sample_id,
        "target_event_kinds": [event.kind.value for event in timeline.targets.events],
        "predicted_event_kinds": [
            _prediction_kind(token_id, model) for token_id in prediction_ids
        ],
        "target_nonidle_frames": target_nonidle_frames,
        "expected_at_target_frames": expected_at_target_frames,
        "predicted_at_target_frames": predicted_at_target_frames,
        "target_lexical_ids": target_lexical_ids,
        "predicted_lexical_ids": predicted_lexical_ids,
        "target_text": tokenizer.decode(
            target_lexical_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        ),
        "predicted_text_at_target_frames": tokenizer.decode(
            predicted_lexical_ids,
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        ),
        "start_text_stop_exact_at_target_frames": (
            predicted_at_target_frames == expected_at_target_frames
        ),
        "prediction_ids": list(prediction_ids),
    }


def _teacher_forced_report(
    model: QwenDuplexThinker,
    dataset: Dataset,
    collator: DuplexCollator,
    tokenizer: object,
) -> dict[str, Any]:
    timelines = _timeline_values(dataset)
    group_loss_sums = {group: 0.0 for group in GROUPS}
    target_counts = {group: 0 for group in GROUPS}
    prediction_counts = {group: 0 for group in GROUPS}
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    examples = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for timeline in timelines:
            batch = _cuda_batch(collator([timeline]), model)
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)[0].tolist()
            examples.append(_sequence_summary(predictions, timeline, model, tokenizer))
            current_weight = float(outputs.loss_weight_sum)
            weighted_loss_sum += float(outputs.loss) * current_weight
            weight_sum += current_weight
            for group in GROUPS:
                count = int(outputs.target_counts[group])
                target_counts[group] += count
                prediction_counts[group] += int(outputs.prediction_counts[group])
                group_loss_sums[group] += float(outputs.group_losses[group]) * count
    valid = sum(target_counts.values())
    return {
        "total_loss": weighted_loss_sum / max(weight_sum, 1.0),
        "group_losses": {
            group: group_loss_sums[group] / max(target_counts[group], 1)
            for group in GROUPS
        },
        "target_counts": target_counts,
        "predicted_fractions": {
            group: prediction_counts[group] / max(valid, 1) for group in GROUPS
        },
        "examples": examples,
    }


def _cached_free_prediction(
    model: QwenDuplexThinker,
    batch: Mapping[str, torch.Tensor],
) -> list[int]:
    attention = batch["attention_mask"].bool()
    if attention.shape[0] != 1 or not bool(attention.all()):
        raise ValueError("Cached Session 09 generation requires one unpadded timeline.")
    timeline_length = attention.shape[1]
    embedding = model.base_thinker.get_input_embeddings()
    audio_embeddings = model._restore_audio(
        input_features=batch["input_features"],
        feature_attention_mask=batch["feature_attention_mask"],
        preconv_feature_lengths=batch["preconv_feature_lengths"],
        current_attention=attention,
        timeline_length=timeline_length,
        hidden_width=embedding.weight.shape[1],
        device=embedding.weight.device,
        dtype=embedding.weight.dtype,
    )
    bos = model.base_thinker.config.bos_token_id
    controls = {
        model.control_tokens.idle,
        model.control_tokens.start,
        model.control_tokens.stop,
    }
    predictions: list[int] = []
    cache = None
    for frame in range(timeline_length):
        previous = bos if frame == 0 else predictions[-1]
        token = torch.tensor([[previous]], dtype=torch.long, device=embedding.weight.device)
        token_embedding = embedding(token)
        if frame > 0 and previous in controls:
            text_embedding = torch.zeros_like(token_embedding)
            control_embedding = token_embedding
        else:
            text_embedding = token_embedding
            control_embedding = torch.zeros_like(token_embedding)
        fused = text_embedding + control_embedding + audio_embeddings[:, frame : frame + 1]
        output = model.base_thinker.model(
            inputs_embeds=fused,
            attention_mask=torch.ones(
                (1, frame + 1), dtype=torch.bool, device=embedding.weight.device
            ),
            position_ids=torch.tensor([[frame]], device=embedding.weight.device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        cache = output.past_key_values
        token_id = int(
            model.base_thinker.lm_head(output.last_hidden_state[:, -1:]).argmax().item()
        )
        predictions.append(token_id)
    return predictions


def _cached_free_report(
    model: QwenDuplexThinker,
    dataset: Dataset,
    collator: DuplexCollator,
    tokenizer: object,
) -> dict[str, Any]:
    model.eval()
    model.gradient_checkpointing_disable()
    examples = []
    counts = Counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for timeline in _timeline_values(dataset):
            batch = _cuda_batch(collator([timeline]), model)
            predictions = _cached_free_prediction(model, batch)
            summary = _sequence_summary(predictions, timeline, model, tokenizer)
            examples.append(summary)
            counts.update(kind.lower() for kind in summary["predicted_event_kinds"])
    total = sum(counts.values())
    return {
        "predicted_fractions": {
            group: counts[group] / max(total, 1) for group in GROUPS
        },
        "examples": examples,
    }


def _write_session09_preflight(
    output_dir: Path,
    dataset: Dataset,
) -> dict[str, Any]:
    timelines = _timeline_values(dataset)
    for timeline in timelines:
        causal = timeline.causal
        if not causal.text_mask[0] or causal.control_mask[0]:
            raise AssertionError(f"{timeline.sample_id}: frame zero is not BOS-only.")
        for frame in range(1, len(timeline.targets.events)):
            previous = timeline.targets.events[frame - 1]
            if previous.kind is EventKind.TEXT:
                if (
                    not causal.text_mask[frame]
                    or causal.control_mask[frame]
                    or causal.text_ids[frame] != previous.text_id
                ):
                    raise AssertionError(
                        f"{timeline.sample_id}: lexical causal shift failed at frame {frame}."
                    )
            elif (
                causal.text_mask[frame]
                or not causal.control_mask[frame]
                or causal.control_ids[frame] != causal.labels[frame - 1]
            ):
                raise AssertionError(
                    f"{timeline.sample_id}: control causal shift failed at frame {frame}."
                )
    histogram = Counter(
        event.kind.value for timeline in timelines for event in timeline.targets.events
    )
    histogram[EventKind.PADDING.value] += 0
    interrupted = [timeline for timeline in timelines if timeline.interruption is not None]
    if not interrupted:
        raise RuntimeError("Session 09 preflight found no deterministic interruption.")
    report = {
        "label_histogram": dict(sorted(histogram.items())),
        "causal_shift_check": "passed",
        "decode_round_trip_check": "passed",
        "windows": [
            {
                "sample_id": timeline.sample_id,
                "interruption": (
                    None
                    if timeline.interruption is None
                    else {
                        "cut_frame": timeline.interruption.cut_frame,
                        "original_stop_frame": timeline.interruption.original_stop_frame,
                        "original_user_frame": timeline.interruption.original_user_frame,
                        "shifted_frames": timeline.interruption.shifted_frames,
                    }
                ),
            }
            for timeline in timelines
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "preflight.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    table = format_timeline_table(interrupted[0])
    (output_dir / "decoded_timeline.txt").write_text(table + "\n", encoding="utf-8")
    print("Session 09 exact label histogram:", json.dumps(report["label_histogram"], sort_keys=True))
    print("Session 09 decoded interrupted timeline:\n" + table)
    required = (EventKind.IDLE.value, EventKind.START.value, EventKind.TEXT.value, EventKind.STOP.value)
    if any(histogram[name] == 0 for name in required):
        raise RuntimeError(f"Session 09 preflight is missing required labels: {histogram}.")
    return report


def _training_trend(path: Path) -> dict[str, Any]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if "train_total_loss" in row and int(row.get("step", 0)) > 0:
                rows.append(row)
    if not rows:
        raise RuntimeError("No per-step training metrics were recorded.")
    width = min(10, max(1, len(rows) // 4))
    keys = ["train_total_loss", *[f"train_{group}_loss" for group in GROUPS]]
    return {
        "logged_steps": len(rows),
        "window_steps": width,
        "first_window_mean": {
            key: sum(float(row[key]) for row in rows[:width]) / width for key in keys
        },
        "last_window_mean": {
            key: sum(float(row[key]) for row in rows[-width:]) / width for key in keys
        },
    }


def _session09_gate(
    baseline: Mapping[str, Any],
    teacher: Mapping[str, Any],
    free: Mapping[str, Any],
    timelines: Sequence[WindowTimeline],
    trend: Mapping[str, Any],
    stop_token_id: int,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if trend["last_window_mean"]["train_total_loss"] >= trend["first_window_mean"]["train_total_loss"]:
        failures.append("training loss did not trend downward")
    for group in ("text", "start", "stop"):
        if baseline["target_counts"][group] <= 0:
            failures.append(f"{group} received no examples")
        if teacher["group_losses"][group] >= baseline["group_losses"][group]:
            failures.append(f"{group} loss did not improve")
    for name, report in (("teacher", teacher), ("cached_free", free)):
        fractions = report["predicted_fractions"]
        if fractions["idle"] >= 0.98:
            failures.append(f"{name} predictions collapsed to IDLE")
        if not any(
            example["start_text_stop_exact_at_target_frames"]
            for example in report["examples"]
        ):
            failures.append(f"{name} produced no exact START/text/STOP example")
    interrupted_index = next(
        index for index, timeline in enumerate(timelines) if timeline.interruption is not None
    )
    interrupted = timelines[interrupted_index]
    assert interrupted.interruption is not None
    cut = interrupted.interruption.cut_frame
    free_ids = free["examples"][interrupted_index]["prediction_ids"]
    predicted_stops = [
        index
        for index, token_id in enumerate(free_ids)
        if token_id == stop_token_id
    ]
    if not any(frame >= cut for frame in predicted_stops):
        failures.append("interrupted cached/free example emitted no STOP after shifted user onset")
    return not failures, failures


def train_from_config(config: Mapping[str, Any]) -> Session08Trainer | SimpleNamespace:
    """Run the configured job; callers decide whether it is a smoke or real dataset run."""

    seed_everything(config["training"]["seed"])
    torch.cuda.reset_peak_memory_stats()
    model, processor, targets = build_training_model(config)
    train_dataset, eval_dataset, collator = build_datasets_and_collator(
        config, processor, model
    )
    session09 = config["data"].get("selected_windows") is not None
    output_dir = Path(config["training"]["output_dir"])
    if session09 and (output_dir / "metrics.jsonl").exists() and not config["training"].get(
        "resume_from_checkpoint"
    ):
        raise FileExistsError(
            f"Refusing to append a fresh Session 09 run to {output_dir / 'metrics.jsonl'}."
        )
    preflight = _write_session09_preflight(output_dir, train_dataset) if session09 else None
    baseline = (
        _teacher_forced_report(model, train_dataset, collator, processor.tokenizer)
        if session09
        else None
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
    final = output_dir / "final"
    trainer.save_model(final)
    if not session09:
        return trainer

    assert preflight is not None and baseline is not None
    step_count = trainer.state.global_step
    teacher = _teacher_forced_report(model, train_dataset, collator, processor.tokenizer)
    free = _cached_free_report(model, train_dataset, collator, processor.tokenizer)
    fixed_batch = _cpu_model_batch(collator, train_dataset[0])
    reference_logits = _reference_logits(model, fixed_batch)
    trend = _training_trend(output_dir / "metrics.jsonl")
    timelines = _timeline_values(train_dataset)
    stop_token_id = model.control_tokens.stop
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    nonzero_gradient_count = len(audit.nonzero_gradient_names)

    del trainer, audit, model, processor, train_dataset, eval_dataset, collator
    _release_cuda_objects()

    seed_everything(config["training"]["seed"])
    reloaded_model, reloaded_processor, reloaded_targets = build_training_model(config)
    _load_adapter_weights(reloaded_model, final)
    reloaded_train, _, reloaded_collator = build_datasets_and_collator(
        config, reloaded_processor, reloaded_model
    )
    reloaded_teacher = _teacher_forced_report(
        reloaded_model,
        reloaded_train,
        reloaded_collator,
        reloaded_processor.tokenizer,
    )
    reloaded_free = _cached_free_report(
        reloaded_model,
        reloaded_train,
        reloaded_collator,
        reloaded_processor.tokenizer,
    )
    reloaded_logits = _reference_logits(reloaded_model, fixed_batch)
    reload_max_abs = float((reference_logits.float() - reloaded_logits.float()).abs().max())
    torch.testing.assert_close(reloaded_logits, reference_logits, rtol=1e-3, atol=1e-3)
    if [example["prediction_ids"] for example in teacher["examples"]] != [
        example["prediction_ids"] for example in reloaded_teacher["examples"]
    ]:
        raise AssertionError("Teacher-forced predictions changed after adapter reload.")
    if [example["prediction_ids"] for example in free["examples"]] != [
        example["prediction_ids"] for example in reloaded_free["examples"]
    ]:
        raise AssertionError("Cached/free predictions changed after adapter reload.")
    passed, failures = _session09_gate(
        baseline,
        teacher,
        free,
        _timeline_values(reloaded_train),
        trend,
        stop_token_id,
    )
    report = {
        "overfit_gate": "PASS" if passed else "FAIL",
        "failures": failures,
        "optimizer_steps": step_count,
        "preflight": preflight,
        "baseline_teacher_forced": baseline,
        "final_teacher_forced": teacher,
        "final_cached_free": free,
        "training_trend": trend,
        "reload": {
            "teacher_predictions_identical": True,
            "cached_free_predictions_identical": True,
            "max_abs_logit_difference": reload_max_abs,
            "rtol": 1e-3,
            "atol": 1e-3,
        },
        "discovered_target_count": len(reloaded_targets),
        "trainable_lora_parameter_count": trainable_count,
        "nonzero_lora_gradient_parameter_count": nonzero_gradient_count,
        "peak_vram_allocated_bytes": peak_allocated,
        "peak_vram_reserved_bytes": peak_reserved,
        "assertions": {
            "alignment": "passed",
            "mask_and_audio_lengths": "passed",
            "causal_target_shift": "passed",
            "adapter_reload": "passed",
        },
    }
    with (output_dir / "overfit_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"OVERFIT_GATE={'PASS' if passed else 'FAIL'}")
    if failures:
        print("Gate failures:", "; ".join(failures))
    return SimpleNamespace(
        state=SimpleNamespace(global_step=step_count),
        session09_report=report,
    )


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
