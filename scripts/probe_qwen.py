"""Reproducible CUDA compatibility probe for the Qwen2.5-Omni-3B Thinker.

The probe is deliberately fail-fast. It writes its JSON result after every
stage so a failed load or shape assertion leaves the raw observation behind.
It performs inference only and never modifies installed packages.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from typing import Any


MODEL_ID = "Qwen/Qwen2.5-Omni-3B"
SAMPLE_RATE = 16_000
CLIP_SECONDS = 2
EXPECTED_FRAMES = 50
EXPECTED_WIDTH = 2_048
PACKAGE_NAMES = (
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "bitsandbytes",
    "qwen-omni-utils",
    "huggingface-hub",
    "flash-attn",
    "pillow",
    "torchvision",
    "triton",
)


class ProbeFailure(RuntimeError):
    """An acceptance condition failed with evidence already in the result."""


def _shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return list(shape) if shape is not None else None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as error:
        return {"command": command, "error": repr(error)}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _source_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path).resolve()
    return {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def _jsonable_error(error: BaseException) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error),
        "repr": repr(error),
        "traceback": "".join(traceback.format_exception(error)),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value, key=str)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


class Recorder:
    def __init__(self, output: Path, result: dict[str, Any]) -> None:
        self.output = output
        self.result = result

    def write(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.result, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.output)

    @contextlib.contextmanager
    def stage(self, name: str):
        entry: dict[str, Any] = {"status": "running"}
        self.result.setdefault("stages", {})[name] = entry
        self.write()
        try:
            yield entry
        except BaseException as error:
            entry["status"] = "failed"
            entry["error"] = _jsonable_error(error)
            self.write()
            raise
        else:
            entry["status"] = "passed"
            self.write()


def _memory(torch: Any, device: Any) -> dict[str, int]:
    torch.cuda.synchronize(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }


def _reset_peak(torch: Any, device: Any) -> dict[str, int]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    return _memory(torch, device)


def _snapshot_record(config_path: str, commit: str) -> dict[str, Any]:
    # Keep the snapshot path itself: resolving the config symlink would move us
    # into the Hub's content-addressed blobs directory and lose shard names.
    path = Path(config_path).absolute()
    snapshot = path.parent
    files = []
    total_bytes = 0
    for item in sorted(snapshot.glob("*.safetensors")):
        size = item.stat().st_size
        total_bytes += size
        files.append({"name": item.name, "bytes": size})
    return {
        "resolved_commit": commit,
        "snapshot_directory": str(snapshot),
        "config_path": str(path),
        "checkpoint_files_present": files,
        "checkpoint_bytes_present": total_bytes,
    }


def _loading_keys(info: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(info.get("missing_keys", []))
    unexpected = sorted(info.get("unexpected_keys", []))
    mismatched = info.get("mismatched_keys", [])
    error_messages = info.get("error_msgs", [])
    critical_prefixes = ("thinker.", "audio_tower.", "visual.", "model.", "lm_head.")
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
        "error_messages": error_messages,
        "critical_missing_keys": missing,
        "critical_unexpected_keys": [key for key in unexpected if key.startswith(critical_prefixes)],
    }


def _load_thinker(
    *,
    route: str,
    model_id: str,
    revision: str,
    root_config: Any,
    device_name: str,
    torch: Any,
    thinker_class: Any,
    full_class: Any,
) -> tuple[Any, dict[str, Any], Any | None]:
    common = {
        "revision": revision,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": device_name},
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
        "output_loading_info": True,
    }
    if route == "direct_thinker":
        model, info = thinker_class.from_pretrained(model_id, **common)
        return model, info, None
    if route == "full_audio_disabled":
        root_config.enable_audio_output = False
        parent, info = full_class.from_pretrained(model_id, config=root_config, **common)
        return parent.thinker, info, parent
    raise ValueError(f"Unknown load route: {route}")


def _processor_batch(processor: Any, waveforms: list[Any]) -> Any:
    prompt = processor.audio_bos_token + processor.audio_token + processor.audio_eos_token
    return processor(
        text=[prompt] * len(waveforms),
        audio=waveforms,
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_tensors="pt",
    )


def run(args: argparse.Namespace, recorder: Recorder) -> None:
    with recorder.stage("environment") as stage:
        import torch
        import transformers
        from transformers.models.qwen2_5_omni import modeling_qwen2_5_omni
        from transformers.models.qwen2_5_omni import processing_qwen2_5_omni

        stage.update(
            {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "packages": {name: _distribution_version(name) for name in PACKAGE_NAMES},
                "torch": {
                    "version": torch.__version__,
                    "compiled_cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(),
                    "cuda_available": torch.cuda.is_available(),
                    "device_count": torch.cuda.device_count(),
                },
                "nvidia_smi": _command(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,name,driver_version,memory.total,compute_cap",
                        "--format=csv,noheader",
                    ]
                ),
                "nvcc": _command(["nvcc", "--version"]),
                "git": {
                    "head": _command(["git", "rev-parse", "HEAD"]),
                    "status": _command(["git", "status", "--short", "--branch"]),
                },
                "installed_sources": {
                    "modeling": _source_record(modeling_qwen2_5_omni.__file__),
                    "processing": _source_record(processing_qwen2_5_omni.__file__),
                    "transformers_init": _source_record(transformers.__file__),
                },
            }
        )
        try:
            torch.cuda.init()
        except BaseException as error:
            stage["cuda_initialization"] = {"status": "failed", "error": _jsonable_error(error)}
        else:
            stage["cuda_initialization"] = {"status": "passed"}
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise ProbeFailure("CUDA GPU required, but PyTorch reports no available CUDA device")

        device = torch.device(args.device)
        properties = torch.cuda.get_device_properties(device)
        bf16_supported = torch.cuda.is_bf16_supported()
        flash_package = _distribution_version("flash-attn")
        stage["selected_gpu"] = {
            "device": str(device),
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
            "preferred_24gb_met": properties.total_memory >= 24 * 1024**3,
        }
        stage["bf16"] = {"torch_reports_supported": bf16_supported}
        stage["flash_attention_2"] = {
            "package_version": flash_package,
            "transformers_reports_available": transformers.utils.is_flash_attn_2_available(),
            "model_class_declares_support": bool(
                getattr(modeling_qwen2_5_omni.Qwen2_5OmniThinkerForConditionalGeneration, "_supports_flash_attn", False)
            ),
        }
        if not bf16_supported:
            raise ProbeFailure("Selected CUDA GPU does not support BF16 in this PyTorch build")
        try:
            left = torch.randn((128, 128), device=device, dtype=torch.bfloat16)
            right = torch.randn((128, 128), device=device, dtype=torch.bfloat16)
            with torch.inference_mode():
                product = left @ right
            torch.cuda.synchronize(device)
            stage["bf16"]["kernel"] = {"status": "passed", "output_shape": _shape(product)}
        except BaseException as error:
            stage["bf16"]["kernel"] = {"status": "failed", "error": _jsonable_error(error)}
            raise ProbeFailure("BF16 CUDA kernel test failed") from error
        if transformers.utils.is_flash_attn_2_available():
            try:
                from flash_attn import flash_attn_func

                query = torch.randn((1, 8, 4, 64), device=device, dtype=torch.bfloat16)
                with torch.inference_mode():
                    flash_output = flash_attn_func(query, query, query, causal=True)
                torch.cuda.synchronize(device)
                stage["flash_attention_2"]["kernel"] = {
                    "status": "passed",
                    "output_shape": _shape(flash_output),
                }
            except BaseException as error:
                stage["flash_attention_2"]["kernel"] = {
                    "status": "failed",
                    "error": _jsonable_error(error),
                }
        else:
            stage["flash_attention_2"]["kernel"] = {
                "status": "not_run",
                "reason": "flash-attn is not installed or Transformers cannot load it",
            }

    from transformers import (
        Qwen2_5OmniConfig,
        Qwen2_5OmniForConditionalGeneration,
        Qwen2_5OmniProcessor,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )
    from transformers.utils import cached_file
    import numpy as np

    with recorder.stage("checkpoint_and_config") as stage:
        config_path = cached_file(args.model, "config.json", revision=args.revision)
        root_config = Qwen2_5OmniConfig.from_pretrained(args.model, revision=args.revision)
        commit = root_config._commit_hash
        if not commit:
            raise ProbeFailure("Transformers did not report the resolved checkpoint commit")
        stage["requested"] = {"model": args.model, "revision": args.revision}
        stage["download"] = _snapshot_record(config_path, commit)
        thinker_vocab = root_config.thinker_config.text_config.vocab_size
        control_ids = {
            "tts_text_pad_token_id": root_config.talker_config.tts_text_pad_token_id,
            "tts_text_start_token_id": root_config.talker_config.tts_text_start_token_id,
            "tts_text_end_token_id": root_config.talker_config.tts_text_end_token_id,
        }
        stage["configuration"] = {
            "model_type": root_config.model_type,
            "thinker_vocab_size": thinker_vocab,
            "thinker_hidden_size": root_config.thinker_config.text_config.hidden_size,
            "position_id_per_seconds": root_config.thinker_config.position_id_per_seconds,
            "seconds_per_chunk": root_config.thinker_config.seconds_per_chunk,
            "control_ids": control_ids,
            "control_ids_in_thinker_vocab": {name: 0 <= value < thinker_vocab for name, value in control_ids.items()},
        }
        if root_config.thinker_config.position_id_per_seconds != 25 or root_config.thinker_config.seconds_per_chunk != 2:
            raise ProbeFailure("Checkpoint does not validate the fixed 25 Hz / 2 second contract")
        if not all(0 <= value < thinker_vocab for value in control_ids.values()):
            raise ProbeFailure("At least one TTS text control ID is outside the Thinker vocabulary")

    device = torch.device(args.device)
    processor = None
    thinker = None
    parent = None
    load_errors: dict[str, Any] = {}
    with recorder.stage("model_load") as stage:
        selected_route = None
        loading = None
        load_attempts = {}
        for route in ("direct_thinker", "full_audio_disabled"):
            attempt = {"vram_before": _reset_peak(torch, device)}
            load_attempts[route] = attempt
            try:
                thinker, raw_info, parent = _load_thinker(
                    route=route,
                    model_id=args.model,
                    revision=commit,
                    root_config=root_config,
                    device_name=args.device,
                    torch=torch,
                    thinker_class=Qwen2_5OmniThinkerForConditionalGeneration,
                    full_class=Qwen2_5OmniForConditionalGeneration,
                )
                loading = _loading_keys(raw_info)
                if loading["critical_missing_keys"] or loading["critical_unexpected_keys"] or loading["mismatched_keys"]:
                    raise ProbeFailure(f"Critical checkpoint keys failed for route {route}")
                attempt["loading_info"] = loading
                attempt["vram_after"] = _memory(torch, device)
                attempt["status"] = "passed"
                selected_route = route
                break
            except BaseException as error:
                load_errors[route] = _jsonable_error(error)
                attempt["status"] = "failed"
                attempt["error"] = load_errors[route]
                attempt["vram_at_failure"] = _memory(torch, device)
                thinker = None
                parent = None
                gc.collect()
                torch.cuda.empty_cache()
                attempt["vram_after_cleanup"] = _memory(torch, device)
        stage["attempts"] = load_attempts
        stage["attempt_errors"] = load_errors
        if selected_route is None or thinker is None or loading is None:
            raise ProbeFailure("Neither supported Thinker loading route produced a complete Thinker")
        thinker.eval()
        stage["selected_route"] = selected_route
        stage["selection_reason"] = (
            "Direct Thinker load avoids constructing the parent, Talker, and token-to-wave modules."
            if selected_route == "direct_thinker"
            else "Direct Thinker load failed; the full parent was instantiated with audio output disabled."
        )
        stage["loading_info"] = loading
        stage["module_presence_before_prune"] = {
            "parent_retained": parent is not None,
            "talker": hasattr(parent, "talker") if parent is not None else False,
            "token2wav": hasattr(parent, "token2wav") if parent is not None else False,
            "vision": hasattr(thinker, "visual"),
            "audio_tower": hasattr(thinker, "audio_tower"),
            "text_model": hasattr(thinker, "model"),
        }
        stage["vram"] = _memory(torch, device)

    with recorder.stage("processor_shapes") as stage:
        processor = Qwen2_5OmniProcessor.from_pretrained(args.model, revision=commit)
        samples = np.arange(SAMPLE_RATE * CLIP_SECONDS, dtype=np.float32) / SAMPLE_RATE
        clips = [
            np.sin(2 * np.pi * 220.0 * samples).astype(np.float32),
            (0.5 * np.sin(2 * np.pi * 440.0 * samples)).astype(np.float32),
        ]
        batches = {}
        processor_outputs = {}
        for count in (1, 2):
            processed = _processor_batch(processor, clips[:count])
            processor_outputs[count] = processed
            batches[str(count)] = {
                "waveform_shapes": [list(clip.shape) for clip in clips[:count]],
                "input_features_shape": _shape(processed.input_features),
                "feature_attention_mask_shape": _shape(processed.feature_attention_mask),
                "feature_attention_mask_sums": processed.feature_attention_mask.sum(-1).tolist(),
                "input_ids_shape": _shape(processed.input_ids),
            }
        stage["sampling_rate"] = SAMPLE_RATE
        stage["clip_seconds"] = CLIP_SECONDS
        stage["batches"] = batches

    with recorder.stage("audio_encoding") as stage:
        processed = processor_outputs[2]
        input_features = processed.input_features.to(device=device, dtype=torch.bfloat16)
        feature_mask = processed.feature_attention_mask.to(device=device)
        pre_convolution_lengths = feature_mask.sum(-1)
        audio_feature_lengths, output_lengths = thinker.audio_tower._get_feat_extract_output_lengths(
            pre_convolution_lengths
        )
        before = _reset_peak(torch, device)
        with torch.inference_mode():
            audio_output = thinker.get_audio_features(
                input_features=input_features,
                feature_attention_mask=feature_mask,
                return_dict=True,
            )
        flattened = audio_output.last_hidden_state
        restored = list(torch.split(flattened, output_lengths.tolist(), dim=0))
        stage.update(
            {
                "input_features_shape": _shape(input_features),
                "feature_attention_mask_shape": _shape(feature_mask),
                "pre_convolution_lengths": pre_convolution_lengths.tolist(),
                "post_convolution_lengths": audio_feature_lengths.tolist(),
                "llm_output_lengths": output_lengths.tolist(),
                "return_type": type(audio_output).__name__,
                "last_hidden_state_layout": "flattened across samples on dimension 0",
                "last_hidden_state_shape": _shape(flattened),
                "restored_per_sample_shapes": [_shape(value) for value in restored],
                "vram_before": before,
                "vram_after": _memory(torch, device),
            }
        )
        if output_lengths.tolist() != [EXPECTED_FRAMES, EXPECTED_FRAMES]:
            raise ProbeFailure(f"Two-second audio did not yield {EXPECTED_FRAMES} vectors per sample")
        if any(tuple(value.shape) != (EXPECTED_FRAMES, EXPECTED_WIDTH) for value in restored):
            raise ProbeFailure(f"Restored audio features are not [{EXPECTED_FRAMES}, {EXPECTED_WIDTH}] per sample")

    with recorder.stage("vision_prune") as stage:
        before = _memory(torch, device)
        if not hasattr(thinker, "visual"):
            raise ProbeFailure("Thinker unexpectedly has no vision tower before pruning")
        del thinker.visual
        gc.collect()
        torch.cuda.empty_cache()
        stage["module_presence_after_prune"] = {
            "talker": hasattr(parent, "talker") if parent is not None else False,
            "token2wav": hasattr(parent, "token2wav") if parent is not None else False,
            "vision": hasattr(thinker, "visual"),
            "audio_tower": hasattr(thinker, "audio_tower"),
            "text_model": hasattr(thinker, "model"),
        }
        stage["vram_before"] = before
        stage["vram_after"] = _memory(torch, device)

    with recorder.stage("thinker_forward_and_cache") as stage:
        inputs_embeds = torch.stack(restored, dim=0)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        position_ids = torch.arange(inputs_embeds.shape[1], device=device).expand(inputs_embeds.shape[0], -1)
        before = _reset_peak(torch, device)
        with torch.inference_mode():
            output = thinker(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
        prefill_cache_length = output.past_key_values.get_seq_length()
        one_step_ids = torch.full(
            (inputs_embeds.shape[0], 1),
            root_config.talker_config.tts_text_pad_token_id,
            dtype=torch.long,
            device=device,
        )
        one_step_embeds = thinker.get_input_embeddings()(one_step_ids)
        cached_mask = torch.ones(
            (inputs_embeds.shape[0], inputs_embeds.shape[1] + 1), dtype=torch.long, device=device
        )
        cached_position_ids = torch.full(
            (inputs_embeds.shape[0], 1), inputs_embeds.shape[1], dtype=torch.long, device=device
        )
        with torch.inference_mode():
            cached_output = thinker(
                inputs_embeds=one_step_embeds,
                attention_mask=cached_mask,
                position_ids=cached_position_ids,
                past_key_values=output.past_key_values,
                use_cache=True,
                return_dict=True,
            )
        stage.update(
            {
                "prefill": {
                    "inputs_embeds_shape": _shape(inputs_embeds),
                    "attention_mask_shape": _shape(attention_mask),
                    "accepted_position_ids_shape": _shape(position_ids),
                    "logits_shape": _shape(output.logits),
                    "cache_type": type(output.past_key_values).__name__,
                    "cache_sequence_length": prefill_cache_length,
                },
                "cached_one_step": {
                    "inputs_embeds_shape": _shape(one_step_embeds),
                    "attention_mask_shape": _shape(cached_mask),
                    "accepted_position_ids_shape": _shape(cached_position_ids),
                    "logits_shape": _shape(cached_output.logits),
                    "cache_sequence_length": cached_output.past_key_values.get_seq_length(),
                },
                "position_ids_explanation": "[batch, sequence] is accepted and expanded internally to [3, batch, sequence].",
                "vram_before": before,
                "vram_after": _memory(torch, device),
            }
        )
        if output.logits.shape[:2] != (2, EXPECTED_FRAMES):
            raise ProbeFailure("Batch-two Thinker prefill returned an unexpected sequence layout")
        if cached_output.logits.shape[:2] != (2, 1):
            raise ProbeFailure("Cached one-step Thinker call returned an unexpected sequence layout")

    with recorder.stage("cleanup") as stage:
        before = _memory(torch, device)
        del cached_output, output, one_step_embeds, one_step_ids, inputs_embeds
        del restored, flattened, audio_output, input_features, feature_mask
        del thinker, parent, processor, processor_outputs
        gc.collect()
        torch.cuda.empty_cache()
        stage["vram_before"] = before
        stage["vram_after"] = _memory(torch, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default="main", help="Hub branch, tag, or immutable commit to resolve once")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=Path("notes/session06_probe.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result: dict[str, Any] = {
        "schema_version": 1,
        "probe": "qwen2_5_omni_3b_thinker_compatibility",
        "status": "running",
        "inference_only": True,
        "arguments": {
            "model": args.model,
            "revision": args.revision,
            "device": args.device,
            "output": str(args.output),
        },
    }
    recorder = Recorder(args.output, result)
    recorder.write()
    try:
        run(args, recorder)
    except BaseException as error:
        result["status"] = "failed"
        result["terminal_error"] = _jsonable_error(error)
        recorder.write()
        print(json.dumps({"status": "failed", "output": str(args.output), "error": str(error)}))
        raise SystemExit(1) from error
    result["status"] = "passed"
    recorder.write()
    print(json.dumps({"status": "passed", "output": str(args.output)}))


if __name__ == "__main__":
    main()
