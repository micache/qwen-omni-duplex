"""Run explicit Qwen2.5-Omni Thinker full-duplex streaming inference."""

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import soundfile as sf
import torch

from duplex.streaming import QwenDuplexStreamer, write_trace_jsonl
from duplex.timeline import DEFAULT_SYSTEM_PROMPT, prompt_token_ids
from duplex.training import (
    load_adapter_weights,
    build_training_model,
    load_training_config,
)


def _context_token_ids(tokenizer: object, system: str, text_context: str) -> list[int]:
    if not text_context:
        return prompt_token_ids(tokenizer, system)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if text_context:
        messages.append({"role": "user", "content": text_context})
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if messages and callable(apply_template):
        encoded = apply_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(encoded, Mapping):
            encoded = encoded.get("input_ids")
        if encoded is None:
            raise ValueError("Chat-template tokenization returned no input_ids.")
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], (list, tuple)):
            if len(encoded) != 1:
                raise ValueError("Context tokenization unexpectedly returned a batch.")
            encoded = encoded[0]
        return [int(token_id) for token_id in encoded]
    context = "\n".join(value for value in (system, text_context) if value)
    return list(tokenizer.encode(context, add_special_tokens=True)) if context else []


def _load_generation_model(
    config_path: Path,
    *,
    adapter_path: Path | None,
    allow_download: bool,
    base_only: bool,
):
    config = load_training_config(config_path)
    if allow_download:
        config["model"]["local_files_only"] = False
    model, processor, _ = build_training_model(config)
    if not base_only:
        selected_adapter = adapter_path or Path(config["training"]["output_dir"]) / "final"
        if not selected_adapter.is_dir():
            raise FileNotFoundError(
                f"Adapter directory does not exist: {selected_adapter}. "
                "Pass --base-only only for an explicit unadapted checkpoint smoke."
            )
        load_adapter_weights(model, selected_adapter)
    model.eval()
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
    )
    parser.add_argument("--text-context", default="")
    parser.add_argument("--max-silent-chunks", type=int, default=4)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--base-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Real-checkpoint generation requires an available CUDA GPU.")
    waveform, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if waveform.ndim != 1:
        raise SystemExit(f"Input audio must be mono, got shape {waveform.shape}.")
    if sample_rate != 16_000:
        raise SystemExit(f"Input audio must be 16 kHz, got {sample_rate} Hz.")

    model, processor = _load_generation_model(
        args.config,
        adapter_path=args.adapter,
        allow_download=args.allow_download,
        base_only=args.base_only,
    )
    context_ids = _context_token_ids(processor.tokenizer, args.system, args.text_context)
    streamer = QwenDuplexStreamer(
        model,
        processor,
        max_silent_chunks=args.max_silent_chunks,
        sample=args.sample,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    result = streamer.run(
        waveform,
        sample_id=args.sample_id or args.audio.stem,
        context_token_ids=context_ids,
        sample_rate_hz=sample_rate,
    )
    write_trace_jsonl(args.trace, result.trace)
    summary = {
        "sample_id": result.sample_id,
        "text": result.text,
        "timed_out": result.timed_out,
        "stop_reason": result.stop_reason,
        "input_duration_s": result.input_duration_s,
        "processed_silent_chunks": result.processed_silent_chunks,
        "event_count": len(result.trace),
        "lexical_hidden_state_count": len(result.lexical_hidden_states),
        "word_spans": [asdict(span) for span in result.word_spans],
        "trace": str(args.trace),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
