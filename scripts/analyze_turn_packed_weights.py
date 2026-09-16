#!/usr/bin/env python3
"""Measure turn-packed label counts and derive bounded loss weights."""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/TASTE-IF-SFT-48K"))
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--interruption-probability", type=float, default=0.2)
    parser.add_argument("--max-weight-ratio", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _resampled_samples(frames: int, sample_rate: int) -> int:
    divisor = math.gcd(sample_rate, 16_000)
    return math.ceil(frames * (16_000 // divisor) / (sample_rate // divisor))


def _qwen_frames(samples: int) -> int:
    full_chunks, remainder = divmod(samples, 32_000)
    result = full_chunks * 50
    if remainder:
        preconv = max(3, remainder // 160)
        post_conv = (preconv - 1) // 2 + 1
        result += (post_conv - 2) // 2 + 1
    return result


def _audio_samples(value: dict[str, object]) -> int:
    info = sf.info(io.BytesIO(value["bytes"]))
    return _resampled_samples(info.frames, info.samplerate)


def _bounded_inverse_sqrt(counts: dict[str, float], max_ratio: float) -> dict[str, float]:
    raw = {name: 1.0 / math.sqrt(max(count, 1.0)) for name, count in counts.items()}
    floor = max(raw.values()) / max_ratio
    capped = {name: max(value, floor) for name, value in raw.items()}
    text_scale = capped["text"]
    return {name: value / text_scale for name, value in capped.items()}


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.interruption_probability <= 1.0:
        raise ValueError("interruption probability must be in [0, 1].")
    if args.max_weight_ratio < 1.0:
        raise ValueError("max weight ratio must be at least one.")

    from transformers import Qwen2_5OmniProcessor

    processor = Qwen2_5OmniProcessor.from_pretrained(
        "Qwen/Qwen2.5-Omni-3B",
        revision="f75b40e3da2003cdd6e1829b1f420ca70797c34e",
        local_files_only=True,
    )
    pattern = "shuffled_dev.parquet" if args.split == "dev" else "shuffled_train_part_*.parquet"
    paths = sorted((args.dataset_root / "data").glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {args.split} shards under {args.dataset_root / 'data'}.")

    conversations = 0
    total_frames = 0
    text_tokens = 0
    expected_interrupted_text = 0.0
    eligible = 0
    minimum_margin: int | None = None
    for path_index, path in enumerate(paths, start=1):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=100,
            columns=("instruction_audio", "response_audio", "message"),
        ):
            rows = batch.to_pylist()
            texts = [row["message"][1]["text"] for row in rows]
            encoded = processor.tokenizer(texts, add_special_tokens=False)["input_ids"]
            for row, token_ids in zip(rows, encoded):
                user_samples = _audio_samples(row["instruction_audio"])
                assistant_samples = _audio_samples(row["response_audio"])
                frames = _qwen_frames(user_samples + assistant_samples)
                user_frames = min(max(round(user_samples * 25 / 16_000), 1), frames - 2)
                assistant_frames = frames - user_frames
                margin = assistant_frames - len(token_ids) - 2
                minimum_margin = margin if minimum_margin is None else min(minimum_margin, margin)
                if margin < 0:
                    raise ValueError(
                        f"{path}: response needs {len(token_ids) + 2} events but has "
                        f"only {assistant_frames} assistant frames."
                    )
                count = len(token_ids)
                conversations += 1
                total_frames += frames
                text_tokens += count
                if count >= 4:
                    eligible += 1
                    retained_if_interrupted = (count + 2.0) / 2.0
                    expected_interrupted_text += (
                        (1.0 - args.interruption_probability) * count
                        + args.interruption_probability * retained_if_interrupted
                    )
                else:
                    expected_interrupted_text += count
        print(f"scanned {path_index}/{len(paths)}: {path.name}", file=sys.stderr)

    base = {
        "text": float(text_tokens),
        "idle": float(total_frames - text_tokens - 2 * conversations),
        "start": float(conversations),
        "stop": float(conversations),
    }
    expected = {
        "text": expected_interrupted_text,
        "idle": float(total_frames - 2 * conversations) - expected_interrupted_text,
        "start": float(conversations),
        "stop": float(conversations),
    }
    capped_inverse_sqrt = _bounded_inverse_sqrt(expected, args.max_weight_ratio)
    selected_weights = {"text": 1.0, "idle": 0.1, "start": 4.0, "stop": 4.0}
    selected_mass = {
        name: expected[name] * selected_weights[name] for name in expected
    }
    selected_total = sum(selected_mass.values())
    payload = {
        "split": args.split,
        "conversations": conversations,
        "total_frames": total_frames,
        "minimum_assistant_capacity_margin_frames": minimum_margin,
        "interruption_probability": args.interruption_probability,
        "interruption_eligible_conversations": eligible,
        "base_counts": base,
        "expected_augmented_counts_per_epoch": expected,
        "diagnostic_capped_inverse_sqrt_weights": capped_inverse_sqrt,
        "exact_idle_weight_for_equal_idle_text_mass": expected["text"] / expected["idle"],
        "selected_full_training_weights": selected_weights,
        "selected_expected_weighted_mass": selected_mass,
        "selected_expected_weighted_fraction": {
            name: value / selected_total for name, value in selected_mass.items()
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
