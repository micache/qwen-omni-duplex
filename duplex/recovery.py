"""Frozen continuous-span reconstruction for Session 15 repair."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path

from .dataset import SpanMetadata, Split, load_conversation, read_manifest
from .timeline import (ControlTokenIds, InterruptionConfig, WindowTimeline,
                       augment_synthetic_interruption, build_window_timeline)


@dataclass(frozen=True)
class ContinuousSpan:
    conversation_id: str
    split: Split
    normal: WindowTimeline
    interrupted: WindowTimeline


def load_continuous_spans(config, tokenizer, controls: ControlTokenIds, bos_token_id: int):
    """Rebuild targets against the outer span, never from cropped chunk targets."""
    from scripts.run_session11_diagnostic import _infer_user_words

    path = Path(config["diagnostic"]["subset_manifest"])
    manifest = json.loads(path.read_text())
    entries = {entry.conversation_id: entry for entry in read_manifest(
        Path(config["data"]["root"]) / config["data"]["manifest"])}
    spans = []
    for item in manifest["items"]:
        record = load_conversation(entries[item["conversation_id"]],
                                   dataset_root=config["data"]["root"],
                                   speaker_label_map=config["data"]["speaker_label_map"])
        record = replace(record, user_words=_infer_user_words(record, config["diagnostic"]))
        metadata = SpanMetadata(record.conversation_id, Split(item["split"]),
                                float(item["span_start_seconds"]), float(item["span_end_seconds"]))
        normal = build_window_timeline(record, metadata, tokenizer=tokenizer,
                                       control_tokens=controls, thinker_bos_token_id=bos_token_id,
                                       frame_rate_hz=25)
        seed = int.from_bytes(hashlib.sha256(
            f"{config['training']['interruption_seed']}\0{record.conversation_id}".encode()).digest()[:8], "big")
        interrupted = augment_synthetic_interruption(
            normal, config=InterruptionConfig(probability=1.0,
                                              min_assistant_frames=config["data"]["min_assistant_frames"]),
            rng=random.Random(seed), control_tokens=controls, thinker_bos_token_id=bos_token_id)
        if interrupted is normal:
            raise ValueError(f"{record.conversation_id}: frozen span lacks a continuous interruption candidate.")
        if len(normal.targets.events) != 200 or len(interrupted.targets.events) != 200:
            raise ValueError(f"{record.conversation_id}: continuous target length changed.")
        spans.append(ContinuousSpan(record.conversation_id, metadata.split, normal, interrupted))
    return spans, manifest
