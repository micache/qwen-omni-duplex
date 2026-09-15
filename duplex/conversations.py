"""Complete-conversation dataset and exact audio-frame context preflight."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .contract import prompt_token_ids
from .dataset import (ConversationMetadata, ManifestEntry, Split, assign_split,
                      load_conversation, read_manifest)
from .streaming import prepare_partial_audio_for_extractor, split_fixed_audio_chunks
from .timeline import build_window_timeline


@dataclass(frozen=True)
class ConversationLength:
    conversation_id: str
    split: Split
    duration_seconds: float
    audio_frames: int
    constructed_length: int


def exact_audio_frame_count(waveform, processor, audio_tower) -> int:
    """Count valid encoder outputs using the same feature masks as inference."""
    extractor = getattr(processor, "feature_extractor", processor)
    total = 0
    for chunk in split_fixed_audio_chunks(waveform):
        if type(extractor).__name__ == "WhisperFeatureExtractor" and int(extractor.hop_length) == 160:
            preconv = max(3, chunk.valid_samples // 160)
            total += int(torch.as_tensor(audio_tower._get_feat_extract_output_lengths(
                torch.tensor([preconv]))[1]).item())
            continue
        valid = chunk.waveform[chunk.valid_mask]
        prepared, valid_preconv = prepare_partial_audio_for_extractor(valid, extractor)
        features = extractor([prepared], sampling_rate=16_000, padding=True,
                             return_attention_mask=True, return_tensors="pt")
        mask = features.get("feature_attention_mask") if isinstance(features, Mapping) else getattr(features, "feature_attention_mask", None)
        if mask is None:
            mask = features.get("attention_mask") if isinstance(features, Mapping) else getattr(features, "attention_mask", None)
        if mask is None:
            raise ValueError("Qwen feature extractor returned no attention mask.")
        if chunk.valid_samples < 32_000:
            mask = torch.as_tensor(mask).clone()
            mask.zero_()
            mask[:, :min(valid_preconv, mask.shape[1])] = 1
        lengths = torch.as_tensor(mask).long().sum(dim=-1)
        total += int(torch.as_tensor(audio_tower._get_feat_extract_output_lengths(lengths)[1]).item())
    return total


class CompleteConversationDataset(Dataset):
    """Each indexed item reconstructs exactly one complete DailyTalk recording."""

    def __init__(self, entries: list[ManifestEntry], *, root: str | Path,
                 split: Split, config: Mapping, processor, model,
                 preflight_lengths: Mapping[str, ConversationLength]) -> None:
        self.entries = tuple(entries)
        self.root = Path(root)
        self.split = split
        self.config = config
        self.processor = processor
        self.model = model
        self.preflight_lengths = preflight_lengths

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        entry = self.entries[index]
        record = load_conversation(entry, dataset_root=self.root,
                                   speaker_label_map=self.config["data"].get("speaker_label_map"))
        from scripts.run_session11_diagnostic import _infer_user_words
        record = replace(record, user_words=_infer_user_words(record, self.config["diagnostic"]))
        metadata = ConversationMetadata(record.conversation_id, self.split, 0.0, record.duration_seconds)
        return build_window_timeline(record, metadata, tokenizer=self.processor.tokenizer,
                                     control_tokens=self.model.control_tokens,
                                     thinker_bos_token_id=self.model.base_thinker.config.bos_token_id,
                                     frame_rate_hz=25,
                                     valid_frame_count=self.preflight_lengths[record.conversation_id].audio_frames)


def preflight_complete_conversations(config: Mapping, processor, model):
    """Verify every constructed sequence before exposing any training items."""
    root = Path(config["data"]["root"])
    entries = read_manifest(root / config["data"].get("manifest", "dailytalk.jsonl"))
    split_salt = f"DailyTalkContiguous-session11-{config['training']['split_seed']}"
    partitions = {split: [] for split in Split}
    lengths: dict[str, ConversationLength] = {}
    context_length = len(prompt_token_ids(processor.tokenizer))
    thinker_config = model.base_thinker.config.text_config
    context_limit = int(thinker_config.max_position_embeddings)
    violations = []
    corrupt = []
    for entry in entries:
        split = assign_split(entry.conversation_id, salt=split_salt)
        try:
            record = load_conversation(entry, dataset_root=root,
                                       speaker_label_map=config["data"].get("speaker_label_map"))
        except Exception as error:
            corrupt.append({"id": entry.conversation_id, "error": str(error)})
            continue
        try:
            frames = exact_audio_frame_count(record.user_waveform, processor,
                                             model.base_thinker.audio_tower)
        except Exception as error:
            raise RuntimeError(f"{entry.conversation_id}: audio-frame preflight failed: {error}") from error
        required = context_length + frames
        value = ConversationLength(entry.conversation_id, split, record.duration_seconds,
                                   frames, required)
        lengths[entry.conversation_id] = value
        if required > context_limit:
            violations.append(value)
        else:
            partitions[split].append(entry)
    if violations:
        detail = ", ".join(f"{row.conversation_id}: {row.constructed_length}" for row in violations)
        raise ValueError(f"Conversations exceed Thinker context limit {context_limit}: {detail}")
    report = {"context_limit": context_limit, "prompt_tokens": context_length,
              "conversations": {split.value: len(partitions[split]) for split in Split},
              "duration_seconds": {split.value: sum(lengths[e.conversation_id].duration_seconds for e in partitions[split]) for split in Split},
              "audio_frames": {split.value: sum(lengths[e.conversation_id].audio_frames for e in partitions[split]) for split in Split},
              "excluded": corrupt}
    datasets = {split: CompleteConversationDataset(partitions[split], root=root, split=split,
                        config=config, processor=processor, model=model,
                        preflight_lengths=lengths) for split in Split}
    return datasets, report
