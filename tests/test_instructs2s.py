"""Prepared InstructS2S audio must retain its speech boundary and causal targets."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from duplex.instructs2s import InstructS2SFirstTurnDataset
from duplex.timeline import ControlTokenIds
from duplex.turn_packed import NoiseAugmentationConfig, TurnPackedCollator


class Tokenizer:
    def encode(self, text, **kwargs):
        return [4, 5, 6]

    def decode(self, ids, **kwargs):
        return "answer" if len(ids) == 3 else "part"


class Extractor:
    def __call__(self, waveforms, **kwargs):
        lengths = [len(waveform) // 160 for waveform in waveforms]
        mask = torch.zeros(len(lengths), max(lengths), dtype=torch.long)
        for index, length in enumerate(lengths):
            mask[index, :length] = 1
        return {"input_features": torch.zeros(len(lengths), 128, max(lengths)),
                "attention_mask": mask}


def prepared_dataset(tmp_path):
    sf.write(tmp_path / "question.flac", np.ones(35000, dtype=np.float32) * 0.1, 16000)
    row = {"id": "example", "audio": "question.flac", "user_samples": 35000,
           "assistant_samples": 32000, "answer": "answer", "frames": 150}
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n")
    return InstructS2SFirstTurnDataset(tmp_path, split="train")


@pytest.mark.parametrize("interruption_probability", [0.0, 1.0])
def test_prepared_audio_through_augmented_causal_collation(tmp_path, interruption_probability):
    dataset = prepared_dataset(tmp_path)
    item = dataset[0]
    assert item.user_valid_samples == 35000
    assert len(item.user_waveform) == 64000
    assert not item.user_waveform[35000:].any()
    assert dataset.frame_lengths == [150]
    tower = SimpleNamespace(_get_feat_extract_output_lengths=lambda lengths: (lengths // 2, lengths // 4))
    collator = TurnPackedCollator(
        audio_processor=Extractor(), audio_tower=tower, tokenizer=Tokenizer(),
        control_tokens=ControlTokenIds(idle=1, start=2, stop=3, thinker_vocab_size=11),
        thinker_bos_token_id=2, interruption_probability=interruption_probability,
        min_assistant_frames=1, noise_dataset=None,
        noise_config=NoiseAugmentationConfig(), augmentation_seed=17,
    )
    batch = collator([item])
    labels = batch["labels"][0]
    start = round(35000 * 25 / 16000)
    assert labels[start].item() == 2
    assert labels[:start].tolist() == [1] * start
    assert batch["control_ids"][0, start + 1].item() == 2
    assert batch["control_mask"][0, start + 1].item()
    assert batch["bootstrap_mask"][0].nonzero().flatten().tolist() == [0]
    assert batch["audio_chunk_counts"].tolist() == [3]
    assert batch["preconv_feature_lengths"].tolist() == [200, 200, 200]
    assert batch["context_ids"].shape == (1, 0)
    assert labels.shape == (150,)
    assert (labels == 3).sum().item() == 1
    interruption = batch["sample_metadata"][0]["interruption"]
    if interruption_probability:
        assert interruption.cut_frame < start + 4
        assert labels[interruption.cut_frame + 1:].tolist() == [1] * (149 - interruption.cut_frame)
    else:
        assert interruption is None
        assert labels[start + 1:start + 5].tolist() == [4, 5, 6, 3]


def test_manifest_audio_mismatch_is_rejected(tmp_path):
    dataset = prepared_dataset(tmp_path)
    dataset.records[0]["user_samples"] += 1
    with pytest.raises(ValueError, match="differs from its manifest"):
        dataset[0]
