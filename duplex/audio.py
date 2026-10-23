"""Shared 16 kHz resampling and exact fixed-chunk encoder lengths."""

import math
from collections.abc import Mapping

import numpy as np
import torch

TARGET_SAMPLE_RATE_HZ = 16_000


def resample_waveform_to_16khz(
    waveform: np.ndarray, source_sample_rate_hz: int
) -> np.ndarray:
    """Resample one mono waveform to 16 kHz; no other helper resamples audio."""

    from scipy.signal import resample_poly

    if (
        isinstance(source_sample_rate_hz, bool)
        or not isinstance(source_sample_rate_hz, int)
        or source_sample_rate_hz <= 0
    ):
        raise ValueError("Source sample rate must be a positive integer.")
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim != 1:
        raise ValueError(f"Expected a mono waveform, got shape {waveform.shape}.")
    if source_sample_rate_hz == TARGET_SAMPLE_RATE_HZ:
        return waveform.copy()
    divisor = math.gcd(source_sample_rate_hz, TARGET_SAMPLE_RATE_HZ)
    result = resample_poly(
        waveform,
        TARGET_SAMPLE_RATE_HZ // divisor,
        source_sample_rate_hz // divisor,
    )
    return np.asarray(result, dtype=np.float32)


def exact_audio_frame_count(waveform, processor, audio_tower) -> int:
    """Count valid encoder outputs using the same feature masks as inference."""
    from .streaming import prepare_partial_audio_for_extractor, split_fixed_audio_chunks

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
