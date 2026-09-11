"""Opt-in Session 07 checks against the cached Qwen2.5-Omni checkpoint."""

import gc
import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from duplex.model import MODEL_ID, QwenDuplexThinker


REVISION = "f75b40e3da2003cdd6e1829b1f420ca70797c34e"
RUN_GPU = os.environ.get("RUN_QWEN_GPU_TESTS") == "1"
pytestmark = pytest.mark.skipif(
    not RUN_GPU,
    reason="set RUN_QWEN_GPU_TESTS=1 for the explicit cached-checkpoint GPU smoke",
)


@dataclass
class LoadedQwen:
    root_config: object
    processor: object
    thinker: object
    duplex: QwenDuplexThinker
    device: torch.device


@pytest.fixture(scope="module")
def loaded_qwen():
    if not torch.cuda.is_available():
        pytest.fail("RUN_QWEN_GPU_TESTS=1 requires an available CUDA device")

    from transformers import (
        Qwen2_5OmniConfig,
        Qwen2_5OmniProcessor,
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    device = torch.device("cuda:0")
    root_config = Qwen2_5OmniConfig.from_pretrained(
        MODEL_ID, revision=REVISION, local_files_only=True
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(
        MODEL_ID, revision=REVISION, local_files_only=True
    )
    thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=REVISION,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    thinker.eval()
    if hasattr(thinker, "visual"):
        del thinker.visual
        gc.collect()
        torch.cuda.empty_cache()
    duplex = QwenDuplexThinker(thinker, qwen_config=root_config).eval()
    yield LoadedQwen(root_config, processor, thinker, duplex, device)

    del duplex, thinker, processor, root_config
    gc.collect()
    torch.cuda.empty_cache()


def _audio_batch(loaded: LoadedQwen, count: int):
    samples = np.arange(32_000, dtype=np.float32) / 16_000
    clips = [
        np.sin(2 * np.pi * frequency * samples).astype(np.float32)
        for frequency in (220.0, 440.0)[:count]
    ]
    prompt = (
        loaded.processor.audio_bos_token
        + loaded.processor.audio_token
        + loaded.processor.audio_eos_token
    )
    extracted = loaded.processor(
        text=[prompt] * count,
        audio=clips,
        sampling_rate=16_000,
        padding=True,
        return_tensors="pt",
    )
    feature_mask = extracted.feature_attention_mask.to(loaded.device)
    timeline_length = loaded.duplex.frames_per_chunk
    return {
        "text_ids": torch.zeros(
            (count, timeline_length), dtype=torch.long, device=loaded.device
        ),
        "text_mask": torch.zeros(
            (count, timeline_length), dtype=torch.bool, device=loaded.device
        ),
        "control_ids": torch.zeros(
            (count, timeline_length), dtype=torch.long, device=loaded.device
        ),
        "control_mask": torch.zeros(
            (count, timeline_length), dtype=torch.bool, device=loaded.device
        ),
        "attention_mask": torch.ones(
            (count, timeline_length), dtype=torch.long, device=loaded.device
        ),
        "input_features": extracted.input_features.to(
            device=loaded.device, dtype=torch.bfloat16
        ),
        "feature_attention_mask": feature_mask,
        "preconv_feature_lengths": feature_mask.sum(dim=-1),
        "position_ids": torch.arange(timeline_length, device=loaded.device)
        .unsqueeze(0)
        .expand(count, -1),
        "use_cache": False,
        "last_position_only": True,
    }


def test_gpu_text_only_zero_contribution_parity(loaded_qwen) -> None:
    loaded = loaded_qwen
    bos = loaded.thinker.config.bos_token_id
    text_ids = torch.tensor([[bos, 9707, 0]], device=loaded.device)
    attention_mask = torch.ones_like(text_ids)
    position_ids = torch.arange(3, device=loaded.device).unsqueeze(0)

    with torch.inference_mode():
        base = loaded.thinker(
            input_ids=text_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        duplex = loaded.duplex(
            text_ids=text_ids,
            text_mask=torch.ones_like(text_ids, dtype=torch.bool),
            control_ids=torch.zeros_like(text_ids),
            control_mask=torch.zeros_like(text_ids, dtype=torch.bool),
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )

    torch.testing.assert_close(duplex.logits, base.logits, rtol=0, atol=0)


def test_gpu_real_two_second_audio_forward(loaded_qwen) -> None:
    with torch.inference_mode():
        output = loaded_qwen.duplex(**_audio_batch(loaded_qwen, 1))

    assert output.lexical_hidden_states.shape == (1, 50, 2048)
    assert output.logits.shape == (1, 1, 151_936)


def test_gpu_batch_two_audio_forward(loaded_qwen) -> None:
    with torch.inference_mode():
        output = loaded_qwen.duplex(**_audio_batch(loaded_qwen, 2))

    assert output.lexical_hidden_states.shape == (2, 50, 2048)
    assert output.logits.shape == (2, 1, 151_936)


def test_gpu_no_grad_peak_vram(loaded_qwen) -> None:
    torch.cuda.reset_peak_memory_stats(loaded_qwen.device)
    with torch.inference_mode():
        output = loaded_qwen.duplex(**_audio_batch(loaded_qwen, 1))
    torch.cuda.synchronize(loaded_qwen.device)

    allocated = torch.cuda.max_memory_allocated(loaded_qwen.device)
    reserved = torch.cuda.max_memory_reserved(loaded_qwen.device)
    assert not output.lexical_hidden_states.requires_grad
    assert allocated > 0
    assert reserved >= allocated
    print(f"peak CUDA bytes: allocated={allocated}, reserved={reserved}")
