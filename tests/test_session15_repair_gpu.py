"""Opt-in real adapter parity over one continuous four-chunk public span."""

import os
from pathlib import Path
import torch
import pytest

from duplex.dataset import DuplexCollator
from duplex.recovery import load_continuous_spans
from duplex.training import (_cuda_batch, _load_adapter_weights, build_training_model,
                             load_training_config)
from generate import _context_token_ids


@pytest.mark.skipif(os.environ.get("RUN_QWEN_GPU_TESTS") != "1", reason="opt-in GPU checkpoint test")
def test_real_four_chunk_batched_cached_parity():
    config = load_training_config("configs/session15_main.yaml")
    model, processor, _ = build_training_model(config)
    adapter = Path(os.environ.get("REPAIR_PARITY_ADAPTER", "outputs/session15-main-111b/selected"))
    _load_adapter_weights(model, adapter)
    model.eval()
    model.gradient_checkpointing_disable()
    spans, _ = load_continuous_spans(config, processor.tokenizer, model.control_tokens,
                                     model.base_thinker.config.bos_token_id)
    example = spans[0].normal
    context = _context_token_ids(processor.tokenizer, config["generation"]["system"], "")
    collator = DuplexCollator(audio_processor=processor, tokenizer=processor.tokenizer,
                              control_tokens=model.control_tokens,
                              thinker_bos_token_id=model.base_thinker.config.bos_token_id,
                              context_token_ids=context, interruption_probability=0)
    batch = _cuda_batch(collator([example]), model)
    embedding = model.base_thinker.get_input_embeddings()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        batched = model(**batch).logits.float()
        audio = model._restore_audio(input_features=batch["input_features"],
                                     feature_attention_mask=batch["feature_attention_mask"],
                                     preconv_feature_lengths=batch["preconv_feature_lengths"],
                                     audio_chunk_counts=batch["audio_chunk_counts"],
                                     current_attention=batch["attention_mask"],
                                     timeline_length=200, hidden_width=embedding.weight.shape[1],
                                     device=embedding.weight.device, dtype=embedding.weight.dtype)
        prefix = embedding(batch["context_ids"])
        prefill = model.base_thinker.model(inputs_embeds=prefix,
                                          attention_mask=torch.ones((1, len(context)), device=prefix.device, dtype=torch.bool),
                                          position_ids=torch.arange(len(context), device=prefix.device)[None, :],
                                          use_cache=True, return_dict=True)
        cache = prefill.past_key_values
        for frame in (0, 1, 25, 49, 50, 51, 99, 100, 101, 149, 150, 151, 199):
            # Advance every intervening frame using its target previous event.
            start = 0 if frame == 0 else previous_frame + 1
            for current in range(start, frame + 1):
                text = embedding(batch["text_ids"][:, current:current + 1]) * batch["text_mask"][:, current:current + 1, None]
                control = embedding(batch["control_ids"][:, current:current + 1]) * batch["control_mask"][:, current:current + 1, None]
                fused = text + control + audio[:, current:current + 1]
                result = model.base_thinker.model(inputs_embeds=fused,
                                                 attention_mask=torch.ones((1, len(context) + current + 1), device=fused.device, dtype=torch.bool),
                                                 position_ids=torch.tensor([[len(context) + current]], device=fused.device),
                                                 past_key_values=cache, use_cache=True, return_dict=True)
                cache = result.past_key_values
            cached = model.base_thinker.lm_head(result.last_hidden_state[:, -1:]).float()[0, 0]
            reference = batched[0, frame]
            assert int(cached.argmax()) == int(reference.argmax()), f"selected ID differs at frame {frame}"
            # BF16 attention kernel accumulation and cache path can differ by a few ulps.
            print(f"frame {frame}: max_abs_logit_delta={(cached-reference).abs().max().item():.3f}")
            torch.testing.assert_close(cached, reference, rtol=0.03, atol=1.0)
            previous_frame = frame
