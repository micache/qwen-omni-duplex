"""Batched inference for this experiment: native Thinker and empty-context duplex.

No merged weights, sampling, question transcript, reference, or forced STOP.
Only the final-position vocabulary projection is needed during generation.
"""
from __future__ import annotations

import numpy as np
import torch
from types import MethodType
from transformers.cache_utils import Cache, StaticLayer

from duplex.streaming import QwenDuplexStreamer, pad_audio_to_chunk_boundary


class PackedStaticLayer(StaticLayer):
    """Same in-place cache updates, with contiguous per-example FlashAttention storage."""
    def lazy_initialization(self, keys, values):
        self.dtype, self.device = keys.dtype, keys.device
        self.batch_size, self.num_heads = keys.shape[:2]
        self.k_head_dim, self.v_head_dim = keys.shape[-1], values.shape[-1]
        self.keys = torch.zeros(self.batch_size, self.max_cache_len, self.num_heads,
            self.k_head_dim, device=self.device, dtype=self.dtype).transpose(1, 2)
        self.values = torch.zeros(self.batch_size, self.max_cache_len, self.num_heads,
            self.v_head_dim, device=self.device, dtype=self.dtype).transpose(1, 2)
        self.cumulative_length = self.cumulative_length.to(self.device)
        self.is_initialized = True

    def copy_prefix_from(self, other, indices, length):
        self.keys[:, :, :length].copy_(other.keys[:, :, :length].index_select(0, indices))
        self.values[:, :, :length].copy_(other.values[:, :, :length].index_select(0, indices))
        self.cumulative_length.copy_(other.cumulative_length)


def cached_flash_attention(self, hidden_states, attention_mask=None, position_ids=None,
                           past_key_values=None, position_embeddings=None, **kwargs):
    """Original Qwen projections/RoPE, then FlashAttention over the valid KV prefix.

    Only the benchmark's one-frame packed cache uses this path. Other calls
    retain the original Transformers forward, including reference parity checks.
    The built-in PyTorch 2.10 varlen operation avoids copying/repeating KV heads.
    """
    if past_key_values is None or not isinstance(past_key_values.layers[self.layer_idx], PackedStaticLayer):
        return self._voicebench_original_forward(hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            position_embeddings=position_embeddings, **kwargs)
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import apply_rotary_pos_emb
    batch, query_length, _ = hidden_states.shape
    if query_length != 1 or self.training or self.sliding_window is not None:
        raise ValueError("Cached FlashAttention is limited to one-frame full-attention evaluation")
    q = self.q_proj(hidden_states).view(batch, 1, -1, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(batch, 1, -1, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(batch, 1, -1, self.head_dim).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    k, v = past_key_values.update(k, v, self.layer_idx)
    layer = past_key_values.layers[self.layer_idx]
    length = layer.max_cache_len
    offsets = torch.arange(batch + 1, device=q.device, dtype=torch.int32)
    used = layer.cumulative_length.expand(batch).contiguous().to(torch.int32)
    output = torch.ops.aten._flash_attention_forward(
        q.transpose(1, 2).reshape(batch, self.num_heads, self.head_dim),
        k.transpose(1, 2).reshape(batch * length, self.num_key_value_heads, self.head_dim),
        v.transpose(1, 2).reshape(batch * length, self.num_key_value_heads, self.head_dim),
        offsets, offsets * length, 1, length, 0.0, False, False,
        scale=self.scaling, seqused_k=used)[0]
    return self.o_proj(output.reshape(batch, 1, -1).contiguous()), None


class BatchedDuplex:
    def __init__(self, model, processor, *, max_new_tokens=2048, max_silent_chunks=42):
        if not getattr(model, "requires_empty_context", False):
            raise ValueError("This benchmark engine requires the empty-context checkpoint")
        self.model, self.processor = model, processor
        self.max_new_tokens, self.max_silent_chunks = max_new_tokens, max_silent_chunks
        self.streamer = QwenDuplexStreamer(model, processor, max_new_tokens=max_new_tokens,
                                           max_silent_chunks=max_silent_chunks, sample=False)
        self.device, self.dtype = self.streamer.device, self.streamer.dtype
        self.model.eval()
        self.model.gradient_checkpointing_disable()
        for layer in self.model.base_thinker.model.layers:
            attention = layer.self_attn
            attention._voicebench_original_forward = attention.forward
            attention.forward = MethodType(cached_flash_attention, attention)
        self.silence = None
        self.graph_step = None

    def prepare_graph(self, batch, length):
        """Capture the unchanged decoder with in-place KV updates, avoiding CPU launch overhead."""
        if self.graph_step is not None and self.graph_step[0] == (batch, length):
            self.graph_step[1].reset()
            self.graph_step[4].zero_()
            return self.graph_step
        self.graph_step = None
        torch.cuda.empty_cache()
        cache = Cache(layers=[PackedStaticLayer(length) for _ in self.model.base_thinker.model.layers])
        fused = torch.zeros(batch, 1, self.model.base_thinker.get_input_embeddings().weight.shape[1],
                            device=self.device, dtype=self.dtype)
        positions = torch.zeros(batch, 1, device=self.device, dtype=torch.long)
        mask = torch.zeros(1, 1, 1, length, device=self.device, dtype=torch.bool)
        mask[..., 0] = True
        def forward():
            output = self.model.base_thinker.model(inputs_embeds=fused,
                attention_mask={"full_attention": mask}, position_ids=positions,
                past_key_values=cache, use_cache=True, return_dict=True)
            return self.model.base_thinker.lm_head(output.last_hidden_state[:, -1])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                cache.reset()
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        cache.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            logits = forward()
        cache.reset()
        mask.zero_()
        self.graph_step = ((batch, length), cache, fused, positions, mask, graph, logits)
        return self.graph_step

    def encode_chunks(self, chunks):
        """Same full 2-second waveform/mel masks and FP32 encoder as training."""
        result = []
        for start in range(0, len(chunks), 32):
            group = chunks[start:start + 32]
            features = self.processor.feature_extractor(group, sampling_rate=16000,
                padding=True, return_attention_mask=True, return_tensors="pt")
            values = features.input_features.to(self.device)
            mask = features.get("feature_attention_mask", features.get("attention_mask")).to(self.device)
            lengths = mask.long().sum(-1)
            if not torch.all(lengths == 200):
                raise ValueError("Every prepared chunk must have 200 valid mel frames")
            audio = self.model._restore_audio(input_features=values, feature_attention_mask=mask,
                preconv_feature_lengths=lengths,
                current_attention=torch.ones((len(group), 50), dtype=torch.bool, device=self.device),
                timeline_length=50, hidden_width=self.model.base_thinker.config.text_config.hidden_size,
                device=self.device, dtype=self.dtype)
            result.extend(audio.unbind(0))
        return result

    @torch.inference_mode()
    def audit_graph(self, waveforms, steps=150):
        """Compare graph/eager logits with identical past events, isolating numeric error."""
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            rows = []
            for wave in waveforms:
                padded = pad_audio_to_chunk_boundary(wave)
                rows.append(torch.stack(self.encode_chunks([padded[i:i+32000]
                    for i in range(0, len(padded), 32000)])))
            silence = self.encode_chunks([np.zeros(32000, dtype=np.float32)])[0]
            batch = len(rows)
            _, _, graph_input, positions, mask, graph, graph_logits = self.prepare_graph(batch, 2250)
            controls = self.model.control_tokens
            previous = torch.full((batch, 1), self.model.base_thinker.config.bos_token_id,
                                  device=self.device, dtype=torch.long)
            active = torch.zeros(batch, 1, device=self.device, dtype=torch.bool)
            cache, differences, means, agreements, worst, grammar_agreements = None, [], [], [], None, []
            for frame in range(steps):
                chunk, within = divmod(frame, 50)
                audio = torch.stack([row[chunk, within] if chunk < len(row) else silence[within]
                                     for row in rows]).unsqueeze(1)
                fused = self.model.scale_audio_for_response(audio, active) + self.model.base_thinker.get_input_embeddings()(previous)
                eager = self.model.base_thinker.model(inputs_embeds=fused,
                    attention_mask=torch.ones(batch, frame+1, device=self.device, dtype=torch.bool),
                    position_ids=torch.full((batch, 1), frame, device=self.device, dtype=torch.long),
                    past_key_values=cache, use_cache=True, return_dict=True)
                cache = eager.past_key_values
                logits = self.model.base_thinker.lm_head(eager.last_hidden_state[:, -1])
                graph_input.copy_(fused)
                positions.fill_(frame)
                mask[..., frame] = True
                graph.replay()
                diff = (logits.float() - graph_logits.float()).abs()
                differences.append(float(diff.max()))
                means.append(float(diff.mean()))
                if worst is None or differences[-1] > worst["absolute_difference"]:
                    flat = int(diff.argmax())
                    row, token = divmod(flat, diff.shape[1])
                    worst = {"frame": frame, "row": row, "token": token,
                        "eager_logit": float(logits[row, token]), "graph_logit": float(graph_logits[row, token]),
                        "absolute_difference": differences[-1]}
                agreements.extend((logits.argmax(-1) == graph_logits.argmax(-1)).cpu().tolist())
                start = logits[:, controls.start].clone()
                idle = logits[:, controls.idle]
                logits[:, controls.start] = -torch.inf
                selected = torch.where(active[:, 0], logits.argmax(-1),
                    torch.where(start > idle, controls.start, controls.idle))
                graph_start, graph_idle = graph_logits[:, controls.start].clone(), graph_logits[:, controls.idle]
                graph_logits[:, controls.start] = -torch.inf
                graph_selected = torch.where(active[:, 0], graph_logits.argmax(-1),
                    torch.where(graph_start > graph_idle, controls.start, controls.idle))
                grammar_agreements.extend((selected == graph_selected).cpu().tolist())
                active = torch.where(selected[:, None] == controls.start, True,
                    torch.where(selected[:, None] == controls.stop, False, active))
                previous = selected[:, None]
            result = {"steps": steps, "batch": batch, "maximum_logit_difference": max(differences),
                      "mean_logit_difference": float(np.mean(means)), "raw_argmax_agreement": float(np.mean(agreements)),
                      "grammar_argmax_agreement": float(np.mean(grammar_agreements)), "worst_logit": worst}
            if result["mean_logit_difference"] > 0.05 or result["grammar_argmax_agreement"] < 0.99:
                raise RuntimeError(f"Graph precision parity failed: {result}")
            return result

    @torch.inference_mode()
    def generate(self, waveforms, *, audit_frames=(), use_graph=True):
        # Captured casts must live inside the graph pool, rather than the
        # autocast weight cache whose tensors expire between generate calls.
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            if self.silence is None:
                self.silence = self.encode_chunks([np.zeros(32000, dtype=np.float32)])[0]
            chunks, counts = [], []
            for waveform in waveforms:
                padded = pad_audio_to_chunk_boundary(waveform)
                group = [padded[i:i + 32000] for i in range(0, len(padded), 32000)]
                counts.append(len(group))
                chunks.extend(group)
            encoded = self.encode_chunks(chunks)
            audio_rows, offset = [], 0
            for count in counts:
                audio_rows.append(torch.stack(encoded[offset:offset + count]))
                offset += count
            del chunks, encoded
            batch = len(waveforms)
            controls = self.model.control_tokens
            control_set = {controls.idle, controls.start, controls.stop}
            embedding = self.model.base_thinker.get_input_embeddings()
            active = [False] * batch
            seen_stop = [False] * batch
            done = [False] * batch
            events = [[] for _ in range(batch)]
            raw_events = [[] for _ in range(batch)]
            lexical = [[] for _ in range(batch)]
            finish = [None] * batch
            numeric_audit = [[] for _ in range(batch)]
            previous = torch.full((batch, 1), self.model.base_thinker.config.bos_token_id,
                                  dtype=torch.long, device=self.device)
            cache = None
            limit = max(counts) * 50 + self.max_silent_chunks * 50
            attention = torch.ones((batch, limit), dtype=torch.bool, device=self.device)
            graph_step = self.prepare_graph(batch, limit) if use_graph else None
            live_ids = list(range(batch))
            for frame in range(limit):
                chunk_index, within = divmod(frame, 50)
                # Finished rows continue as ignored dummy rows; no cross-example attention.
                audio = torch.stack([audio_rows[i][chunk_index, within] if chunk_index < counts[i] else self.silence[within]
                                     for i in live_ids]).unsqueeze(1)
                audio = self.model.scale_audio_for_response(audio,
                    torch.tensor([active[i] for i in live_ids], device=self.device, dtype=torch.bool)[:, None])
                # A previous event enters exactly one channel; both use the same embedding.
                # Combining the two exclusive channels preserves frame_event_inputs semantics.
                fused = audio + embedding(previous)
                if graph_step is None:
                    output = self.model.base_thinker.model(inputs_embeds=fused,
                        attention_mask=attention[:, :frame + 1],
                        position_ids=torch.full((batch, 1), frame, dtype=torch.long, device=self.device),
                        past_key_values=cache, use_cache=True, return_dict=True)
                    cache = output.past_key_values
                    logits = self.model.base_thinker.lm_head(output.last_hidden_state[:, -1])
                else:
                    _, _, graph_fused, graph_positions, graph_mask, graph, logits = graph_step
                    graph_fused.copy_(fused)
                    graph_positions.fill_(frame)
                    graph_mask[..., frame] = True
                    graph.replay()
                raw = logits.argmax(-1)
                if frame in audit_frames:
                    values, indices = logits.topk(5, dim=-1)
                    for i, v, ix in zip(live_ids, values.float().cpu().tolist(), indices.cpu().tolist()):
                        numeric_audit[i].append({"frame": frame, "top_ids": ix, "top_logits": v})
                state = torch.tensor([active[i] for i in live_ids], device=self.device, dtype=torch.bool)
                # Inactive: only IDLE/START. Active: every token except START.
                start_score, idle_score = logits[:, controls.start], logits[:, controls.idle]
                inactive = torch.where(start_score > idle_score, controls.start, controls.idle)
                logits[:, controls.start] = -torch.inf
                selected = torch.where(state, logits.argmax(-1), inactive)
                chosen_cpu, raw_cpu = torch.stack((selected, raw)).cpu().tolist()
                previous = selected[:, None]
                for i, token, raw_token in zip(live_ids, chosen_cpu, raw_cpu):
                    if done[i]:
                        continue
                    if token not in control_set and len(lexical[i]) >= self.max_new_tokens:
                        done[i], finish[i] = True, "max_new_tokens"
                        continue
                    events[i].append(token)
                    raw_events[i].append(raw_token)
                    if token == controls.start:
                        active[i] = True
                    elif token == controls.stop:
                        active[i], seen_stop[i] = False, True
                    elif token != controls.idle:
                        lexical[i].append(token)
                    input_finished = frame + 1 >= counts[i] * 50
                    in_tail = frame >= counts[i] * 50
                    boundary = (frame + 1) % 50 == 0
                    if input_finished and seen_stop[i] and not active[i] and (boundary or in_tail):
                        done[i], finish[i] = True, "STOP"
                    elif frame + 1 >= (counts[i] + self.max_silent_chunks) * 50:
                        done[i], finish[i] = True, "max_silent_chunks"
                if all(done):
                    break
                # Re-capture only when at least half the rows finish, at a chunk
                # boundary. Preserve each survivor's entire KV prefix, position,
                # previous event and response state. Finished examples stay saved.
                keep = [j for j, i in enumerate(live_ids) if not done[i]]
                if graph_step is not None and within == 49 and 0 < len(keep) <= len(live_ids) // 2:
                    survivor_indices = torch.tensor(keep, device=self.device, dtype=torch.long)
                    old_step = graph_step
                    graph_step = self.prepare_graph(len(keep), limit)
                    for old_layer, new_layer in zip(old_step[1].layers, graph_step[1].layers):
                        new_layer.copy_prefix_from(old_layer, survivor_indices, frame + 1)
                    graph_step[4][..., :frame + 1] = True
                    previous = previous.index_select(0, survivor_indices)
                    live_ids = [live_ids[j] for j in keep]
                    del old_step
            return [{"response": self.processor.tokenizer.decode(evt, skip_special_tokens=True,
                      clean_up_tokenization_spaces=False), "token_ids": ids, "event_ids": evt,
                     "raw_event_ids": raw, "finish_reason": reason,
                     "grammar_overrides": sum(a != b for a, b in zip(evt, raw)),
                     "input_frames": count * 50, **({"numeric_audit": audit} if audit else {})}
                    for ids, evt, raw, reason, count, audit in zip(lexical, events, raw_events, finish, counts, numeric_audit)]


def final_position_head(module, args):
    """Discard unused prefill logits; leave original weights and final logits intact."""
    return (args[0][:, -1:, :], *args[1:])


class BatchedNative:
    def __init__(self, model, processor, *, max_new_tokens=2048):
        self.model, self.processor, self.max_new_tokens = model, processor, max_new_tokens
        self.model.eval()
        self.processor.tokenizer.padding_side = "left"
        self.head_hook = model.lm_head.register_forward_pre_hook(final_position_head)

    @torch.inference_mode()
    def generate(self, waveforms):
        texts = [self.processor.apply_chat_template([
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "audio", "audio": wave}]}],
            tokenize=False, add_generation_prompt=True) for wave in waveforms]
        # Extract each native waveform before padding. Padding raw waveforms before
        # STFT changes short clips' boundary features and can change recognition.
        items = [self.processor(text=text, audio=[wave], padding=True, return_tensors="pt",
                use_audio_in_video=False).to(self.model.device).to(self.model.dtype)
                 for text, wave in zip(texts, waveforms)]
        length = max(item.input_ids.shape[1] for item in items)
        mel_length = max(item.feature_attention_mask.shape[1] for item in items)
        token_rows, embed_rows, mask_rows, feature_masks = [], [], [], []
        for item in items:
            audio = self.model.get_audio_features(item.input_features,
                item.feature_attention_mask, return_dict=True).last_hidden_state
            embeds = self.model.get_input_embeddings()(item.input_ids)
            _, _, audio_mask = self.model.get_placeholder_mask(item.input_ids, inputs_embeds=embeds)
            embeds = embeds.masked_scatter(audio_mask, audio.to(embeds.dtype))
            padding = length - item.input_ids.shape[1]
            token_rows.append(torch.nn.functional.pad(item.input_ids, (padding, 0), value=self.model.config.pad_token_id))
            embed_rows.append(torch.nn.functional.pad(embeds, (0, 0, padding, 0)))
            mask_rows.append(torch.nn.functional.pad(item.attention_mask, (padding, 0)))
            feature_masks.append(torch.nn.functional.pad(item.feature_attention_mask,
                (0, mel_length - item.feature_attention_mask.shape[1])))
        inputs = {"input_ids": torch.cat(token_rows), "inputs_embeds": torch.cat(embed_rows),
                  "attention_mask": torch.cat(mask_rows), "feature_attention_mask": torch.cat(feature_masks)}
        # Native generate pads finished rows until its longest answer ends.
        # Continue in blocks and remove completed rows with their cache entries.
        # Full token prefixes are retained for native generation processors.
        live = list(range(len(waveforms)))
        ids = [[] for _ in waveforms]
        used = 0
        while live and used < self.max_new_tokens:
            prefix_length = inputs["input_ids"].shape[1]
            output = self.model.generate(**inputs, do_sample=False, use_cache=True,
                max_new_tokens=min(64, self.max_new_tokens - used), return_dict_in_generate=True)
            new = output.sequences[:, prefix_length:]
            blocks = new.cpu().tolist()
            eos = self.model.config.eos_token_id
            keep = []
            for local, row in enumerate(blocks):
                if eos in row:
                    ids[live[local]].extend(row[:row.index(eos) + 1])
                else:
                    ids[live[local]].extend(row)
                    keep.append(local)
            used += new.shape[1]
            if not keep or used >= self.max_new_tokens:
                break
            selection = torch.tensor(keep, device=self.model.device)
            cache = output.past_key_values
            cache.batch_select_indices(selection)
            self.model.rope_deltas = self.model.rope_deltas.index_select(0, selection)
            inputs = {"input_ids": output.sequences.index_select(0, selection),
                "attention_mask": torch.cat([inputs["attention_mask"],
                    torch.ones_like(new)], dim=1).index_select(0, selection),
                "feature_attention_mask": inputs["feature_attention_mask"].index_select(0, selection),
                "past_key_values": cache}
            live = [live[i] for i in keep]
        outputs = []
        eos, pad = self.model.config.eos_token_id, self.model.config.pad_token_id
        for row in ids:
            if eos in row:
                row = row[:row.index(eos) + 1]
            reason = "STOP" if row and row[-1] == eos else "max_new_tokens"
            outputs.append({"response": self.processor.tokenizer.decode(row, skip_special_tokens=True,
                clean_up_tokenization_spaces=False), "token_ids": row, "finish_reason": reason})
        return outputs
