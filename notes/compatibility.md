# Compatibility

## Session 06 — Qwen2.5-Omni Thinker probe passed 2026-09-11

The inference-only probe in `scripts/probe_qwen.py` passed on the repository
Python 3.11.16 environment and NVIDIA GeForce RTX 2060 SUPER (8,192 MiB,
compute capability 7.5, driver 576.88). The accepted software set is torch
2.10.0+cu126 (compiled CUDA 12.6), torchvision 0.25.0+cu126, Transformers
5.17.0, Accelerate 1.15.0, PEFT 0.20.0, bitsandbytes 0.50.2,
huggingface-hub 1.31.0, Pillow 12.3.0, and Triton 3.6.0.
`qwen-omni-utils`, `flash-attn`, and `nvcc` are absent. The exact installed
Transformers source paths and SHA-256 hashes are in
`notes/session06_probe.json`.

The accepted checkpoint is `Qwen/Qwen2.5-Omni-3B` revision
`f75b40e3da2003cdd6e1829b1f420ca70797c34e`. The snapshot contains three model
shards totaling 11,972,663,208 bytes. Loading the root checkpoint directly as
`Qwen2_5OmniThinkerForConditionalGeneration` succeeded with zero missing,
mismatched, critical-unexpected, or loader-error entries. Its 1,102 unexpected
keys are all unused root-checkpoint `talker.*` or `token2wav.*` tensors. The
full `Qwen2_5OmniForConditionalGeneration` route, `disable_talker()`, and
`.thinker` extraction are therefore not required.

The direct Thinker route is the lowest-memory verified route: it never creates
Talker or token-to-wave modules. It does create the audio and vision towers.
After audio encoding, deleting only `thinker.visual`, collecting Python
objects, and emptying the CUDA cache reduced allocation from 9,463,640,576 to
8,117,599,232 bytes while retaining the audio tower and text model. The
subsequent batch-two prefill and cached decode passed, demonstrating that this
pruning did not corrupt the Thinker.

For a raw 32,000-sample (2 s at 16 kHz) clip, the processor emits
`input_features [1,128,30000]` and `feature_attention_mask [1,30000]` with 200
valid entries. Batch two emits `[2,128,30000]` and `[2,30000]`, with mask sums
`[200,200]`. The model length helper maps each sample from 200 valid feature
positions to 100 post-convolution positions and then 50 LLM positions.
`get_audio_features` returns `BaseModelOutputWithPooling` whose
`last_hidden_state` is flattened across the batch as `[100,2048]`; splitting
dimension zero with the model-derived lengths `[50,50]` restores two
`[50,2048]` tensors. Thus 2 seconds gives exactly 50 valid 2,048-wide vectors,
but only after model-derived length restoration.

The Thinker text vocabulary is 151,936 entries. `tts_text_pad/start/end` are
151859/151860/151861 and all are in range. With `inputs_embeds`, two-dimensional
position IDs are accepted: `[2,50]` for prefill and `[2,1]` for a cached step;
the implementation expands them internally to the three multimodal position
axes. Prefill produced logits `[2,50,151936]` and a length-50 `DynamicCache`;
the cached call used attention mask `[2,51]`, produced logits `[2,1,151936]`,
and advanced the cache to 51.

BF16 is reported supported and a CUDA BF16 matrix-multiply kernel passed.
FlashAttention 2 is declared by the model class but is not usable in this
environment because `flash-attn` is not installed; no FA2 result is claimed.
The working attention path is SDPA. Torch 2.14.0+cu126 was rejected because its
eager Triton rotary-position operation required a system C compiler that this
machine does not provide; the official stable CUDA 12.6 torch 2.10.0 build
completed the same call without that runtime compiler dependency.

Peak CUDA allocator measurements in bytes:

| Operation | Peak allocated | Peak reserved |
| --- | ---: | ---: |
| Direct Thinker load | 9,446,580,736 | 9,527,361,536 |
| One batch-two audio encoding call | 9,470,790,144 | 9,554,624,512 |
| Batch-two Thinker prefill plus cached step, after vision pruning | 8,154,948,608 | 8,193,572,864 |

Cleanup reduced live allocation from 8,152,775,680 to 9,671,168 bytes and
reserved memory from 8,193,572,864 to 29,360,128 bytes. CUDA allocator totals
above physical VRAM reflect WSL managed-memory paging; 24 GB remains preferred
for practical work on this checkpoint.

Reproduction command:

```bash
.venv/bin/python scripts/probe_qwen.py --revision f75b40e3da2003cdd6e1829b1f420ca70797c34e
```

The complete machine-readable result, including raw loading keys, shapes,
source hashes, environment commands, and per-stage memory snapshots, is
`notes/session06_probe.json`.
