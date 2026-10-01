from types import SimpleNamespace
import json

import numpy as np
import torch
import pytest

from benchmarks.voicebench import RunSettings, run_benchmark
from duplex.streaming import QwenDuplexStreamer
from benchmarks.voicebench_batch import PackedStaticLayer


def test_empty_answers_are_kept_in_benchmark_denominator(tmp_path):
    settings = RunSettings(model_mode="base", data="commoneval", split="test", modality="audio")
    backend = SimpleNamespace(generate_audio=lambda *args, **kwargs: "")
    output = tmp_path / "answers.jsonl"
    result = run_benchmark(settings, output=output,
        dataset_loader=lambda _: [{"prompt": "Question", "audio": None}],
        backend_loader=lambda _: backend)
    assert result["written"] == 1
    assert json.loads(output.read_text())["response"] == ""


def test_streaming_token_cap_stops_without_forcing_stop_or_adding_tail():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return " ".join(str(x) for x in ids)

    class Streamer(QwenDuplexStreamer):
        @property
        def device(self):
            return torch.device("cpu")

        def _audio_features(self, chunk):
            return torch.zeros(1, 50, 2)

        def _step(self, **kwargs):
            frame = kwargs["sequence_length"]
            token = 2 if frame == 0 else 7 + frame - 1
            return token, token, False, torch.zeros(1, 1, 2), None, 0.0

    model = SimpleNamespace(timeline=SimpleNamespace(chunk_seconds=2.0, frame_rate_hz=25),
        control_tokens=SimpleNamespace(idle=1, start=2, stop=3), eval=lambda: None)
    streamer = Streamer(model, SimpleNamespace(tokenizer=Tokenizer()), max_new_tokens=2, max_silent_chunks=42)
    result = streamer.run(np.zeros(32000, dtype=np.float32), sample_id="cap", context_token_ids=[])
    assert [r.event_id for r in result.trace] == [2, 7, 8]
    assert result.stop_reason == "max_new_tokens"
    assert result.timed_out
    assert result.processed_silent_chunks == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FlashAttention parity")
def test_cached_flash_ignores_future_slots_and_matches_valid_prefix():
    torch.manual_seed(17)
    q = torch.randn(4, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(4, 256, 2, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    lengths = [17, 42, 203, 255]
    for row, length in enumerate(lengths):
        k[row, length:] = 1000
        v[row, length:] = 1000
    offsets = torch.arange(5, device="cuda", dtype=torch.int32)
    actual = torch.ops.aten._flash_attention_forward(q.reshape(4, 16, 128),
        k.reshape(1024, 2, 128), v.reshape(1024, 2, 128), offsets, offsets * 256,
        1, 256, 0.0, False, False,
        seqused_k=torch.tensor(lengths, device="cuda", dtype=torch.int32))[0]
    for row, length in enumerate(lengths):
        expected = torch.nn.functional.scaled_dot_product_attention(
            q[row:row+1].transpose(1, 2), k[row:row+1, :length].transpose(1, 2),
            v[row:row+1, :length].transpose(1, 2), enable_gqa=True)
        torch.testing.assert_close(actual[row], expected[0, :, 0], rtol=0, atol=0)


def test_compacted_cache_preserves_selected_examples_and_history():
    old = PackedStaticLayer(12)
    keys = torch.arange(4 * 2 * 3 * 5, dtype=torch.float32).reshape(4, 2, 3, 5)
    old.update(keys, -keys)
    new = PackedStaticLayer(12)
    new.lazy_initialization(keys[:2], -keys[:2])
    new.copy_prefix_from(old, torch.tensor([2, 0]), 3)
    torch.testing.assert_close(new.keys[:, :, :3], keys[[2, 0]])
    torch.testing.assert_close(new.values[:, :, :3], -keys[[2, 0]])
    assert new.get_seq_length() == 3
    assert new.keys.transpose(1, 2).is_contiguous()
    next_key = torch.ones(2, 2, 1, 5)
    new.update(next_key, -next_key)
    torch.testing.assert_close(new.keys[:, :, :3], keys[[2, 0]])
    torch.testing.assert_close(new.keys[:, :, 3:4], next_key)
    assert new.get_seq_length() == 4


def test_voicebench_jsonl_preserves_unicode_line_separators(tmp_path):
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / "scripts"))
    from score_voicebench_suite import read
    path = tmp_path / "unicode.jsonl"
    example = {"prompt": "First\u2028second\u0085third", "response": "Four\u2029five"}
    path.write_text(json.dumps(example, ensure_ascii=False) + "\n")
    assert read(path) == [example]
