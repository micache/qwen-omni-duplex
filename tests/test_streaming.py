"""Streaming stops at the token budget without fabricating control events."""

from types import SimpleNamespace

import numpy as np
import torch

from duplex.streaming import QwenDuplexStreamer

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
