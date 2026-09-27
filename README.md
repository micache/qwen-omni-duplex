# Qwen Omni Duplex

I am exploring whether a speech model can decide *when* to write as well as *what* to write. This project uses the [Qwen2.5-Omni-3B Thinker](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) to read incoming audio and emit a 25 Hz sequence of `IDLE`, `START`, text tokens, and `STOP`. It generates text, not speech. The longer-term question is whether this event stream can support a full-duplex assistant without a separate turn-taking model.

## Examples

The clips below replay saved frame-by-frame model decisions. They are **training-example traces**, paced at 25 Hz for viewing—not live inference or a latency benchmark. In each video, the spoken prompt is followed by the dataset's reference speech so you can hear the answer; the model receives silence during that response interval and produces text only.

### Countryside

“Say it slowly: describe a peaceful countryside in one sentence.”

[Prompt MP3](demo/input.mp3) · [Video with audio](demo/stream.mp4) · [Event trace](demo/training_audio_stream.jsonl)

[![Countryside text stream](demo/stream.gif)](demo/stream.mp4)

### Counting to thirty

“Count from one to thirty quickly.” This one has a longer, 69-token text sequence.

[Prompt MP3](demo/counting_input.mp3) · [Video with audio](demo/counting_stream.mp4) · [Event trace](demo/counting_training_audio_stream.jsonl)

[![Counting text stream](demo/counting_stream.gif)](demo/counting_stream.mp4)

## How it works

Each example is laid out on a fixed 25 Hz timeline. The audio is processed in two-second chunks; a text/control embedding and the current audio representation are added in the Thinker's hidden space. The model predicts the next event with a weighted loss, using LoRA adapters in the text decoder while the audio encoder stays frozen. `IDLE` keeps the stream listening, `START` begins a text response, and `STOP` ends it. The dataset view is [TASTE-IF-SFT-48K](https://huggingface.co/datasets/Jaylin0418/TASTE-IF-SFT-48K).

The implementation is in [`duplex/turn_packed.py`](duplex/turn_packed.py) (timeline/data), [`duplex/model.py`](duplex/model.py) (fusion and loss), and [`duplex/streaming.py`](duplex/streaming.py) (generation). The training recipe is [`configs/turn_packed_main.yaml`](configs/turn_packed_main.yaml); experiment details and limitations are in [`notes/experiments.md`](notes/experiments.md).

To replay either saved trace in a terminal, from the repository root:

```bash
.venv/bin/python scripts/replay_stream_trace.py --report demo/counting_report.json --trace demo/counting_training_audio_stream.jsonl
```

Independent student research project. Repository code: Apache 2.0. The Qwen model and TASTE data retain their own licenses.
