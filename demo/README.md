# One-sample replay

`index.html` is a static replay of the saved `read_aloud_012247` overfit result. It does not load the model or run inference. `sample.js` contains the 123 event records exported from `training_audio_stream.jsonl`; `input.wav` is the user instruction audio from the MIT-licensed [TASTE dataset](https://huggingface.co/datasets/Jaylin0418/TASTE-IF-SFT-48K). The model's text is shown incrementally, while the original user audio plays once. There is no generated assistant audio.

The exporter checks that the report passed the training-audio streaming gate, the trace matches its event IDs, and the decoded text equals the reference. It derives the replay clock from each chunk's `available_time_s` and each recorded step's `compute_ms`. The video is a browser recording of this page with the user audio added; it is not a live latency measurement. The training waveform contains 2.531 seconds of speech plus 2.388 seconds of silence. Streaming from speech alone failed in this experiment.

To rebuild `sample.js` and `input.wav` from a local TASTE dev snapshot and saved report/trace:

```bash
.venv/bin/python scripts/export_demo_sample.py --report /path/to/report.json --trace /path/to/training_audio_stream.jsonl
```
