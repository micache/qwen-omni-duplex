"""Export the verified one-sample stream as small, static demo assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from duplex.turn_packed import load_local_taste


SAMPLE_ID = "read_aloud_012247"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--taste-root", type=Path, default=Path("data/TASTE-IF-SFT-48K"))
    parser.add_argument("--output", type=Path, default=Path("demo"))
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines()]
    if report.get("sample_id") != SAMPLE_ID or report.get("aligned_stream_gate") != "PASS":
        raise ValueError("Expected a passing training-audio stream for the selected sample.")
    if len(rows) != report["training_frames"] or [row["event_id"] for row in rows] != report["training_audio_stream"]["event_ids"]:
        raise ValueError("The stream trace does not match the report.")
    if any(row["grammar_mask_changed_raw_argmax"] for row in rows):
        raise ValueError("The saved stream contains grammar-forced decisions.")
    if "".join(row["decoded_delta"] for row in rows) != report["reference_text"]:
        raise ValueError("The stream text does not match the reference.")

    source = load_local_taste(args.taste_root, split="dev")
    matches = [i for i, value in enumerate(source.dataset["idx"]) if value == SAMPLE_ID]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {SAMPLE_ID} row in TASTE dev.")
    sample = source[matches[0]]
    input_text = source.dataset[matches[0]]["message"][0]["text"]
    if sample.assistant_text != report["reference_text"]:
        raise ValueError("The local dataset text differs from the remote report.")
    args.output.mkdir(parents=True, exist_ok=True)
    sf.write(args.output / "input.wav", sample.user_waveform, sample.sample_rate_hz)

    waveform = sample.user_waveform
    peaks = [
        round(float(np.sqrt(np.mean(np.square(block, dtype=np.float64)))), 5)
        for block in np.array_split(waveform, 84)
    ]
    scale = max(peaks) or 1.0
    clock = 0.0
    events = []
    for frame, row in enumerate(rows):
        clock = max(clock, float(row["available_time_s"])) + float(row["compute_ms"]) / 1000
        events.append({
            "frame": frame,
            "chunk": row["chunk_index"],
            "kind": row["event_type"],
            "delta": row["decoded_delta"],
            "audioTime": round(float(row["audio_time_s"]), 3),
            "emitTime": round(clock, 3),
        })
    payload = {
        "sampleId": SAMPLE_ID,
        "inputText": input_text,
        "referenceText": sample.assistant_text,
        "userSeconds": round(len(waveform) / sample.sample_rate_hz, 4),
        "trainingSeconds": round(len(sample.input_waveform) / sample.sample_rate_hz, 4),
        "frameRate": 25,
        "chunkSeconds": 2,
        "waveform": [round(max(0.08, peak / scale), 3) for peak in peaks],
        "events": events,
    }
    (args.output / "sample.js").write_text(
        "// Recorded model trace; see scripts/export_demo_sample.py.\n"
        "window.DUPLEX_DEMO = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"Exported {len(events)} frames, {len(waveform)} audio samples, replay end {clock:.2f}s")


if __name__ == "__main__":
    main()
