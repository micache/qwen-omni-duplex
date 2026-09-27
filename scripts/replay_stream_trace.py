"""Replay a verified saved streaming trace in a plain terminal."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path


def load_trace(report_path: Path, trace_path: Path) -> tuple[dict, list[dict]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    stream = report["training_audio_stream"]
    if report["training_overfit_gate"] != "PASS" or report["aligned_stream_gate"] != "PASS":
        raise ValueError("The saved model did not pass both training-audio gates.")
    if len(rows) != report["training_frames"]:
        raise ValueError("Trace length does not match the training timeline.")
    if [row["event_id"] for row in rows] != stream["event_ids"]:
        raise ValueError("Trace decisions do not match the saved report.")
    if [row["chunk_index"] * 50 + row["frame_index"] for row in rows] != list(range(len(rows))):
        raise ValueError("Trace frame positions are not contiguous.")
    if any(abs(row["audio_time_s"] - index / 25) > 0.001 for index, row in enumerate(rows)):
        raise ValueError("Trace positions are not on the validated 25 Hz timeline.")
    if any(row["sample_id"] != f"{report['sample_id']}-training-audio" for row in rows):
        raise ValueError("Trace sample ID does not match the report.")
    if any(row["grammar_mask_changed_raw_argmax"] for row in rows):
        raise ValueError("Some displayed decisions were changed by the grammar mask.")
    if "".join(row["decoded_delta"] for row in rows) != report["reference_text"]:
        raise ValueError("Generated text does not match the reference.")
    return report, rows


def screen_lines(report: dict, rows: list[dict], frame: int | None) -> list[str]:
    sample = report["sample_id"]
    header = [
        f"$ replay saved stream: {sample}",
        "Qwen2.5-Omni-3B Thinker + LoRA | text only",
        "25 Hz event-timeline replay; NOT live inference or measured latency",
        "input: user audio, then silence (reference speech in video is for comparison)",
        "",
    ]
    if frame is None:
        return header + ["waiting for the first 2 s audio chunk ...", "", "event: -", "text: "]
    row = rows[frame]
    seen = rows[: frame + 1]
    counts = Counter(event["event_type"] for event in seen)
    answer = "".join(event["decoded_delta"] for event in seen)
    event = row["event_type"]
    shown = f"TEXT {row['decoded_delta']!r}" if event == "TEXT" else event
    recent = "  ".join(event["event_type"] for event in seen[-6:])
    return header + [
        f"frame {frame + 1:03d}/{len(rows)}    audio t={row['audio_time_s']:05.2f}s    chunk {row['chunk_index'] + 1:02d}",
        f"event: {shown}",
        f"counts: IDLE {counts['IDLE']:3d}  START {counts['START']}  TEXT {counts['TEXT']:2d}  STOP {counts['STOP']}",
        f"recent: {recent}",
        "",
        "text:",
    ] + (textwrap.wrap(answer, width=72, break_long_words=False) or [""])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    report, rows = load_trace(args.report, args.trace)
    start = time.monotonic()
    for frame in [None, *range(len(rows))]:
        target = 0.0 if frame is None else 2.0 + frame / 25.0
        time.sleep(max(0.0, start + target - time.monotonic()))
        sys.stdout.write("\033[2J\033[H" + "\n".join(screen_lines(report, rows, frame)) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
