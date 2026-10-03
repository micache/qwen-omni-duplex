"""V1-only text timing bridge to the pinned Full-Duplex-Bench scorers.

No speech is synthesized. Word spans replace ASR, and nonempty response
envelopes replace VAD. These are text timing proxies, not official audio scores.
"""
from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from scipy.spatial.distance import jensenshannon

from duplex.streaming import EventTraceRecord, group_lexical_events_into_words

REVISION = "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"
SUBSETS = {
    "candor_pause_handling": ("pause_handling", 216, "pause.json"),
    "synthetic_pause_handling": ("pause_handling", 137, "pause.json"),
    "icc_backchannel": ("backchannel", 55, None),
    "candor_turn_taking": ("smooth_turn_taking", 119, "turn_taking.json"),
    "synthetic_user_interruption": ("user_interruption", 200, "interrupt.json"),
}
RATE = 25
CHUNK_DELAY_S = 2.0
DRIVE_FILES = {
    "candor_pause_handling": "1uls3atEz7bZ1IkVq3Rw0yQyLAjjVkklc",
    "candor_turn_taking": "1sb9mwOqDCK9BEpMb6fDVYOJU1vd5RFlb",
    "icc_backchannel": "1HttNZbi0bYHe7a-CO_9vg_z7hMM8a5h0",
    "synthetic_pause_handling": "1iV0X6z3Z9SrmJvxJ2Hkij3nb8iuWbJyv",
    "synthetic_user_interruption": "1I36wGbPtZObjqI_1h11Rb2s1ulerb65a",
}


def download_data(root: Path):
    """Fetch only the five v1 archives from the authors' public Drive release."""
    import gdown
    for subset, file_id in DRIVE_FILES.items():
        path = root / "downloads" / f"{subset}.zip"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            if gdown.download(id=file_id, output=str(path), quiet=False, resume=True) is None:
                raise RuntimeError(f"Download failed: {subset}")
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.filename.startswith("__MACOSX/"):
                    continue
                if not (root / member.filename).resolve().is_relative_to(root.resolve()):
                    raise ValueError(f"Unsafe archive member: {member.filename}")
                archive.extract(member, root)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_data(root: Path):
    """Resample only the 727 released inputs; keep annotations on their clock."""
    rows = []
    for subset, (task, expected, annotation_name) in SUBSETS.items():
        directories = sorted(p for p in (root / subset).iterdir() if p.is_dir() and p.name.isdigit())
        if len(directories) != expected:
            raise ValueError(f"{subset}: expected {expected} sample directories, found {len(directories)}")
        for directory in directories:
            source = directory / "input.wav"
            waveform, rate = sf.read(source, dtype="float32")
            if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
                raise ValueError(f"Invalid mono audio: {source}")
            original_duration = len(waveform) / rate
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            if rate != 16000:
                divisor = math.gcd(rate, 16000)
                waveform = resample_poly(waveform, 16000 // divisor, rate // divisor).astype(np.float32)
            if abs(len(waveform) / 16000 - original_duration) > 1 / 16000:
                raise ValueError(f"Resampling shifted the clock: {source}")
            output = root / "prepared" / subset / directory.name / "input.wav"
            output.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output, waveform, 16000, subtype="FLOAT")
            annotation = json.loads((directory / annotation_name).read_text()) if annotation_name else None
            if annotation is not None and (not isinstance(annotation, list) or not annotation):
                raise ValueError(f"Invalid annotation: {directory}")
            for event in annotation or []:
                start, end = event["timestamp"]
                if not (0 <= start <= end <= original_duration + 1 / rate):
                    raise ValueError(f"Annotation outside the recording: {directory}")
            rows.append({"id": f"{subset}/{directory.name}", "subset": subset, "task": task,
                "source": str(source.resolve()), "source_sha256": source_sha, "source_rate": rate,
                "audio_path": str(output.resolve()), "pcm_sha256": hashlib.sha256(waveform.tobytes()).hexdigest(),
                "samples": len(waveform), "duration_s": original_duration, "annotation": annotation})
    write_json(root / "prepared/manifest.json", {"upstream_revision": REVISION, "records": rows})
    return rows


def read_audio(row):
    wave, rate = sf.read(row["audio_path"], dtype="float32")
    if rate != 16000 or len(wave) != row["samples"] or hashlib.sha256(wave.tobytes()).hexdigest() != row["pcm_sha256"]:
        raise ValueError(f"Prepared audio identity mismatch: {row['id']}")
    return wave


def event_segments(events, tokenizer, controls):
    """Save native control times and lexical envelopes, including open answers."""
    segments, active = [], None
    specials = set(getattr(tokenizer, "all_special_ids", ()))
    decoder = SimpleNamespace(decode=lambda ids, **kw: tokenizer.decode(
        ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    for frame, event in enumerate(events):
        if event == controls.start:
            if active is not None:
                raise ValueError("Nested START")
            active = {"start_frame": frame, "stop_frame": None, "lexical": []}
        elif event == controls.stop:
            if active is None:
                raise ValueError("STOP without START")
            active["stop_frame"] = frame
            segments.append(active)
            active = None
        elif event != controls.idle:
            if active is None:
                raise ValueError("Text outside START/STOP")
            if event not in specials:
                active["lexical"].append((frame, event))
    if active is not None:
        segments.append(active)
    for segment in segments:
        records = []
        for frame, event in segment.pop("lexical"):
            records.append(EventTraceRecord("", frame // 50, frame % 50, frame / RATE,
                (frame // 50 + 1) * 2.0, 0.0, event, "TEXT", "", "active", "active",
                False, event, (frame // 50 + 1) * 2.0, False))
        words = group_lexical_events_into_words(records, decoder)
        segment["text"] = decoder.decode([record.event_id for record in records])
        segment["words"] = [{"text": word.text, "timestamp": [
            min(t.audio_time_s for t in word.token_timings),
            max(t.audio_time_s for t in word.token_timings) + 1 / RATE]} for word in words]
        segment["interval"] = ([segment["words"][0]["timestamp"][0],
            segment["words"][-1]["timestamp"][1]] if words else None)
    return segments


def timed_output(segments, horizon, *, delay_s=CHUNK_DELAY_S, crop_start=0.0):
    """A constant two-second playout buffer preserves durations and causality.

    Event f is scheduled at f/25 + 2, after its complete input chunk arrives.
    Keep output inside the original recording, just like upstream output.wav.
    Interruption transcription starts at the annotated interruption end.
    """
    chunks, intervals = [], []
    for segment in segments:
        selected = []
        for word in segment["words"]:
            start, end = [t + delay_s for t in word["timestamp"]]
            if end <= crop_start or start >= horizon:
                continue
            selected.append({"text": word["text"], "timestamp": [max(crop_start, start), min(horizon, end)]})
        chunks.extend(selected)
        if selected:
            intervals.append([selected[0]["timestamp"][0], selected[-1]["timestamp"][1]])
    return {"text": " ".join(word["text"] for word in chunks), "chunks": chunks,
            "response_intervals": intervals, "duration_s": horizon, "delay_s": delay_s,
            "crop_start_s": crop_start, "speech_output": False}


def score_output(row, output, ground_truth=None):
    """Released v1 formulas with transcript/VAD replaced by text timestamps."""
    chunks, intervals = output["chunks"], output["response_intervals"]
    if row["task"] != "backchannel":
        duration = chunks[-1]["timestamp"][1] - chunks[0]["timestamp"][0] if chunks else 0
        takeover = bool(chunks) and (duration >= 1.0 or len(chunks) > 3)
        latency = None
        if takeover and row["task"] in {"smooth_turn_taking", "user_interruption"}:
            boundary = row["annotation"][0]["timestamp"][0 if row["task"] == "smooth_turn_taking" else 1]
            latency = max(0.0, chunks[0]["timestamp"][0] - boundary)
        return {"takeover": takeover, "latency_s": latency, "word_count": len(chunks)}
    # Match upstream literally: it overwrites TOR for successive short regions,
    # appends those regions even when classified as takeovers, and breaks at >3s.
    takeover, predictions, any_takeover = False, [], False
    for start, end in intervals:
        duration = end - start
        if duration > 3.0:
            takeover, any_takeover = True, True
            break
        words = [word for word in chunks if (
            word["timestamp"][0] >= start and word["timestamp"][1] <= end) or (
            word["timestamp"][0] <= end and word["timestamp"][1] > end) or (
            word["timestamp"][0] <= start and word["timestamp"][1] > start)]
        takeover = len(words) > 2 or duration >= 1.0
        any_takeover |= takeover
        predictions.append([start, end])
    jsd = 1.0
    if predictions:
        if ground_truth is None:
            raise ValueError("Missing ICC timing reference")
        counts = np.zeros(int(output["duration_s"] / 0.2) + 1, dtype=np.float64)
        for start, end in predictions:
            for index in range(int(start / 0.2), min(int(end / 0.2), len(counts) - 1) + 1):
                counts[index] += 1
        counts += 1e-10
        counts /= counts.sum()
        resized = np.interp(np.linspace(0, 1, len(counts)), np.linspace(0, 1, len(ground_truth)), ground_truth)
        jsd = float(jensenshannon(counts, resized))
    return {"takeover": bool(takeover), "any_region_takeover": bool(any_takeover),
        "frequency_per_second": len(predictions) / output["duration_s"], "jsd": jsd,
        "counted_regions": len(predictions)}


def summarize(records, clock="causal"):
    summary = {}
    for subset, (_, expected, _) in SUBSETS.items():
        rows = [row["scores"][clock] for row in records if row["subset"] == subset]
        if not rows:
            continue
        latencies = [row["latency_s"] for row in rows if row.get("latency_s") is not None]
        result = {"samples": len(rows), "expected": expected,
            "takeovers": sum(row["takeover"] for row in rows),
            "tor": sum(row["takeover"] for row in rows) / len(rows)}
        if subset == "icc_backchannel":
            result.update(frequency_per_second=float(np.mean([row["frequency_per_second"] for row in rows])),
                jsd=float(np.mean([row["jsd"] for row in rows])),
                any_region_takeover_rate=float(np.mean([row["any_region_takeover"] for row in rows])))
        if "taking" in subset or "interruption" in subset:
            result.update(latency_s=float(np.mean(latencies)) if latencies else None,
                latency_samples=len(latencies))
        summary[subset] = result
    return summary
