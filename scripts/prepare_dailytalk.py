"""Validate DailyTalkContiguous and write a small split index."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duplex.dataset import DATASET_ID, assign_split, load_conversation, read_manifest


def download_snapshot(dataset_root: Path, *, revision: str | None = None) -> Path:
    """Download only when called from the explicit ``--download`` CLI path."""

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            local_dir=dataset_root,
            revision=revision,
        )
    )


def prepare_index(
    manifest_path: Path,
    output_path: Path,
    *,
    expected_sample_rate_hz: int | None = None,
) -> dict[str, int]:
    """Validate every local conversation and write non-audio split metadata."""

    if manifest_path.resolve() == output_path.resolve():
        raise ValueError("Output index must not overwrite the source manifest.")
    entries = read_manifest(manifest_path)
    counts = {"train": 0, "validation": 0, "test": 0}
    source_sample_rate = expected_sample_rate_hz
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for entry in entries:
            record = load_conversation(
                entry,
                dataset_root=manifest_path.parent,
                expected_source_sample_rate_hz=source_sample_rate,
            )
            if source_sample_rate is None:
                source_sample_rate = record.source_sample_rate_hz
            split = assign_split(record.conversation_id)
            counts[split.value] += 1
            json.dump(
                {
                    "conversation_id": record.conversation_id,
                    "path": entry.relative_audio_path.as_posix(),
                    "duration": record.duration_seconds,
                    "source_sample_rate_hz": record.source_sample_rate_hz,
                    "split": split.value,
                    "assistant_word_count": len(record.assistant_words),
                    "user_word_count": len(record.user_words),
                },
                output,
                sort_keys=True,
            )
            output.write("\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Directory containing dailytalk.jsonl and data_stereo/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL split/validation index (never contains waveforms).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly download the approximately 13.8 GB Hugging Face snapshot.",
    )
    parser.add_argument("--revision", help="Optional Hugging Face snapshot revision.")
    parser.add_argument("--expected-sample-rate", type=int)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if args.download:
        dataset_root = download_snapshot(dataset_root, revision=args.revision)
    counts = prepare_index(
        dataset_root / "dailytalk.jsonl",
        args.output,
        expected_sample_rate_hz=args.expected_sample_rate,
    )
    print(
        f"Validated {sum(counts.values())} conversations: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
