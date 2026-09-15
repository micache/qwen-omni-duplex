"""Print a compact inspection of one local DailyTalkContiguous sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duplex.dataset import (
    assign_split,
    find_turn_boundaries,
    load_conversation,
    read_manifest,
    sample_window_metadata,
)
from duplex.timeline import (
    ControlTokenIds,
    build_window_timeline,
    format_timeline_table,
)


def _word_preview(words, limit: int = 8) -> str:
    preview = " ".join(word.text for word in words[:limit])
    if len(words) > limit:
        preview += " …"
    return preview or "<none>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Path to dailytalk.jsonl.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--random-windows", type=int, default=2)
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="Print the Session 04 frame/event table for one proposed window.",
    )
    parser.add_argument(
        "--window-index",
        type=int,
        default=0,
        help="Proposed window to use with --timeline.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Local tokenizer directory; network downloads are disabled.",
    )
    parser.add_argument("--idle-token-id", type=int)
    parser.add_argument("--start-token-id", type=int)
    parser.add_argument("--stop-token-id", type=int)
    parser.add_argument("--thinker-vocab-size", type=int)
    parser.add_argument("--thinker-bos-token-id", type=int)
    parser.add_argument("--frame-rate-hz", type=int, default=25)
    args = parser.parse_args()

    entries = read_manifest(args.manifest)
    try:
        entry = entries[args.index]
    except IndexError as error:
        raise SystemExit(
            f"Sample index {args.index} is outside [0, {len(entries) - 1}]."
        ) from error
    record = load_conversation(entry, dataset_root=args.manifest.parent)
    boundaries = find_turn_boundaries(record)
    windows = sample_window_metadata(
        record, random_count=args.random_windows, seed=args.seed
    )

    print(
        f"{record.conversation_id} | split={assign_split(record.conversation_id).value} "
        f"| duration={record.duration_seconds:.3f}s"
    )
    print(
        "channels: left=assistant/model (reference only), right=user (model input); "
        f"source={record.source_sample_rate_hz}Hz, returned={record.sample_rate_hz}Hz"
    )
    print(
        f"words: assistant={len(record.assistant_words)} "
        f"[{_word_preview(record.assistant_words)}]"
    )
    print(f"       user={len(record.user_words)} [{_word_preview(record.user_words)}]")
    if boundaries:
        rendered = ", ".join(
            f"{item.from_speaker.value}->{item.to_speaker.value}@{item.time_seconds:.3f}s"
            for item in boundaries
        )
    else:
        rendered = "<none in available annotations>"
    print(f"turn boundaries: {rendered}")
    print("proposed windows:")
    for window in windows:
        boundary = (
            ""
            if window.boundary_seconds is None
            else f", boundary={window.boundary_seconds:.3f}s"
        )
        print(
            f"  {window.kind.value}: {window.start_seconds:.3f}.."
            f"{window.end_seconds:.3f}s{boundary}"
        )

    if args.timeline:
        required = {
            "--tokenizer": args.tokenizer,
            "--idle-token-id": args.idle_token_id,
            "--start-token-id": args.start_token_id,
            "--stop-token-id": args.stop_token_id,
            "--thinker-vocab-size": args.thinker_vocab_size,
            "--thinker-bos-token-id": args.thinker_bos_token_id,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit(
                "--timeline requires explicit local tokenizer/control settings: "
                + ", ".join(missing)
            )
        try:
            selected_window = windows[args.window_index]
        except IndexError as error:
            raise SystemExit(
                f"Timeline window index {args.window_index} is outside "
                f"[0, {len(windows) - 1}]."
            ) from error
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise SystemExit(
                "--timeline requires the declared transformers dependency."
            ) from error
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            local_files_only=True,
        )
        control_tokens = ControlTokenIds(
            idle=args.idle_token_id,
            start=args.start_token_id,
            stop=args.stop_token_id,
            thinker_vocab_size=args.thinker_vocab_size,
        )
        timeline = build_window_timeline(
            record,
            selected_window,
            tokenizer=tokenizer,
            control_tokens=control_tokens,
            thinker_bos_token_id=args.thinker_bos_token_id,
            frame_rate_hz=args.frame_rate_hz,
        )
        print("timeline:")
        print(format_timeline_table(timeline))


if __name__ == "__main__":
    main()
