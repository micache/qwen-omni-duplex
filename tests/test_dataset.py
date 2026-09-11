import json
import wave
from pathlib import Path

import numpy as np
import pytest

from duplex.dataset import (
    MAIN_SPEAKER_LABEL,
    ConversationRecord,
    ManifestEntry,
    Speaker,
    Split,
    WindowKind,
    WordSpan,
    assign_split,
    load_conversation,
    partition_by_split,
    read_manifest,
    sample_window_metadata,
)


USER_LABEL = "TEST_USER"
SPEAKER_MAP = {
    MAIN_SPEAKER_LABEL: Speaker.ASSISTANT,
    USER_LABEL: Speaker.USER,
}


def write_wav(
    path: Path,
    *,
    channels: int = 2,
    sample_rate: int = 8_000,
    seconds: int = 4,
) -> None:
    frame_count = sample_rate * seconds
    left = np.full(frame_count, 2_000, dtype="<i2")
    right = np.full(frame_count, -4_000, dtype="<i2")
    samples = left if channels == 1 else np.column_stack((left, right)).reshape(-1)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def make_fixture(
    tmp_path: Path,
    *,
    channels: int = 2,
    sample_rate: int = 8_000,
    sidecar: object | None = None,
    write_sidecar: bool = True,
) -> tuple[Path, ManifestEntry]:
    audio_dir = tmp_path / "data_stereo"
    audio_dir.mkdir()
    audio_path = audio_dir / "sample.wav"
    write_wav(audio_path, channels=channels, sample_rate=sample_rate)
    if sidecar is None:
        sidecar = {
            "alignments": [
                ["hello", [0.2, 0.5], MAIN_SPEAKER_LABEL],
                ["yes", [1.0, 1.2], USER_LABEL],
                ["great", [2.0, 2.3], MAIN_SPEAKER_LABEL],
            ]
        }
    if write_sidecar:
        audio_path.with_suffix(".json").write_text(
            json.dumps(sidecar), encoding="utf-8"
        )
    manifest = tmp_path / "dailytalk.jsonl"
    manifest.write_text(
        json.dumps({"path": "data_stereo/sample.wav", "duration": 4.0}) + "\n",
        encoding="utf-8",
    )
    return manifest, read_manifest(manifest)[0]


def test_valid_loading_maps_left_assistant_and_right_user(tmp_path: Path) -> None:
    _, entry = make_fixture(tmp_path)
    record = load_conversation(
        entry,
        dataset_root=tmp_path,
        expected_source_sample_rate_hz=8_000,
        retain_assistant_waveform=True,
        speaker_label_map=SPEAKER_MAP,
    )

    assert record.sample_rate_hz == 16_000
    assert record.source_sample_rate_hz == 8_000
    assert record.user_waveform.shape == (64_000,)
    assert record.assistant_waveform is not None
    assert record.user_waveform.mean() < 0
    assert record.assistant_waveform.mean() > 0
    assert [word.text for word in record.user_words] == ["yes"]
    assert [word.text for word in record.assistant_words] == ["hello", "great"]
    assert [word.source_index for word in record.user_words] == [1]
    assert [word.source_index for word in record.assistant_words] == [0, 2]


def test_public_main_only_schema_loads_with_empty_user_words(tmp_path: Path) -> None:
    sidecar = {
        "alignments": [["assistant", [0.2, 0.5], MAIN_SPEAKER_LABEL]]
    }
    _, entry = make_fixture(tmp_path, sidecar=sidecar)

    record = load_conversation(entry, dataset_root=tmp_path)

    assert [word.text for word in record.assistant_words] == ["assistant"]
    assert record.user_words == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"channel_roles": (Speaker.USER, Speaker.ASSISTANT)}, "left=assistant"),
        ({"expected_source_sample_rate_hz": 16_000}, "does not match expected"),
    ],
)
def test_swapped_channels_and_bad_sample_rate_are_rejected(
    tmp_path: Path, kwargs: dict, message: str
) -> None:
    _, entry = make_fixture(tmp_path)
    with pytest.raises(ValueError, match=message):
        load_conversation(
            entry,
            dataset_root=tmp_path,
            speaker_label_map=SPEAKER_MAP,
            **kwargs,
        )


def test_non_stereo_audio_is_rejected(tmp_path: Path) -> None:
    _, entry = make_fixture(tmp_path, channels=1)
    with pytest.raises(ValueError, match="exactly 2 channels"):
        load_conversation(entry, dataset_root=tmp_path, speaker_label_map=SPEAKER_MAP)


def test_bad_timestamps_are_rejected(tmp_path: Path) -> None:
    sidecar = {
        "alignments": [
            ["later", [1.0, 1.2], MAIN_SPEAKER_LABEL],
            ["earlier", [0.5, 0.7], MAIN_SPEAKER_LABEL],
        ]
    }
    _, entry = make_fixture(tmp_path, sidecar=sidecar)
    with pytest.raises(ValueError, match="not ordered"):
        load_conversation(entry, dataset_root=tmp_path)


def test_missing_sidecar_is_rejected(tmp_path: Path) -> None:
    _, entry = make_fixture(tmp_path, write_sidecar=False)
    with pytest.raises(FileNotFoundError, match="Missing JSON sidecar"):
        load_conversation(entry, dataset_root=tmp_path)


def test_manifest_rejects_unsafe_path_and_bad_duration(tmp_path: Path) -> None:
    manifest = tmp_path / "dailytalk.jsonl"
    manifest.write_text(
        '{"path":"../escape.wav","duration":4}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unsafe"):
        read_manifest(manifest)

    manifest.write_text(
        '{"path":"data_stereo/sample.wav","duration":0}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="positive and finite"):
        read_manifest(manifest)


def test_audio_duration_must_match_manifest(tmp_path: Path) -> None:
    _, entry = make_fixture(tmp_path)
    bad_entry = ManifestEntry(entry.conversation_id, entry.relative_audio_path, 3.5)
    with pytest.raises(ValueError, match="does not match WAV duration"):
        load_conversation(
            bad_entry, dataset_root=tmp_path, speaker_label_map=SPEAKER_MAP
        )


def test_unknown_sidecar_speaker_is_rejected_by_default(tmp_path: Path) -> None:
    _, entry = make_fixture(tmp_path)
    with pytest.raises(ValueError, match="unverified speaker label"):
        load_conversation(entry, dataset_root=tmp_path)


def test_deterministic_splits_and_no_split_leakage() -> None:
    conversation_ids = [f"data_stereo/{index}" for index in range(200)]
    first = [assign_split(conversation_id) for conversation_id in conversation_ids]
    second = [assign_split(conversation_id) for conversation_id in conversation_ids]
    assert first == second
    assert set(first) == set(Split)

    entries = [
        ManifestEntry(conversation_id, Path(f"{conversation_id}.wav"), 4.0)
        for conversation_id in conversation_ids
    ]
    partitions = partition_by_split(entries)
    split_ids = [
        {entry.conversation_id for entry in partitions[split]} for split in Split
    ]
    assert split_ids[0].isdisjoint(split_ids[1])
    assert split_ids[0].isdisjoint(split_ids[2])
    assert split_ids[1].isdisjoint(split_ids[2])
    assert set.union(*split_ids) == set(conversation_ids)


def test_duplicate_conversation_is_rejected_as_potential_leakage() -> None:
    entry = ManifestEntry("duplicate", Path("duplicate.wav"), 4.0)
    with pytest.raises(ValueError, match="potentially leaky split"):
        partition_by_split((entry, entry))


def test_boundary_centered_window_metadata() -> None:
    record = ConversationRecord(
        conversation_id="conversation-7",
        duration_seconds=6.0,
        sample_rate_hz=16_000,
        source_sample_rate_hz=8_000,
        user_waveform=np.zeros(96_000, dtype=np.float32),
        assistant_reference_path=Path("conversation-7.wav"),
        assistant_waveform=None,
        user_words=(WordSpan("user", 2.0, 2.3, Speaker.USER),),
        assistant_words=(
            WordSpan("assistant-one", 0.2, 0.5, Speaker.ASSISTANT),
            WordSpan("assistant-two", 4.0, 4.3, Speaker.ASSISTANT),
        ),
    )

    windows = sample_window_metadata(record, random_count=1, seed=17)

    assert windows[0].kind is WindowKind.RANDOM
    described = [
        (window.kind, window.start_seconds, window.end_seconds)
        for window in windows[1:]
    ]
    assert described == [
        (WindowKind.ASSISTANT_TO_USER, 1.0, 3.0),
        (WindowKind.USER_TO_ASSISTANT, 3.0, 5.0),
    ]
    assert [window.boundary_seconds for window in windows[1:]] == [2.0, 4.0]
    assert all(
        window.split is assign_split(record.conversation_id) for window in windows
    )
