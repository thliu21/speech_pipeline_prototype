import asyncio
import json
import math
import wave
from pathlib import Path

from speech_proto.benchmark_suite import (
    ExpectedSegment,
    ManifestItem,
    boundary_metrics,
    build_synthetic_streams,
    final_segments_from_events,
    load_local_recordings,
    read_manifest,
    run_stream_item,
    segment_wer,
    tail_loss_flags,
    write_manifest,
)


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.jsonl"
    item = ManifestItem(
        id="sample",
        dataset="unit",
        audio_path="sample.wav",
        reference_text="hello world",
        expected_segments=[ExpectedSegment("hello", 0, 500), ExpectedSegment("world", 900, 1300)],
    )

    write_manifest([item], path)

    assert read_manifest(path) == [item]


def test_load_local_recordings_uses_reference_metadata(tmp_path):
    wav_path = tmp_path / "clip.wav"
    wav_path.write_bytes(b"RIFF")
    metadata = {
        "id": "clip",
        "path": str(wav_path),
        "reference_text": "hello world",
        "started_at": "2026-05-25T00:00:00",
    }
    (tmp_path / "clip.json").write_text(json.dumps(metadata), encoding="utf-8")

    items = load_local_recordings(tmp_path, limit=10)

    assert len(items) == 1
    assert items[0].id == "clip"
    assert items[0].reference_text == "hello world"


def test_boundary_metrics_match_with_tolerance():
    expected = [
        ExpectedSegment("one", 0, 1000),
        ExpectedSegment("two", 1400, 2200),
        ExpectedSegment("three", 2500, 3200),
    ]
    finals = [{"end_ms": 950}, {"end_ms": 2700}, {"end_ms": 3200}]

    metrics = boundary_metrics(expected, finals, tolerance_ms=100)

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_boundary_metrics_accepts_endpoint_inside_inter_segment_gap():
    expected = [
        ExpectedSegment("one", 0, 4815),
        ExpectedSegment("two", 5615, 9000),
    ]
    finals = [{"end_ms": 5900}, {"end_ms": 9000}]

    metrics = boundary_metrics(expected, finals, tolerance_ms=800)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_final_segments_from_events_keeps_latest_revision():
    finals = [
        {"segment_id": 1, "revision": 1, "text": "hello", "end_ms": 1000},
        {"segment_id": 1, "revision": 2, "text": "hello world", "end_ms": 1300},
        {"segment_id": 2, "revision": 1, "text": "again", "end_ms": 2400},
    ]

    collapsed = final_segments_from_events(finals)

    assert [item["text"] for item in collapsed] == ["hello world", "again"]


def test_segment_wer_penalizes_bad_segmentation():
    expected = [ExpectedSegment("hello world"), ExpectedSegment("again")]
    finals = [{"text": "hello"}, {"text": "world again"}]

    assert segment_wer(expected, finals) == 2 / 3


def test_tail_loss_flags_detect_first_last_and_plural_loss():
    flags = tail_loss_flags("Many animals", "animals animal")

    assert flags == {
        "first_word_missing": True,
        "last_word_missing": True,
        "tail_s_loss": True,
    }


def test_run_stream_item_with_mock_pipeline(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_tone_wav(wav_path, duration_sec=0.4)
    item = ManifestItem(
        id="tone",
        dataset="unit",
        audio_path=str(wav_path),
        reference_text="mock final transcript",
        expected_segments=[ExpectedSegment("mock final transcript", 0, 400)],
    )
    config = {
        "pipeline": {
            "asr_mode": "mock",
            "denoise": "off",
            "vad_threshold": 0.001,
            "speech_start_frames": 1,
            "pre_roll_ms": 10,
            "soft_end_ms": 50,
            "silence_end_ms": 100,
            "sentence_mode": "raw",
        }
    }

    result, events = asyncio.run(run_stream_item(item, config))

    assert result.id == "tone"
    assert result.segment_count == 1
    assert result.wall_rtf is not None
    assert any(event["type"] == "transcript" for event in events)


def test_build_synthetic_streams_adds_expected_boundaries(tmp_path):
    source = []
    for index in range(3):
        path = tmp_path / f"{index}.wav"
        _write_tone_wav(path, duration_sec=0.1)
        source.append(
            ManifestItem(
                id=str(index),
                dataset="unit",
                audio_path=str(path),
                reference_text=f"sentence {index}",
            )
        )

    synthetic = build_synthetic_streams(source, tmp_path / "synthetic", count=1, gap_ms_values=[800])

    assert len(synthetic) == 1
    assert Path(synthetic[0].audio_path).is_file()
    assert [segment.start_ms for segment in synthetic[0].expected_segments] == [0, 900, 1800]
    assert [segment.end_ms for segment in synthetic[0].expected_segments] == [100, 1000, 1900]


def _write_tone_wav(path: Path, duration_sec: float, sample_rate: int = 16000) -> None:
    frames = []
    for index in range(int(duration_sec * sample_rate)):
        value = int(8000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        frames.append(value.to_bytes(2, "little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"".join(frames))
