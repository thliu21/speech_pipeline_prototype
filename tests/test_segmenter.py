from speech_proto.audio_utils import BYTES_PER_FRAME
from speech_proto.segmenter import VadSegmenter


FRAME = b"\0" * BYTES_PER_FRAME


def test_vad_segmenter_uses_preroll_and_detects_end():
    segmenter = VadSegmenter(speech_start_frames=3, silence_end_ms=30, pre_roll_ms=50)

    events = []
    for is_speech in [False, False, True, True, True]:
        events.extend(segmenter.process(FRAME, is_speech))

    assert events[0].kind == "speech_start"
    assert len(events[0].frames) == 5
    assert segmenter.in_speech is True

    events = []
    for is_speech in [False, False, False]:
        events.extend(segmenter.process(FRAME, is_speech))

    assert [event.kind for event in events][-1] == "speech_end"
    assert segmenter.in_speech is False


def test_flush_returns_end_when_active():
    segmenter = VadSegmenter(speech_start_frames=1)
    segmenter.process(FRAME, True)

    event = segmenter.flush()

    assert event is not None
    assert event.kind == "speech_end"


def test_short_silence_emits_soft_endpoint_without_final_end():
    segmenter = VadSegmenter(speech_start_frames=1, silence_end_ms=50, soft_end_ms=20, pre_roll_ms=10)
    segmenter.process(FRAME, True)

    events = []
    for is_speech in [False, False]:
        events.extend(segmenter.process(FRAME, is_speech))

    assert "speech_maybe_end" in [event.kind for event in events]
    assert "speech_end" not in [event.kind for event in events]
    assert segmenter.in_speech is True

    events = segmenter.process(FRAME, True)

    assert "speech_end" not in [event.kind for event in events]
    assert segmenter.in_speech is True


def test_long_silence_finalizes_sentence_after_soft_endpoint():
    segmenter = VadSegmenter(speech_start_frames=1, silence_end_ms=50, soft_end_ms=20, pre_roll_ms=10)
    segmenter.process(FRAME, True)

    events = []
    for _ in range(5):
        events.extend(segmenter.process(FRAME, False))

    assert [event.kind for event in events].count("speech_maybe_end") == 1
    assert [event.kind for event in events][-1] == "speech_end"
    assert segmenter.in_speech is False
