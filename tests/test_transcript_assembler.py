from speech_proto.asr import Transcript
from speech_proto.transcript_assembler import TranscriptAssembler


def final(text: str, start_ms: int, end_ms: int) -> Transcript:
    return Transcript("final", text, start_ms, end_ms)


def test_transcript_assembler_replaces_recent_fragment_with_fuller_text():
    assembler = TranscriptAssembler(repair_window_ms=1800)

    first = assembler.process(final("sunny", 0, 900))
    second = assembler.process(final("I live in sunnyvale", 950, 1800))

    assert first is not None
    assert first.op == "append"
    assert second is not None
    assert second.op == "replace"
    assert second.segment_id == first.segment_id
    assert second.revision == 2
    assert second.text == "I live in sunnyvale"
    assert assembler.transcript == "I live in sunnyvale"


def test_transcript_assembler_merges_suffix_prefix_overlap():
    assembler = TranscriptAssembler(repair_window_ms=1800)
    assembler.process(final("I live in sunny", 0, 1000))

    update = assembler.process(final("sunnyvale", 1100, 1700))

    assert update is not None
    assert update.op == "replace"
    assert update.text == "I live in sunnyvale"
    assert assembler.transcript == "I live in sunnyvale"


def test_transcript_assembler_suppresses_recent_duplicate():
    assembler = TranscriptAssembler(repair_window_ms=1800)
    assembler.process(final("hello world", 0, 1000))

    update = assembler.process(final("hello world", 1050, 1500))

    assert update is None
    assert assembler.transcript == "hello world"


def test_transcript_assembler_appends_without_grounded_overlap():
    assembler = TranscriptAssembler(repair_window_ms=1800)
    first = assembler.process(final("hello", 0, 800))
    second = assembler.process(final("world", 900, 1500))

    assert first is not None
    assert second is not None
    assert second.op == "append"
    assert second.segment_id != first.segment_id
    assert assembler.transcript == "hello world"


def test_transcript_assembler_does_not_repair_outside_window():
    assembler = TranscriptAssembler(repair_window_ms=500)
    first = assembler.process(final("sunny", 0, 800))
    second = assembler.process(final("sunnyvale", 2000, 2800))

    assert first is not None
    assert second is not None
    assert second.op == "append"
    assert assembler.transcript == "sunny sunnyvale"


def test_transcript_assembler_partial_payload_has_protocol_fields():
    assembler = TranscriptAssembler()

    update = assembler.process(Transcript("partial", "hello", 100, 200))

    assert update is not None
    assert update.as_payload() == {
        "type": "partial",
        "text": "hello",
        "start_ms": 100,
        "end_ms": 200,
        "segment_id": 1,
        "revision": 0,
        "op": "partial",
    }
