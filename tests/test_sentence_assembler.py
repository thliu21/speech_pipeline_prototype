from speech_proto.sentence_assembler import RawSentenceBoundaryEngine, SentenceAssembler
from speech_proto.transcript_assembler import TranscriptUpdate


class FakeSentenceEngine:
    name = "fake"

    def __init__(self, outputs: list[list[str]]) -> None:
        self.outputs = outputs
        self.inputs: list[str] = []

    def split(self, text: str) -> list[str]:
        self.inputs.append(text)
        return self.outputs.pop(0)


def final_update(text: str, start_ms: int, end_ms: int, segment_id: int = 1, op: str = "append") -> TranscriptUpdate:
    return TranscriptUpdate(
        type="final",
        text=text,
        start_ms=start_ms,
        end_ms=end_ms,
        segment_id=segment_id,
        revision=1,
        op=op,
    )


def test_sentence_assembler_appends_model_sentences():
    assembler = SentenceAssembler(FakeSentenceEngine([["Hello world.", "This is a test."]]))

    updates = assembler.process(final_update("hello world this is a test", 0, 2000))

    assert [update.op for update in updates] == ["append", "append"]
    assert [update.text for update in updates] == ["Hello world.", "This is a test."]
    assert assembler.transcript == "Hello world. This is a test."


def test_sentence_assembler_replaces_only_recent_sentence():
    engine = FakeSentenceEngine(
        [
            ["Hello world."],
            ["Hello world.", "This is a test."],
            ["Hello world changed.", "This is a better test."],
        ]
    )
    assembler = SentenceAssembler(engine)

    first = assembler.process(final_update("hello world", 0, 1000))
    second = assembler.process(final_update("this is a test", 1100, 2000, segment_id=2))
    third = assembler.process(final_update("this is a better test", 1100, 2400, segment_id=2, op="replace"))

    assert first[0].op == "append"
    assert [update.op for update in second] == ["replace", "append"]
    assert [update.op for update in third] == ["replace"]
    assert third[0].segment_id == second[-1].segment_id
    assert assembler.transcript == "Hello world. This is a better test."


def test_sentence_assembler_partial_passthrough_has_sentence_protocol_fields():
    assembler = SentenceAssembler(FakeSentenceEngine([]))

    updates = assembler.process(
        TranscriptUpdate("partial", "hello", 100, 200, segment_id=4, revision=0, op="partial")
    )

    assert len(updates) == 1
    assert updates[0].as_payload() == {
        "type": "partial",
        "text": "hello",
        "start_ms": 100,
        "end_ms": 200,
        "segment_id": 1,
        "revision": 0,
        "op": "partial",
    }


def test_raw_sentence_boundary_engine_returns_single_sentence():
    engine = RawSentenceBoundaryEngine()

    assert engine.split(" hello world ") == ["hello world"]
    assert engine.split(" ") == []
