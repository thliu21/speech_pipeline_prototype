from speech_proto.asr import Transcript
from speech_proto.events import make_event


def test_event_shape():
    event = make_event("metrics", {"stage": "asr", "latest_ms": 1.2})

    assert event["type"] == "metrics"
    assert "ts_ms" in event
    assert event["payload"]["stage"] == "asr"


def test_transcript_payload_shape():
    payload = Transcript("final", "hello", 100, 900).as_payload()

    assert payload == {"type": "final", "text": "hello", "start_ms": 100, "end_ms": 900}

