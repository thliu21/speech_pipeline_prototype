from fastapi.testclient import TestClient

from speech_proto.app import create_app
from speech_proto.pipeline import PipelineController


def test_websocket_status_event_connects():
    app = create_app(PipelineController(default_asr="mock", default_denoise="off"))

    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as websocket:
            event = websocket.receive_json()

    assert event["type"] == "status"
    assert event["payload"]["running"] is False

