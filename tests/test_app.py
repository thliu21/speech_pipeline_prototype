from fastapi.testclient import TestClient

from speech_proto.app import create_app
from speech_proto.pipeline import PipelineController
from speech_proto.recordings import RecordingManager, RecordingMetadata


def test_websocket_status_event_connects():
    app = create_app(PipelineController(default_asr="mock", default_denoise="off"))

    with TestClient(app) as client:
        with client.websocket_connect("/ws/events") as websocket:
            event = websocket.receive_json()

    assert event["type"] == "status"
    assert event["payload"]["running"] is False


def test_ui_assets_are_not_cached():
    app = create_app(PipelineController(default_asr="mock", default_denoise="off"))

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_recordings_api_lists_and_updates_metadata(tmp_path):
    manager = RecordingManager(tmp_path)
    metadata = RecordingMetadata(
        id="20260525-120000-test",
        title="test",
        path=str(tmp_path / "20260525-120000-test.wav"),
        reference_text="hello",
        started_at="2026-05-25T12:00:00",
        status="ready",
    )
    (tmp_path / "20260525-120000-test.wav").write_bytes(b"RIFF")
    manager._write_metadata(metadata)
    app = create_app(PipelineController(default_asr="mock", default_denoise="off"), recording_manager=manager)

    with TestClient(app) as client:
        response = client.get("/api/recordings")
        assert response.status_code == 200
        assert response.json()["recordings"][0]["reference_text"] == "hello"

        response = client.patch(
            "/api/recordings/20260525-120000-test",
            json={"reference_text": "updated"},
        )
        assert response.status_code == 200
        assert response.json()["recording"]["reference_text"] == "updated"

        response = client.get("/api/recordings/20260525-120000-test/audio")
        assert response.status_code == 200
        assert response.content == b"RIFF"
