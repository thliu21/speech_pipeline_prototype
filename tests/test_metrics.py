from speech_proto.metrics import LatencyTracker, percentile


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10], 95) == 10


def test_latency_tracker_snapshot():
    tracker = LatencyTracker()
    tracker.record("asr", 10)
    tracker.record("asr", 30)

    snapshot = tracker.snapshot("asr")

    assert snapshot.stage == "asr"
    assert snapshot.latest_ms == 30
    assert snapshot.avg_ms == 20
    assert snapshot.count == 2
    assert snapshot.as_event_payload()["stage"] == "asr"

