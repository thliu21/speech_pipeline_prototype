from __future__ import annotations

import statistics
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class StageSnapshot:
    stage: str
    latest_ms: float
    avg_ms: float
    p95_ms: float
    count: int

    def as_event_payload(self) -> dict[str, float | int | str]:
        return {
            "stage": self.stage,
            "latest_ms": round(self.latest_ms, 3),
            "avg_ms": round(self.avg_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "count": self.count,
        }


class LatencyTracker:
    def __init__(self, window_size: int = 240) -> None:
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window_size))

    def record(self, stage: str, latency_ms: float) -> None:
        self._samples[stage].append(max(0.0, float(latency_ms)))

    @contextmanager
    def time_stage(self, stage: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - start) * 1000.0)

    def snapshot(self, stage: str) -> StageSnapshot:
        values = list(self._samples.get(stage, ()))
        if not values:
            return StageSnapshot(stage=stage, latest_ms=0.0, avg_ms=0.0, p95_ms=0.0, count=0)
        return StageSnapshot(
            stage=stage,
            latest_ms=values[-1],
            avg_ms=statistics.fmean(values),
            p95_ms=percentile(values, 95),
            count=len(values),
        )

    def all_snapshots(self) -> list[StageSnapshot]:
        return [self.snapshot(stage) for stage in sorted(self._samples)]

    def as_dict(self) -> dict[str, dict[str, float | int | str]]:
        return {snapshot.stage: snapshot.as_event_payload() for snapshot in self.all_snapshots()}


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percent / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

