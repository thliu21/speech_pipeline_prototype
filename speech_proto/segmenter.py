from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .audio_utils import FRAME_MS


@dataclass(frozen=True)
class SegmentEvent:
    kind: str
    frames: list[bytes] = field(default_factory=list)
    start_ms: int | None = None
    end_ms: int | None = None


class VadSegmenter:
    def __init__(
        self,
        frame_ms: int = FRAME_MS,
        speech_start_frames: int = 3,
        silence_end_ms: int = 800,
        pre_roll_ms: int = 500,
    ) -> None:
        self.frame_ms = frame_ms
        self.speech_start_frames = speech_start_frames
        self.silence_end_frames = max(1, silence_end_ms // frame_ms)
        self.pre_roll_frames = max(1, pre_roll_ms // frame_ms)
        self.reset()

    def reset(self) -> None:
        self.in_speech = False
        self._speech_streak = 0
        self._silence_streak = 0
        self._frame_index = 0
        self._utterance_start_ms: int | None = None
        self._pre_roll: deque[bytes] = deque(maxlen=self.pre_roll_frames)

    def process(self, frame: bytes, is_speech: bool) -> list[SegmentEvent]:
        events: list[SegmentEvent] = []
        frame_start_ms = self._frame_index * self.frame_ms
        self._frame_index += 1
        self._pre_roll.append(frame)

        if is_speech:
            self._speech_streak += 1
            self._silence_streak = 0
        else:
            self._speech_streak = 0
            if self.in_speech:
                self._silence_streak += 1

        if not self.in_speech:
            if self._speech_streak >= self.speech_start_frames:
                self.in_speech = True
                preroll = list(self._pre_roll)
                start_offset_frames = len(preroll)
                self._utterance_start_ms = max(0, frame_start_ms - (start_offset_frames - 1) * self.frame_ms)
                events.append(
                    SegmentEvent(
                        kind="speech_start",
                        frames=preroll,
                        start_ms=self._utterance_start_ms,
                    )
                )
            return events

        events.append(
            SegmentEvent(
                kind="speech_frame",
                frames=[frame],
                start_ms=self._utterance_start_ms,
            )
        )
        if self._silence_streak >= self.silence_end_frames:
            end_ms = frame_start_ms + self.frame_ms
            events.append(
                SegmentEvent(
                    kind="speech_end",
                    end_ms=end_ms,
                    start_ms=self._utterance_start_ms,
                )
            )
            self.in_speech = False
            self._speech_streak = 0
            self._silence_streak = 0
            self._utterance_start_ms = None
            self._pre_roll.clear()
        return events

    def flush(self) -> SegmentEvent | None:
        if not self.in_speech:
            return None
        end_ms = self._frame_index * self.frame_ms
        start_ms = self._utterance_start_ms
        self.in_speech = False
        self._utterance_start_ms = None
        self._speech_streak = 0
        self._silence_streak = 0
        self._pre_roll.clear()
        return SegmentEvent(kind="speech_end", start_ms=start_ms, end_ms=end_ms)

