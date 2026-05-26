from __future__ import annotations

import re
from dataclasses import dataclass

from .asr import Transcript


@dataclass
class TranscriptSegment:
    segment_id: int
    revision: int
    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class TranscriptUpdate:
    type: str
    text: str
    start_ms: int
    end_ms: int
    segment_id: int
    revision: int
    op: str

    def as_payload(self) -> dict[str, int | str]:
        return {
            "type": self.type,
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "segment_id": self.segment_id,
            "revision": self.revision,
            "op": self.op,
        }


class TranscriptAssembler:
    def __init__(self, repair_window_ms: int = 1800) -> None:
        self.repair_window_ms = repair_window_ms
        self._next_segment_id = 1
        self._segments: list[TranscriptSegment] = []

    @property
    def transcript(self) -> str:
        return " ".join(segment.text for segment in self._segments).strip()

    def process(self, transcript: Transcript) -> TranscriptUpdate | None:
        if transcript.type != "final":
            return TranscriptUpdate(
                type=transcript.type,
                text=transcript.text,
                start_ms=transcript.start_ms,
                end_ms=transcript.end_ms,
                segment_id=self._next_segment_id,
                revision=0,
                op="partial",
            )

        text = transcript.text.strip()
        if not text:
            return None

        if self._segments and self._within_repair_window(transcript):
            handled, update = self._repair_last(text, transcript)
            if handled:
                return update

        segment = TranscriptSegment(
            segment_id=self._next_segment_id,
            revision=1,
            text=text,
            start_ms=transcript.start_ms,
            end_ms=transcript.end_ms,
        )
        self._next_segment_id += 1
        self._segments.append(segment)
        return self._update(segment, op="append")

    def _within_repair_window(self, transcript: Transcript) -> bool:
        previous = self._segments[-1]
        gap_ms = transcript.start_ms - previous.end_ms
        return gap_ms <= self.repair_window_ms

    def _repair_last(self, text: str, transcript: Transcript) -> tuple[bool, TranscriptUpdate | None]:
        previous = self._segments[-1]
        previous_norm = _normalize_for_overlap(previous.text)
        text_norm = _normalize_for_overlap(text)
        if not previous_norm or not text_norm:
            return False, None

        if previous_norm == text_norm:
            previous.end_ms = max(previous.end_ms, transcript.end_ms)
            return True, None

        if previous_norm in text_norm:
            return True, self._replace_last(text, transcript.start_ms, transcript.end_ms)

        if text_norm in previous_norm:
            previous.end_ms = max(previous.end_ms, transcript.end_ms)
            return True, None

        overlap = _overlap_prefix_length(previous.text, text)
        if overlap <= 0:
            return False, None

        merged = previous.text + text[_original_index_after_normalized_prefix(text, overlap) :]
        return True, self._replace_last(merged.strip(), previous.start_ms, transcript.end_ms)

    def _replace_last(self, text: str, start_ms: int, end_ms: int) -> TranscriptUpdate:
        segment = self._segments[-1]
        segment.text = text
        segment.start_ms = min(segment.start_ms, start_ms)
        segment.end_ms = max(segment.end_ms, end_ms)
        segment.revision += 1
        return self._update(segment, op="replace")

    @staticmethod
    def _update(segment: TranscriptSegment, op: str) -> TranscriptUpdate:
        return TranscriptUpdate(
            type="final",
            text=segment.text,
            start_ms=segment.start_ms,
            end_ms=segment.end_ms,
            segment_id=segment.segment_id,
            revision=segment.revision,
            op=op,
        )


def _normalize_for_overlap(text: str) -> str:
    return "".join(char.lower() for char in text if _is_match_char(char))


def _overlap_prefix_length(previous: str, current: str) -> int:
    previous_norm = _normalize_for_overlap(previous)
    current_norm = _normalize_for_overlap(current)
    max_length = min(len(previous_norm), len(current_norm))
    for length in range(max_length, 0, -1):
        if not _overlap_long_enough(previous_norm[-length:], current_norm[:length]):
            continue
        if previous_norm[-length:] == current_norm[:length]:
            return length
    return 0


def _overlap_long_enough(left: str, right: str) -> bool:
    text = left or right
    if re.search(r"[a-z0-9]", text):
        return len(text) >= 4
    return len(text) >= 2


def _original_index_after_normalized_prefix(text: str, normalized_length: int) -> int:
    seen = 0
    for index, char in enumerate(text):
        if not _is_match_char(char):
            continue
        seen += 1
        if seen == normalized_length:
            return index + 1
    return len(text)


def _is_match_char(char: str) -> bool:
    return char.isalnum() or "\u4e00" <= char <= "\u9fff"
