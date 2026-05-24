from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScoreResult:
    cer: float | None
    wer: float | None
    reference_chars: int
    reference_words: int

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "cer": self.cer,
            "wer": self.wer,
            "reference_chars": self.reference_chars,
            "reference_words": self.reference_words,
        }


class JsonlRecorder:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self._handle = None

    def __enter__(self) -> "JsonlRecorder":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, item: dict[str, Any]) -> None:
        if self._handle is None:
            return
        self._handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._handle.flush()


class RunClock:
    def __init__(self) -> None:
        self.started_at = time.perf_counter()

    def elapsed_sec(self) -> float:
        return time.perf_counter() - self.started_at


def score_transcript(reference: str | None, hypothesis: str) -> ScoreResult:
    if not reference:
        return ScoreResult(cer=None, wer=None, reference_chars=0, reference_words=0)
    ref_chars = normalize_chars(reference)
    hyp_chars = normalize_chars(hypothesis)
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)
    cer = edit_distance(ref_chars, hyp_chars) / len(ref_chars) if ref_chars else None
    wer = edit_distance(ref_words, hyp_words) / len(ref_words) if ref_words else None
    return ScoreResult(
        cer=cer,
        wer=wer,
        reference_chars=len(ref_chars),
        reference_words=len(ref_words),
    )


def normalize_chars(text: str) -> list[str]:
    normalized = re.sub(r"\s+", "", text.lower())
    normalized = re.sub(r"[^\w\u4e00-\u9fff]", "", normalized)
    return list(normalized)


def normalize_words(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\u4e00-\u9fff\s]", " ", text)
    # Keep CJK characters as standalone tokens while preserving English words.
    tokens: list[str] = []
    for chunk in re.findall(r"[\u4e00-\u9fff]|[a-z0-9_]+", text):
        tokens.append(chunk)
    return tokens


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (0 if ref_token == hyp_token else 1)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]

