from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .audio_utils import require_numpy
from .transcript_assembler import TranscriptSegment, TranscriptUpdate


DEFAULT_SENTENCE_MODEL = "pcs_en"
PUNCTUATOR_MODEL_INFO = {
    "pcs_en": {
        "repo_id": "1-800-BAD-CODE/punct_cap_seg_en",
        "spe_filename": "spe_32k_lc_en.model",
        "model_filename": "punct_cap_seg_en.onnx",
        "config_filename": "config.yaml",
    },
}


class SentenceBoundaryEngine(Protocol):
    name: str

    def warmup(self) -> None:
        ...

    def split(self, text: str) -> list[str]:
        ...


class RawSentenceBoundaryEngine:
    name = "raw"

    def warmup(self) -> None:
        return None

    def split(self, text: str) -> list[str]:
        text = text.strip()
        return [text] if text else []


class PunctuatorSentenceBoundaryEngine:
    def __init__(self, model_name: str = DEFAULT_SENTENCE_MODEL) -> None:
        self.model_name = model_name
        self.name = f"punctuators:{model_name}"
        self._runtime: _PunctuatorOnnxRuntime | None = None

    def warmup(self) -> None:
        self._load_runtime()

    def split(self, text: str) -> list[str]:
        text = _normalize_model_input(text)
        if not text:
            return []
        return self._load_runtime().infer(text)

    def _load_runtime(self) -> "_PunctuatorOnnxRuntime":
        if self._runtime is None:
            self._runtime = _PunctuatorOnnxRuntime.from_pretrained(self.model_name)
        return self._runtime


class _PunctuatorOnnxRuntime:
    def __init__(self, spe_path: Path, onnx_path: Path, config_path: Path, overlap: int = 16) -> None:
        try:
            import onnxruntime as ort
            import yaml
            from sentencepiece import SentencePieceProcessor
        except ImportError as exc:
            raise RuntimeError(
                "sentence_mode='punct-en' requires onnxruntime, sentencepiece, huggingface_hub, and pyyaml"
            ) from exc
        self._tokenizer = SentencePieceProcessor(str(spe_path))
        self._session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self._max_len = int(config["max_length"])
        self._pre_labels = list(config.get("pre_labels", ["<NULL>"]))
        self._post_labels = list(config["post_labels"])
        self._null_token = str(config.get("null_token", "<NULL>"))
        self._overlap = overlap

    @classmethod
    def from_pretrained(cls, model_name: str) -> "_PunctuatorOnnxRuntime":
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("sentence_mode='punct-en' requires huggingface_hub") from exc
        info = _resolve_punctuator_model(model_name)
        spe_path = Path(hf_hub_download(repo_id=info["repo_id"], filename=info["spe_filename"]))
        onnx_path = Path(hf_hub_download(repo_id=info["repo_id"], filename=info["model_filename"]))
        config_path = Path(hf_hub_download(repo_id=info["repo_id"], filename=info["config_filename"]))
        return cls(spe_path, onnx_path, config_path)

    def infer(self, text: str) -> list[str]:
        np = require_numpy()
        segments = self._tokenize(text)
        if not segments:
            return []
        collected: list[_PunctuatorResultSegment] = []
        for segment_ids in segments:
            input_ids = np.array(
                [[self._tokenizer.bos_id(), *segment_ids, self._tokenizer.eos_id()]],
                dtype=np.int64,
            )
            pre_preds, post_preds, cap_preds, seg_preds = self._session.run(None, {"input_ids": input_ids})
            length = input_ids.shape[1]
            ids = input_ids[0, 1 : length - 1].tolist()
            collected.append(
                _PunctuatorResultSegment(
                    ids=ids,
                    pre_preds=[self._pre_label(int(value)) for value in pre_preds[0, 1 : length - 1].tolist()],
                    post_preds=[self._post_label(int(value)) for value in post_preds[0, 1 : length - 1].tolist()],
                    cap_preds=cap_preds[0, 1 : length - 1].tolist(),
                    sbd_preds=seg_preds[0, 1 : length - 1].tolist(),
                )
            )
        return self._produce(collected)

    def _tokenize(self, text: str) -> list[list[int]]:
        ids = self._tokenizer.EncodeAsIds(text)
        max_len = self._max_len - 2
        segments: list[list[int]] = []
        start = 0
        input_idx = 0
        while start < len(ids):
            adjusted_start = start - (0 if input_idx == 0 else self._overlap)
            stop = adjusted_start + max_len
            segments.append(ids[adjusted_start:stop])
            start = stop
            input_idx += 1
        return segments

    def _produce(self, segments: list["_PunctuatorResultSegment"]) -> list[str]:
        ids: list[int] = []
        pre_preds: list[str | None] = []
        post_preds: list[str | None] = []
        cap_preds: list[list[bool]] = []
        sbd_preds: list[bool] = []
        for index, segment in enumerate(segments):
            start = self._overlap // 2 if index > 0 else 0
            stop = len(segment.ids) - (self._overlap // 2 if index < len(segments) - 1 else 0)
            ids.extend(segment.ids[start:stop])
            pre_preds.extend(segment.pre_preds[start:stop])
            post_preds.extend(segment.post_preds[start:stop])
            cap_preds.extend(segment.cap_preds[start:stop])
            sbd_preds.extend(segment.sbd_preds[start:stop])

        output_texts: list[str] = []
        current_chars: list[str] = []
        for token_idx, token in enumerate(self._tokenizer.IdToPiece(item) for item in ids):
            if token.startswith("▁") and current_chars:
                current_chars.append(" ")
            char_start = 1 if token.startswith("▁") else 0
            for token_char_idx, char in enumerate(token[char_start:], start=char_start):
                if token_char_idx == char_start and pre_preds[token_idx] is not None:
                    current_chars.append(str(pre_preds[token_idx]))
                if token_char_idx < len(cap_preds[token_idx]) and cap_preds[token_idx][token_char_idx]:
                    char = char.upper()
                current_chars.append(char)
                label = post_preds[token_idx]
                if label == "<ACRONYM>":
                    current_chars.append(".")
                elif token_char_idx == len(token) - 1 and label is not None:
                    current_chars.append(label)
                if token_char_idx == len(token) - 1 and sbd_preds[token_idx]:
                    output_texts.append("".join(current_chars).strip())
                    current_chars = []
        if current_chars:
            output_texts.append("".join(current_chars).strip())
        return [sentence for sentence in output_texts if sentence]

    def _pre_label(self, label_id: int) -> str | None:
        label = self._pre_labels[label_id]
        return None if label == self._null_token else label

    def _post_label(self, label_id: int) -> str | None:
        label = self._post_labels[label_id]
        return None if label == self._null_token else label


@dataclass(frozen=True)
class _PunctuatorResultSegment:
    ids: list[int]
    pre_preds: list[str | None]
    post_preds: list[str | None]
    cap_preds: list[list[bool]]
    sbd_preds: list[bool]


def _resolve_punctuator_model(model_name: str) -> dict[str, str]:
    if model_name in PUNCTUATOR_MODEL_INFO:
        return PUNCTUATOR_MODEL_INFO[model_name]
    if "/" in model_name:
        return {
            "repo_id": model_name,
            "spe_filename": "sp.model",
            "model_filename": "model.onnx",
            "config_filename": "config.yaml",
        }
    raise ValueError(f"Unknown sentence model {model_name!r}; supported aliases: {sorted(PUNCTUATOR_MODEL_INFO)}")


@dataclass(frozen=True)
class RawTranscriptChunk:
    segment_id: int
    text: str
    start_ms: int
    end_ms: int


class SentenceAssembler:
    def __init__(self, engine: SentenceBoundaryEngine) -> None:
        self.engine = engine
        self._next_segment_id = 1
        self._segments: list[TranscriptSegment] = []
        self._raw_chunks: OrderedDict[int, RawTranscriptChunk] = OrderedDict()

    @property
    def transcript(self) -> str:
        return " ".join(segment.text for segment in self._segments).strip()

    @property
    def raw_transcript(self) -> str:
        return " ".join(chunk.text for chunk in self._raw_chunks.values()).strip()

    def warmup(self) -> None:
        self.engine.warmup()

    def process(self, update: TranscriptUpdate) -> list[TranscriptUpdate]:
        if update.type != "final":
            return [
                TranscriptUpdate(
                    type=update.type,
                    text=update.text,
                    start_ms=update.start_ms,
                    end_ms=update.end_ms,
                    segment_id=self._next_segment_id,
                    revision=0,
                    op="partial",
                )
            ]

        self._upsert_raw(update)
        sentences = self.engine.split(self.raw_transcript)
        if not sentences:
            return []
        return self._apply_sentences(sentences)

    def _upsert_raw(self, update: TranscriptUpdate) -> None:
        self._raw_chunks[update.segment_id] = RawTranscriptChunk(
            segment_id=update.segment_id,
            text=update.text,
            start_ms=update.start_ms,
            end_ms=update.end_ms,
        )

    def _apply_sentences(self, sentences: list[str]) -> list[TranscriptUpdate]:
        times = _estimate_sentence_times(sentences, list(self._raw_chunks.values()))
        locked_count = min(max(0, len(self._segments) - 1), len(sentences))
        candidates = list(zip(sentences[locked_count:], times[locked_count:]))
        if not candidates:
            return []

        updates: list[TranscriptUpdate] = []
        replace_index = locked_count if locked_count < len(self._segments) else None
        if replace_index is not None:
            sentence, (start_ms, end_ms) = candidates.pop(0)
            update = self._replace(replace_index, sentence, start_ms, end_ms)
            if update is not None:
                updates.append(update)

        for sentence, (start_ms, end_ms) in candidates:
            updates.append(self._append(sentence, start_ms, end_ms))
        return updates

    def _append(self, text: str, start_ms: int, end_ms: int) -> TranscriptUpdate:
        segment = TranscriptSegment(
            segment_id=self._next_segment_id,
            revision=1,
            text=text,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        self._next_segment_id += 1
        self._segments.append(segment)
        return self._update(segment, op="append")

    def _replace(self, index: int, text: str, start_ms: int, end_ms: int) -> TranscriptUpdate | None:
        segment = self._segments[index]
        if segment.text == text and segment.start_ms == start_ms and segment.end_ms == end_ms:
            return None
        segment.text = text
        segment.start_ms = start_ms
        segment.end_ms = end_ms
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


def create_sentence_boundary_engine(mode: str, model_name: str = DEFAULT_SENTENCE_MODEL) -> SentenceBoundaryEngine:
    normalized = mode.lower()
    if normalized == "raw":
        return RawSentenceBoundaryEngine()
    if normalized in {"punct-en", "punctuator", "punctuators"}:
        return PunctuatorSentenceBoundaryEngine(model_name)
    raise ValueError(f"Unsupported sentence_mode: {mode}")


def _normalize_model_input(text: str) -> str:
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    return " ".join(words)


def _estimate_sentence_times(sentences: list[str], chunks: list[RawTranscriptChunk]) -> list[tuple[int, int]]:
    if not chunks:
        return [(0, 0) for _ in sentences]
    start_ms = min(chunk.start_ms for chunk in chunks)
    end_ms = max(chunk.end_ms for chunk in chunks)
    duration = max(0, end_ms - start_ms)
    weights = [max(1, len(_word_tokens(sentence))) for sentence in sentences]
    total = max(1, sum(weights))
    times: list[tuple[int, int]] = []
    cursor = start_ms
    for index, weight in enumerate(weights):
        if index == len(weights) - 1:
            sentence_end = end_ms
        else:
            sentence_end = start_ms + round(duration * sum(weights[: index + 1]) / total)
        times.append((cursor, sentence_end))
        cursor = sentence_end
    return times


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)
