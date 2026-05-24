from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audio_utils import SAMPLE_RATE, int16_bytes_to_float32

DEFAULT_MODEL_NAME = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"


@dataclass(frozen=True)
class Transcript:
    type: str
    text: str
    start_ms: int
    end_ms: int

    def as_payload(self) -> dict[str, int | str]:
        return {
            "type": self.type,
            "text": self.text,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
        }


class BaseAsrEngine:
    name = "base"

    def accept_frame(self, frame: bytes, timestamp_ms: int, start_ms: int) -> Transcript | None:
        raise NotImplementedError

    def finalize(self, start_ms: int | None, end_ms: int) -> Transcript | None:
        raise NotImplementedError

    @property
    def model_loaded(self) -> bool:
        return True


class MockAsrEngine(BaseAsrEngine):
    name = "mock"

    def __init__(self) -> None:
        self._accepted_frames = 0
        self._last_partial_bucket = -1

    def accept_frame(self, frame: bytes, timestamp_ms: int, start_ms: int) -> Transcript | None:
        self._accepted_frames += 1
        bucket = self._accepted_frames // 50
        if bucket != self._last_partial_bucket:
            self._last_partial_bucket = bucket
            seconds = self._accepted_frames / 100.0
            return Transcript("partial", f"[mock transcript {seconds:.1f}s]", start_ms, timestamp_ms)
        return None

    def finalize(self, start_ms: int | None, end_ms: int) -> Transcript | None:
        if self._accepted_frames == 0:
            return None
        seconds = self._accepted_frames / 100.0
        text = f"[mock final transcript {seconds:.1f}s]"
        self._accepted_frames = 0
        self._last_partial_bucket = -1
        return Transcript("final", text, start_ms or 0, end_ms)


class SherpaOnnxAsrEngine(BaseAsrEngine):
    name = "sherpa-onnx"

    def __init__(
        self,
        model_dir: str | Path,
        provider: str = "cpu",
        num_threads: int = 1,
        decoding_method: str = "greedy_search",
        use_int8: bool = True,
        decode_every_ms: int = 100,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.provider = provider
        self.num_threads = num_threads
        self.decoding_method = decoding_method
        self.use_int8 = use_int8
        self.decode_every_ms = decode_every_ms
        self._recognizer = self._create_recognizer()
        self._stream = self._recognizer.create_stream()
        self._last_decode_ms = 0
        self._last_text = ""

    @property
    def model_loaded(self) -> bool:
        return True

    def accept_frame(self, frame: bytes, timestamp_ms: int, start_ms: int) -> Transcript | None:
        samples = int16_bytes_to_float32(frame)
        self._stream.accept_waveform(SAMPLE_RATE, samples)
        if timestamp_ms - self._last_decode_ms < self.decode_every_ms:
            return None
        self._last_decode_ms = timestamp_ms
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        text = self._recognizer.get_result(self._stream).strip()
        if text and text != self._last_text:
            self._last_text = text
            return Transcript("partial", text, start_ms, timestamp_ms)
        return None

    def finalize(self, start_ms: int | None, end_ms: int) -> Transcript | None:
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        text = self._recognizer.get_result(self._stream).strip()
        self._recognizer.reset(self._stream)
        self._last_decode_ms = 0
        self._last_text = ""
        if not text:
            return None
        return Transcript("final", text, start_ms or 0, end_ms)

    def _create_recognizer(self):
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError("sherpa-onnx is required for real ASR; run with --asr mock for UI smoke tests") from exc
        paths = resolve_transducer_model_paths(self.model_dir, self.use_int8)
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(paths["tokens"]),
            encoder=str(paths["encoder"]),
            decoder=str(paths["decoder"]),
            joiner=str(paths["joiner"]),
            num_threads=self.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method=self.decoding_method,
            provider=self.provider,
        )


def create_asr_engine(
    mode: str,
    model_dir: str | Path | None = None,
    provider: str = "cpu",
    num_threads: int = 1,
) -> BaseAsrEngine:
    normalized = mode.lower().strip()
    if normalized == "mock":
        return MockAsrEngine()
    if normalized in {"sherpa", "sherpa-onnx", "real"}:
        model_path = Path(model_dir) if model_dir else Path("models") / DEFAULT_MODEL_NAME
        return SherpaOnnxAsrEngine(model_path, provider=provider, num_threads=num_threads)
    raise ValueError(f"Unsupported ASR mode: {mode}")


def resolve_transducer_model_paths(model_dir: Path, use_int8: bool = True) -> dict[str, Path]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    suffix = ".int8.onnx" if use_int8 else ".onnx"
    encoder = _first_existing(
        model_dir,
        [
            f"encoder-epoch-99-avg-1{suffix}",
            "encoder-epoch-99-avg-1.onnx",
            "encoder.onnx",
        ],
    )
    decoder = _first_existing(
        model_dir,
        [
            f"decoder-epoch-99-avg-1{suffix}",
            "decoder-epoch-99-avg-1.onnx",
            "decoder.onnx",
        ],
    )
    joiner = _first_existing(
        model_dir,
        [
            f"joiner-epoch-99-avg-1{suffix}",
            "joiner-epoch-99-avg-1.onnx",
            "joiner.onnx",
        ],
    )
    tokens = model_dir / "tokens.txt"
    missing = [path for path in [encoder, decoder, joiner, tokens] if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing model files: {missing_text}")
    return {"encoder": encoder, "decoder": decoder, "joiner": joiner, "tokens": tokens}


def _first_existing(model_dir: Path, names: list[str]) -> Path:
    for name in names:
        path = model_dir / name
        if path.is_file():
            return path
    return model_dir / names[0]

