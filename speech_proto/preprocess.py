from __future__ import annotations

from dataclasses import dataclass

from .audio_utils import BYTES_PER_FRAME, rms_level


@dataclass(frozen=True)
class PreprocessResult:
    audio: bytes
    is_speech: bool
    audio_level: float
    backend: str


class PassthroughPreprocessor:
    def __init__(self, vad_threshold: float = 0.012) -> None:
        self.vad_threshold = vad_threshold
        self.backend = "off"

    def process_10ms(self, frame: bytes) -> PreprocessResult:
        _validate_frame(frame)
        level = rms_level(frame)
        return PreprocessResult(
            audio=frame,
            is_speech=level >= self.vad_threshold,
            audio_level=level,
            backend=self.backend,
        )


class WebRtcPreprocessor:
    def __init__(self, auto_gain_dbfs: int = 3, noise_suppression_level: int = 2) -> None:
        try:
            from webrtc_noise_gain import AudioProcessor
        except ImportError as exc:
            raise RuntimeError(
                "webrtc-noise-gain is required for WebRTC preprocessing; "
                "install dependencies or run with --denoise off"
            ) from exc
        self._processor = AudioProcessor(auto_gain_dbfs, noise_suppression_level)
        self.backend = "webrtc"

    def process_10ms(self, frame: bytes) -> PreprocessResult:
        _validate_frame(frame)
        result = self._processor.Process10ms(frame)
        audio = bytes(result.audio)
        return PreprocessResult(
            audio=audio,
            is_speech=bool(result.is_speech),
            audio_level=rms_level(audio),
            backend=self.backend,
        )


def create_preprocessor(mode: str, vad_threshold: float = 0.012) -> PassthroughPreprocessor | WebRtcPreprocessor:
    normalized = mode.lower().strip()
    if normalized in {"off", "none", "passthrough", "mock"}:
        return PassthroughPreprocessor(vad_threshold=vad_threshold)
    if normalized in {"webrtc", "webrtc-noise-gain", "light"}:
        return WebRtcPreprocessor()
    raise ValueError(f"Unsupported denoise mode: {mode}")


def _validate_frame(frame: bytes) -> None:
    if len(frame) != BYTES_PER_FRAME:
        raise ValueError(f"Expected {BYTES_PER_FRAME} bytes for a 10ms 16kHz frame, got {len(frame)}")
