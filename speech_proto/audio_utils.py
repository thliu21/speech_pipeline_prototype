from __future__ import annotations

import array
import math
import sys
from collections.abc import Iterable

SAMPLE_RATE = 16_000
FRAME_MS = 10
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_MS // 1000
BYTES_PER_SAMPLE = 2
BYTES_PER_FRAME = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE


def split_frame_bytes(audio_bytes: bytes, frame_size: int = BYTES_PER_FRAME) -> Iterable[bytes]:
    usable = len(audio_bytes) - (len(audio_bytes) % frame_size)
    for offset in range(0, usable, frame_size):
        yield audio_bytes[offset : offset + frame_size]


def rms_level(frame_bytes: bytes) -> float:
    if not frame_bytes:
        return 0.0
    samples = array.array("h")
    samples.frombytes(frame_bytes)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        return 0.0
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    return min(1.0, math.sqrt(mean_square) / 32768.0)


def int16_bytes_to_float32(frame_bytes: bytes):
    np = require_numpy()
    samples = np.frombuffer(frame_bytes, dtype="<i2").astype("float32")
    return samples / 32768.0


def float32_to_int16_bytes(samples) -> bytes:
    np = require_numpy()
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def ensure_mono_float32(samples):
    np = require_numpy()
    arr = np.asarray(samples, dtype="float32")
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    elif arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def resample_float32(samples, input_rate: int, output_rate: int = SAMPLE_RATE):
    np = require_numpy()
    samples = ensure_mono_float32(samples)
    if input_rate == output_rate:
        return samples
    try:
        import soxr
    except ImportError as exc:
        raise RuntimeError("soxr is required when input audio is not already 16 kHz") from exc
    return np.asarray(soxr.resample(samples, input_rate, output_rate), dtype="float32")


def frames_from_float32(samples, input_rate: int, frame_ms: int = FRAME_MS) -> Iterable[bytes]:
    samples_16k = resample_float32(samples, input_rate, SAMPLE_RATE)
    frame_samples = SAMPLE_RATE * frame_ms // 1000
    usable = len(samples_16k) - (len(samples_16k) % frame_samples)
    for offset in range(0, usable, frame_samples):
        yield float32_to_int16_bytes(samples_16k[offset : offset + frame_samples])


def require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for audio sample conversion") from exc
    return np

