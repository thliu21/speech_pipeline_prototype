from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .audio_utils import FRAME_MS, SAMPLE_RATE, frames_from_float32


@dataclass(frozen=True)
class AudioDevice:
    id: int
    name: str
    host_api: str
    max_input_channels: int
    default_samplerate: float
    is_default: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_input_devices() -> list[AudioDevice]:
    sd = require_sounddevice()
    devices = sd.query_devices()
    host_apis = sd.query_hostapis()
    default_input = _default_input_device_id(sd)
    result: list[AudioDevice] = []
    for index, device in enumerate(devices):
        max_input_channels = int(device.get("max_input_channels", 0))
        if max_input_channels <= 0:
            continue
        host_api_index = int(device.get("hostapi", -1))
        host_api = (
            host_apis[host_api_index].get("name", "unknown")
            if 0 <= host_api_index < len(host_apis)
            else "unknown"
        )
        result.append(
            AudioDevice(
                id=index,
                name=str(device.get("name", f"input-{index}")),
                host_api=str(host_api),
                max_input_channels=max_input_channels,
                default_samplerate=float(device.get("default_samplerate", SAMPLE_RATE)),
                is_default=index == default_input,
            )
        )
    return result


def validate_input_device(device_id: int) -> AudioDevice:
    for device in list_input_devices():
        if device.id == device_id:
            return device
    raise ValueError(f"Input device {device_id} was not found")


def default_input_device_id() -> int | None:
    try:
        devices = list_input_devices()
    except RuntimeError:
        return None
    for device in devices:
        if device.is_default:
            return device.id
    return devices[0].id if devices else None


class MicrophoneSource:
    def __init__(self, device_id: int | None = None, frame_ms: int = FRAME_MS) -> None:
        self.device_id = device_id
        self.frame_ms = frame_ms
        self._stream = None
        self._capture_rate = SAMPLE_RATE
        self._samples_per_read = SAMPLE_RATE * frame_ms // 1000

    @property
    def name(self) -> str:
        if self.device_id is None:
            return "System default microphone"
        try:
            return validate_input_device(self.device_id).name
        except Exception:
            return f"Input device {self.device_id}"

    def open(self) -> None:
        sd = require_sounddevice()
        device_id = self.device_id if self.device_id is not None else default_input_device_id()
        self.device_id = device_id
        self._capture_rate = choose_capture_rate(device_id)
        self._samples_per_read = int(self._capture_rate * self.frame_ms / 1000)
        self._stream = sd.InputStream(
            device=device_id,
            channels=1,
            dtype="float32",
            samplerate=self._capture_rate,
            blocksize=self._samples_per_read,
        )
        self._stream.start()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_frame(self) -> bytes:
        if self._stream is None:
            raise RuntimeError("MicrophoneSource is not open")
        samples, overflowed = self._stream.read(self._samples_per_read)
        if overflowed:
            # The frame is still usable; the pipeline will expose backlog through latency.
            pass
        return next(frames_from_float32(samples, self._capture_rate, self.frame_ms))

    async def frames(self):
        self.open()
        try:
            while True:
                yield await asyncio.to_thread(self.read_frame)
        finally:
            self.close()


class WavSource:
    def __init__(self, path: str | Path, realtime: bool = True, frame_ms: int = FRAME_MS) -> None:
        self.path = Path(path)
        self.realtime = realtime
        self.frame_ms = frame_ms

    @property
    def name(self) -> str:
        return str(self.path)

    async def frames(self):
        try:
            import soundfile as sf
        except ImportError as exc:
            raise RuntimeError("soundfile is required for WAV replay") from exc
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        samples, sample_rate = sf.read(self.path, dtype="float32", always_2d=False)
        for frame in frames_from_float32(samples, int(sample_rate), self.frame_ms):
            yield frame
            if self.realtime:
                await asyncio.sleep(self.frame_ms / 1000.0)


def choose_capture_rate(device_id: int | None) -> int:
    sd = require_sounddevice()
    try:
        sd.check_input_settings(device=device_id, channels=1, samplerate=SAMPLE_RATE, dtype="float32")
        return SAMPLE_RATE
    except Exception:
        if device_id is None:
            return SAMPLE_RATE
        device = validate_input_device(device_id)
        return int(device.default_samplerate)


def require_sounddevice():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise RuntimeError("sounddevice is required for microphone capture") from exc
    return sd


def _default_input_device_id(sd_module) -> int | None:
    default = getattr(sd_module, "default", None)
    device = getattr(default, "device", None)
    if isinstance(device, (list, tuple)) and device:
        value = device[0]
        return int(value) if value is not None and int(value) >= 0 else None
    return None
