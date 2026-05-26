from __future__ import annotations

import asyncio
import json
import re
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .audio_source import MicrophoneSource
from .audio_utils import SAMPLE_RATE


@dataclass
class RecordingMetadata:
    id: str
    title: str
    path: str
    reference_text: str = ""
    device_id: int | None = None
    started_at: str = ""
    finished_at: str | None = None
    duration_sec: float = 0.0
    size_bytes: int = 0
    status: str = "ready"
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "path": self.path,
            "reference_text": self.reference_text,
            "device_id": self.device_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": self.duration_sec,
            "size_bytes": self.size_bytes,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class _ActiveRecording:
    metadata: RecordingMetadata
    stop_event: asyncio.Event
    task: asyncio.Task
    started_monotonic: float = field(default_factory=time.monotonic)


class RecordingManager:
    def __init__(self, storage_dir: str | Path = "recordings") -> None:
        self.storage_dir = Path(storage_dir)
        self._active: _ActiveRecording | None = None
        self._lock = asyncio.Lock()

    @property
    def is_recording(self) -> bool:
        return self._active is not None

    async def start(self, device_id: int | None, title: str = "", reference_text: str = "") -> RecordingMetadata:
        async with self._lock:
            if self._active is not None:
                raise RecordingRunningError("Recording is already running")
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            recording_id = _recording_id(title)
            wav_path = self.storage_dir / f"{recording_id}.wav"
            metadata = RecordingMetadata(
                id=recording_id,
                title=title.strip() or recording_id,
                path=str(wav_path),
                reference_text=reference_text.strip(),
                device_id=device_id,
                started_at=_now_iso(),
                status="recording",
            )
            stop_event = asyncio.Event()
            task = asyncio.create_task(self._record(metadata, stop_event))
            self._active = _ActiveRecording(metadata=metadata, stop_event=stop_event, task=task)
            self._write_metadata(metadata)
            return metadata

    async def stop(self) -> RecordingMetadata:
        async with self._lock:
            active = self._active
            if active is None:
                raise RecordingRunningError("Recording is not running")
            active.stop_event.set()
            task = active.task
        await asyncio.wait_for(task, timeout=5)
        async with self._lock:
            metadata = active.metadata
            self._active = None
            return metadata

    def status(self) -> dict[str, Any]:
        if self._active is None:
            return {"recording": False, "active": None}
        metadata = self._active.metadata.as_dict()
        metadata["duration_sec"] = round(time.monotonic() - self._active.started_monotonic, 3)
        return {"recording": True, "active": metadata}

    def list(self) -> list[dict[str, Any]]:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        recordings = [metadata.as_dict() for metadata in self._read_all_metadata()]
        return sorted(recordings, key=lambda item: str(item.get("started_at", "")), reverse=True)

    def update(self, recording_id: str, title: str | None = None, reference_text: str | None = None) -> RecordingMetadata:
        metadata = self._load_metadata(recording_id)
        if title is not None:
            metadata.title = title.strip() or metadata.title
        if reference_text is not None:
            metadata.reference_text = reference_text.strip()
        self._write_metadata(metadata)
        return metadata

    def delete(self, recording_id: str) -> None:
        if self._active is not None and self._active.metadata.id == recording_id:
            raise RecordingRunningError("Stop the active recording before deleting it")
        metadata = self._load_metadata(recording_id)
        wav_path = Path(metadata.path)
        meta_path = self._metadata_path(recording_id)
        if wav_path.exists():
            wav_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    def audio_path(self, recording_id: str) -> Path:
        metadata = self._load_metadata(recording_id)
        path = Path(metadata.path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    async def _record(self, metadata: RecordingMetadata, stop_event: asyncio.Event) -> None:
        source = MicrophoneSource(metadata.device_id)
        frames_written = 0
        wav_path = Path(metadata.path)
        try:
            async with _WavWriter(wav_path) as writer:
                async for frame in source.frames():
                    if stop_event.is_set():
                        break
                    await writer.write(frame)
                    frames_written += 1
            metadata.status = "ready"
        except Exception as exc:
            metadata.status = "error"
            metadata.error = str(exc)
            raise
        finally:
            metadata.finished_at = _now_iso()
            metadata.duration_sec = round(frames_written * 0.01, 3)
            metadata.size_bytes = wav_path.stat().st_size if wav_path.exists() else 0
            self._write_metadata(metadata)

    def _read_all_metadata(self) -> list[RecordingMetadata]:
        result = []
        for path in self.storage_dir.glob("*.json"):
            try:
                result.append(self._metadata_from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        return result

    def _load_metadata(self, recording_id: str) -> RecordingMetadata:
        if not _valid_recording_id(recording_id):
            raise FileNotFoundError(recording_id)
        path = self._metadata_path(recording_id)
        if not path.is_file():
            raise FileNotFoundError(recording_id)
        return self._metadata_from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _write_metadata(self, metadata: RecordingMetadata) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_path(metadata.id).write_text(
            json.dumps(metadata.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _metadata_path(self, recording_id: str) -> Path:
        return self.storage_dir / f"{recording_id}.json"

    @staticmethod
    def _metadata_from_dict(data: dict[str, Any]) -> RecordingMetadata:
        return RecordingMetadata(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            path=str(data.get("path", "")),
            reference_text=str(data.get("reference_text", "")),
            device_id=data.get("device_id"),
            started_at=str(data.get("started_at", "")),
            finished_at=data.get("finished_at"),
            duration_sec=float(data.get("duration_sec", 0.0) or 0.0),
            size_bytes=int(data.get("size_bytes", 0) or 0),
            status=str(data.get("status", "ready")),
            error=data.get("error"),
        )


class _WavWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    async def __aenter__(self) -> "_WavWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = wave.open(str(self.path), "wb")
        self._handle.setnchannels(1)
        self._handle.setsampwidth(2)
        self._handle.setframerate(SAMPLE_RATE)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            await asyncio.to_thread(self._handle.close)
            self._handle = None

    async def write(self, frame: bytes) -> None:
        if self._handle is None:
            raise RuntimeError("WAV writer is not open")
        await asyncio.to_thread(self._handle.writeframes, frame)


class RecordingRunningError(RuntimeError):
    pass


def _recording_id(title: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", title.strip()).strip("-").lower()
    return f"{stamp}-{slug}" if slug else stamp


def _valid_recording_id(recording_id: str) -> bool:
    return bool(re.fullmatch(r"[a-zA-Z0-9\u4e00-\u9fff_.-]+", recording_id))


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
