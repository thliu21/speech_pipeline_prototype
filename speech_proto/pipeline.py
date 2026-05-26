from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .asr import DEFAULT_MODEL_NAME, BaseAsrEngine, Transcript, create_asr_engine
from .audio_source import (
    MicrophoneSource,
    WavSource,
    default_input_device_id,
    list_input_devices,
    validate_input_device,
)
from .audio_utils import FRAME_MS, rms_level
from .benchmark import JsonlRecorder, RunClock, score_transcript
from .events import EventHub
from .metrics import LatencyTracker
from .preprocess import PassthroughPreprocessor, create_preprocessor
from .segmenter import SegmentEvent, VadSegmenter
from .transcript_assembler import TranscriptAssembler


@dataclass(frozen=True)
class PipelineConfig:
    source: str = "mic"
    device_id: int | None = None
    wav_path: str | None = None
    wav_realtime: bool = True
    denoise: str = "off"
    asr_mode: str = "sherpa"
    model_dir: str = str(Path("models") / DEFAULT_MODEL_NAME)
    provider: str = "cpu"
    num_threads: int = 1
    decoding_method: str = "greedy_search"
    max_active_paths: int = 4
    use_int8: bool = True
    decode_every_ms: int = 100
    context_padding_ms: int = 800
    vad_threshold: float = 0.012
    speech_start_frames: int = 3
    silence_end_ms: int = 1400
    pre_roll_ms: int = 800
    soft_end_ms: int = 700
    jsonl_log: str | None = None
    reference_text: str | None = None


@dataclass(frozen=True)
class PipelineSummary:
    source: str
    input_device: str
    audio_duration_sec: float
    wall_time_sec: float
    wall_rtf: float | None
    transcript: str
    metrics: dict[str, dict[str, float | int | str]]
    score: dict[str, float | int | None]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineStatus:
    running: bool = False
    model_loaded: bool = False
    device: str = "cpu"
    current_source: str | None = None
    current_input_device: str | None = None
    selected_device_id: int | None = None
    last_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipelineRunner:
    def __init__(self, config: PipelineConfig, hub: EventHub, stop_event: asyncio.Event | None = None) -> None:
        self.config = config
        self.hub = hub
        self.stop_event = stop_event or asyncio.Event()
        self.latencies = LatencyTracker()
        self.segmenter = VadSegmenter(
            speech_start_frames=config.speech_start_frames,
            silence_end_ms=config.silence_end_ms,
            pre_roll_ms=config.pre_roll_ms,
            soft_end_ms=config.soft_end_ms,
        )
        self.transcripts = TranscriptAssembler()
        self.total_frames = 0
        self.input_name = ""

    async def run(self) -> PipelineSummary:
        clock = RunClock()
        with JsonlRecorder(self.config.jsonl_log) as recorder:
            try:
                preprocessor = self._create_preprocessor()
                asr = create_asr_engine(
                    self.config.asr_mode,
                    model_dir=self.config.model_dir,
                    provider=self.config.provider,
                    num_threads=self.config.num_threads,
                    decoding_method=self.config.decoding_method,
                    max_active_paths=self.config.max_active_paths,
                    use_int8=self.config.use_int8,
                    decode_every_ms=self.config.decode_every_ms,
                    context_padding_ms=self.config.context_padding_ms,
                )
                source = self._create_source()
                await self.hub.publish(
                    "pipeline_state",
                    {
                        "vad": "silence",
                        "queue_depth": 0,
                        "audio_level": 0.0,
                        "input_device": self.input_name,
                        "preprocessor": getattr(preprocessor, "backend", self.config.denoise),
                        "asr": asr.name,
                    },
                )
                await self._run_loop(source, preprocessor, asr, recorder)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self.hub.publish("error", {"message": str(exc), "recoverable": False})
                raise
            finally:
                flush_event = self.segmenter.flush()
                if flush_event is not None:
                    await self._publish_segment_event(flush_event, recorder)

            summary = self._summary(clock)
            recorder.write({"type": "summary", "payload": summary.as_dict()})
            await self.hub.publish("summary", summary.as_dict())
            return summary

    def _create_preprocessor(self):
        try:
            return create_preprocessor(self.config.denoise, vad_threshold=self.config.vad_threshold)
        except RuntimeError as exc:
            if self.config.asr_mode.lower() == "mock":
                return PassthroughPreprocessor()
            raise exc

    def _create_source(self):
        if self.config.source == "mic":
            source = MicrophoneSource(self.config.device_id)
            self.input_name = source.name
            return source
        if self.config.source == "wav":
            if not self.config.wav_path:
                raise ValueError("wav_path is required for WAV source")
            source = WavSource(self.config.wav_path, realtime=self.config.wav_realtime)
            self.input_name = source.name
            return source
        raise ValueError(f"Unsupported source: {self.config.source}")

    async def _run_loop(self, source, preprocessor, asr: BaseAsrEngine, recorder: JsonlRecorder) -> None:
        last_metrics_emit_ms = -1
        async_iterator = source.frames().__aiter__()
        while not self.stop_event.is_set():
            try:
                with self.latencies.time_stage("capture"):
                    frame = await async_iterator.__anext__()
            except StopAsyncIteration:
                break

            self.total_frames += 1
            frame_end_ms = self.total_frames * FRAME_MS

            with self.latencies.time_stage("preprocess"):
                preprocessed = preprocessor.process_10ms(frame)

            with self.latencies.time_stage("vad"):
                segment_events = self.segmenter.process(preprocessed.audio, preprocessed.is_speech)

            with self.latencies.time_stage("segment"):
                await self._handle_segment_events(segment_events, asr, frame_end_ms, recorder)

            if frame_end_ms - last_metrics_emit_ms >= 250:
                last_metrics_emit_ms = frame_end_ms
                await self._publish_metrics(
                    vad_state="speech" if self.segmenter.in_speech else "silence",
                    audio_level=preprocessed.audio_level,
                    recorder=recorder,
                )

        flush_event = self.segmenter.flush()
        if flush_event is not None:
            final = asr.finalize(flush_event.start_ms, flush_event.end_ms or self.total_frames * FRAME_MS)
            if final:
                await self._publish_transcript(final, recorder)

    async def _handle_segment_events(
        self,
        events: list[SegmentEvent],
        asr: BaseAsrEngine,
        frame_end_ms: int,
        recorder: JsonlRecorder,
    ) -> None:
        for event in events:
            await self._publish_segment_event(event, recorder)
            if event.kind in {"speech_start", "speech_frame"}:
                start_ms = event.start_ms or max(0, frame_end_ms - FRAME_MS)
                for frame in event.frames:
                    with self.latencies.time_stage("asr"):
                        transcript = asr.accept_frame(frame, frame_end_ms, start_ms)
                    if transcript:
                        await self._publish_transcript(transcript, recorder)
            elif event.kind == "speech_end":
                with self.latencies.time_stage("asr"):
                    transcript = asr.finalize(event.start_ms, event.end_ms or frame_end_ms)
                if transcript:
                    await self._publish_transcript(transcript, recorder)

    async def _publish_metrics(self, vad_state: str, audio_level: float, recorder: JsonlRecorder) -> None:
        for snapshot in self.latencies.all_snapshots():
            payload = snapshot.as_event_payload()
            recorder.write({"type": "metrics", "payload": payload})
            await self.hub.publish("metrics", payload)
        state = {
            "vad": vad_state,
            "queue_depth": 0,
            "audio_level": round(audio_level, 4),
            "input_device": self.input_name,
        }
        recorder.write({"type": "pipeline_state", "payload": state})
        await self.hub.publish("pipeline_state", state)

    async def _publish_segment_event(self, event: SegmentEvent, recorder: JsonlRecorder) -> None:
        payload = {
            "kind": event.kind,
            "start_ms": event.start_ms,
            "end_ms": event.end_ms,
            "frame_count": len(event.frames),
        }
        recorder.write({"type": "segment", "payload": payload})
        await self.hub.publish("segment", payload)

    async def _publish_transcript(self, transcript: Transcript, recorder: JsonlRecorder) -> None:
        if transcript.type == "final":
            recorder.write({"type": "raw_transcript", "payload": transcript.as_payload()})
        update = self.transcripts.process(transcript)
        if update is None:
            return
        payload = update.as_payload()
        recorder.write({"type": "transcript", "payload": payload})
        await self.hub.publish("transcript", payload)

    def _summary(self, clock: RunClock) -> PipelineSummary:
        audio_duration_sec = self.total_frames * FRAME_MS / 1000.0
        wall_time_sec = clock.elapsed_sec()
        transcript = self.transcripts.transcript
        score = score_transcript(self.config.reference_text, transcript).as_dict()
        return PipelineSummary(
            source=self.config.source,
            input_device=self.input_name,
            audio_duration_sec=round(audio_duration_sec, 3),
            wall_time_sec=round(wall_time_sec, 3),
            wall_rtf=round(wall_time_sec / audio_duration_sec, 3) if audio_duration_sec else None,
            transcript=transcript,
            metrics=self.latencies.as_dict(),
            score=score,
        )


class PipelineController:
    def __init__(
        self,
        default_asr: str = "sherpa",
        default_denoise: str = "off",
        default_model_dir: str = str(Path("models") / DEFAULT_MODEL_NAME),
        provider: str = "cpu",
        num_threads: int = 1,
        decoding_method: str = "greedy_search",
        max_active_paths: int = 4,
    ) -> None:
        self.hub = EventHub()
        self.status = PipelineStatus(device=provider)
        self.status.selected_device_id = default_input_device_id()
        self.default_asr = default_asr
        self.default_denoise = default_denoise
        self.default_model_dir = default_model_dir
        self.provider = provider
        self.num_threads = num_threads
        self.decoding_method = decoding_method
        self.max_active_paths = max_active_paths
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        self._lock = asyncio.Lock()

    async def select_device(self, device_id: int) -> AudioDevicePayload:
        async with self._lock:
            if self.status.running:
                raise PipelineRunningError("Stop the pipeline before switching microphones")
            device = validate_input_device(device_id)
            self.status.selected_device_id = device.id
            return device.as_dict()

    async def start_mic(self, device_id: int | None = None) -> None:
        selected = device_id if device_id is not None else self.status.selected_device_id
        if selected is not None:
            validate_input_device(selected)
        config = PipelineConfig(
            source="mic",
            device_id=selected,
            denoise=self.default_denoise,
            asr_mode=self.default_asr,
            model_dir=self.default_model_dir,
            provider=self.provider,
            num_threads=self.num_threads,
            decoding_method=self.decoding_method,
            max_active_paths=self.max_active_paths,
        )
        await self._start(config)

    async def run_wav(self, path: str, realtime: bool = True, reference_text: str | None = None) -> None:
        config = PipelineConfig(
            source="wav",
            wav_path=path,
            wav_realtime=realtime,
            denoise=self.default_denoise,
            asr_mode=self.default_asr,
            model_dir=self.default_model_dir,
            provider=self.provider,
            num_threads=self.num_threads,
            decoding_method=self.decoding_method,
            max_active_paths=self.max_active_paths,
            reference_text=reference_text,
        )
        await self._start(config)

    async def stop(self) -> None:
        async with self._lock:
            if self._stop_event is not None:
                self._stop_event.set()
            task = self._task
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=3)
            except TimeoutError:
                task.cancel()
        async with self._lock:
            self.status.running = False
            self.status.current_source = None
            self.status.current_input_device = None
            self._task = None
            self._stop_event = None

    async def _start(self, config: PipelineConfig) -> None:
        async with self._lock:
            if self.status.running:
                raise PipelineRunningError("Pipeline is already running")
            self._stop_event = asyncio.Event()
            runner = PipelineRunner(config, self.hub, self._stop_event)
            self._task = asyncio.create_task(self._run_task(runner, config))
            self.status.running = True
            self.status.current_source = config.source
            self.status.current_input_device = str(config.device_id) if config.source == "mic" else config.wav_path
            self.status.last_error = None

    async def _run_task(self, runner: PipelineRunner, config: PipelineConfig) -> None:
        try:
            await runner.run()
            self.status.model_loaded = True
        except Exception as exc:
            self.status.last_error = str(exc)
        finally:
            self.status.running = False
            self.status.current_source = None
            self.status.current_input_device = None


class PipelineRunningError(RuntimeError):
    pass


AudioDevicePayload = dict[str, Any]
