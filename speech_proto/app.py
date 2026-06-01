import asyncio
import argparse
from pathlib import Path
from typing import Any

from .asr import DEFAULT_MODEL_NAME
from .audio_source import list_input_devices, validate_input_device
from .pipeline import PipelineController, PipelineRunningError
from .recordings import RecordingManager, RecordingRunningError
from .sentence_assembler import DEFAULT_SENTENCE_MODEL

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(controller: PipelineController, recording_manager: RecordingManager | None = None):
    try:
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel
    except ImportError as exc:
        raise RuntimeError("fastapi and uvicorn are required to run the Web UI") from exc

    class SelectDeviceRequest(BaseModel):
        device_id: int

    class StartRequest(BaseModel):
        source: str = "mic"
        device_id: int | None = None

    class RunWavRequest(BaseModel):
        path: str
        realtime: bool = True
        reference_text: str | None = None

    class StartRecordingRequest(BaseModel):
        device_id: int | None = None
        title: str = ""
        reference_text: str = ""

    class UpdateRecordingRequest(BaseModel):
        title: str | None = None
        reference_text: str | None = None

    app = FastAPI(title="Speech Transcript Pipeline")
    recordings = recording_manager or RecordingManager()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def prevent_ui_cache(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return controller.status.as_dict()

    @app.get("/api/audio/devices")
    async def audio_devices() -> dict[str, Any]:
        try:
            devices = [device.as_dict() for device in list_input_devices()]
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"devices": devices, "selected_device_id": controller.status.selected_device_id}

    @app.post("/api/audio/select_device")
    async def select_device(request: SelectDeviceRequest) -> dict[str, Any]:
        try:
            device = await controller.select_device(request.device_id)
        except PipelineRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"device": device}

    @app.post("/api/start")
    async def start(request: StartRequest) -> dict[str, Any]:
        if request.source != "mic":
            raise HTTPException(status_code=400, detail="Only source='mic' is supported by /api/start")
        try:
            await controller.start_mic(request.device_id)
        except PipelineRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True}

    @app.post("/api/run_wav")
    async def run_wav(request: RunWavRequest) -> dict[str, Any]:
        try:
            await controller.run_wav(request.path, realtime=request.realtime, reference_text=request.reference_text)
        except PipelineRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True}

    @app.get("/api/recordings")
    async def list_recordings() -> dict[str, Any]:
        return {"recordings": recordings.list(), **recordings.status()}

    @app.get("/api/recordings/status")
    async def recording_status() -> dict[str, Any]:
        return recordings.status()

    @app.post("/api/recordings/start")
    async def start_recording(request: StartRecordingRequest) -> dict[str, Any]:
        if controller.status.running:
            raise HTTPException(status_code=409, detail="Stop the pipeline before recording")
        device_id = request.device_id if request.device_id is not None else controller.status.selected_device_id
        try:
            if device_id is not None:
                validate_input_device(device_id)
            metadata = await recordings.start(
                device_id=device_id,
                title=request.title,
                reference_text=request.reference_text,
            )
        except RecordingRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"recording": metadata.as_dict()}

    @app.post("/api/recordings/stop")
    async def stop_recording() -> dict[str, Any]:
        try:
            metadata = await recordings.stop()
        except RecordingRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"recording": metadata.as_dict()}

    @app.patch("/api/recordings/{recording_id}")
    async def update_recording(recording_id: str, request: UpdateRecordingRequest) -> dict[str, Any]:
        try:
            metadata = recordings.update(
                recording_id,
                title=request.title,
                reference_text=request.reference_text,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Recording was not found") from exc
        return {"recording": metadata.as_dict()}

    @app.delete("/api/recordings/{recording_id}")
    async def delete_recording(recording_id: str) -> dict[str, Any]:
        try:
            recordings.delete(recording_id)
        except RecordingRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Recording was not found") from exc
        return {"ok": True}

    @app.get("/api/recordings/{recording_id}/audio")
    async def recording_audio(recording_id: str):
        try:
            path = recordings.audio_path(recording_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Recording audio was not found") from exc
        return FileResponse(path, media_type="audio/wav", filename=path.name)

    @app.post("/api/stop")
    async def stop() -> dict[str, Any]:
        await controller.stop()
        return {"ok": True}

    @app.websocket("/ws/events")
    async def events(websocket: WebSocket):
        await websocket.accept()
        queue = await controller.hub.subscribe()
        await websocket.send_json({"type": "status", "payload": controller.status.as_dict()})
        try:
            while True:
                event_task = asyncio.create_task(queue.get())
                disconnect_task = asyncio.create_task(websocket.receive_text())
                done, pending = await asyncio.wait(
                    {event_task, disconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                if disconnect_task in done:
                    await disconnect_task
                    continue
                await websocket.send_json(event_task.result())
        except WebSocketDisconnect:
            pass
        finally:
            await controller.hub.unsubscribe(queue)

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Pi5 speech pipeline Web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--asr", default="sherpa", choices=["sherpa", "sherpa-onnx", "mock"])
    parser.add_argument("--denoise", default="off", choices=["webrtc", "off"])
    parser.add_argument("--model-dir", default=str(Path("models") / DEFAULT_MODEL_NAME))
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--decoding-method", default="greedy_search", choices=["greedy_search", "modified_beam_search"])
    parser.add_argument("--max-active-paths", type=int, default=4)
    parser.add_argument("--sentence-mode", default="punct-en", choices=["punct-en", "raw"])
    parser.add_argument("--sentence-model", default=DEFAULT_SENTENCE_MODEL)
    return parser


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run the Web UI") from exc
    args = build_parser().parse_args()
    controller = PipelineController(
        default_asr=args.asr,
        default_denoise=args.denoise,
        default_model_dir=args.model_dir,
        provider=args.provider,
        num_threads=args.num_threads,
        decoding_method=args.decoding_method,
        max_active_paths=args.max_active_paths,
        sentence_mode=args.sentence_mode,
        sentence_model=args.sentence_model,
    )
    uvicorn.run(create_app(controller), host=args.host, port=args.port, ws="websockets")


if __name__ == "__main__":
    main()
