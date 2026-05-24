import argparse
from pathlib import Path
from typing import Any

from .asr import DEFAULT_MODEL_NAME
from .audio_source import list_input_devices
from .pipeline import PipelineController, PipelineRunningError

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(controller: PipelineController):
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

    app = FastAPI(title="Speech Transcript Pipeline")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
                event = await queue.get()
                await websocket.send_json(event)
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
    parser.add_argument("--denoise", default="webrtc", choices=["webrtc", "off"])
    parser.add_argument("--model-dir", default=str(Path("models") / DEFAULT_MODEL_NAME))
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
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
    )
    uvicorn.run(create_app(controller), host=args.host, port=args.port, ws="websockets")


if __name__ == "__main__":
    main()
