from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .asr import DEFAULT_MODEL_NAME
from .events import EventHub
from .pipeline import PipelineConfig, PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline WAV replay/benchmark through the speech pipeline.")
    parser.add_argument("--wav", required=True, help="Path to a WAV file.")
    parser.add_argument("--reference", default=None, help="Optional reference transcript for CER/WER scoring.")
    parser.add_argument("--asr", default="sherpa", choices=["sherpa", "sherpa-onnx", "mock"])
    parser.add_argument("--denoise", default="webrtc", choices=["webrtc", "off"])
    parser.add_argument("--model-dir", default=str(Path("models") / DEFAULT_MODEL_NAME))
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--jsonl-log", default=None, help="Optional JSONL event log path.")
    parser.add_argument("--realtime", action="store_true", help="Replay WAV at real-time speed.")
    return parser


async def run_async(args: argparse.Namespace) -> dict:
    hub = EventHub()
    config = PipelineConfig(
        source="wav",
        wav_path=args.wav,
        wav_realtime=args.realtime,
        denoise=args.denoise,
        asr_mode=args.asr,
        model_dir=args.model_dir,
        provider=args.provider,
        num_threads=args.num_threads,
        jsonl_log=args.jsonl_log,
        reference_text=args.reference,
    )
    runner = PipelineRunner(config, hub)
    summary = await runner.run()
    return summary.as_dict()


def main() -> None:
    args = build_parser().parse_args()
    summary = asyncio.run(run_async(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

