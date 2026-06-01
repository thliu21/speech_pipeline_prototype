#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import tarfile
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

MODEL_URLS = {
    "english": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-streaming-zipformer-en-2023-06-21.tar.bz2",
    "english-fast": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2",
    "bilingual": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",
}
DEFAULT_MODEL_KEY = "english"
DEFAULT_SENTENCE_MODEL = "pcs_en"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Sherpa-ONNX ASR models and sentence models.")
    parser.add_argument("--kind", default="asr", choices=["asr", "sentence"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--url", default=None, help="Override the model URL. The directory name is inferred from the archive.")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--force", action="store_true", help="Re-download and replace an existing model directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.kind == "sentence":
        download_sentence_model(args)
        return
    download_asr_model(args)


def download_asr_model(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_key = args.model or DEFAULT_MODEL_KEY
    if model_key not in MODEL_URLS and not args.url:
        raise ValueError(f"Unknown ASR model {model_key!r}; choose one of {sorted(MODEL_URLS)}")
    url = args.url or MODEL_URLS[model_key]
    archive_path = out_dir / Path(urlparse(url).path).name
    model_dir = out_dir / archive_path.name.removesuffix(".tar.bz2")

    if model_dir.exists() and not args.force:
        print(f"Model already exists: {model_dir}")
        return

    if model_dir.exists() and args.force:
        shutil.rmtree(model_dir)

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, archive_path)
    print(f"Extracting {archive_path}")
    safe_extract_tar(archive_path, out_dir)
    print(f"Ready: {model_dir}")


def download_sentence_model(args: argparse.Namespace) -> None:
    model_name = args.model or DEFAULT_SENTENCE_MODEL
    cache_dir = Path(args.out_dir) / "hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    print(f"Downloading sentence model {model_name} into HF_HOME={cache_dir}")
    from speech_proto.sentence_assembler import create_sentence_boundary_engine

    create_sentence_boundary_engine("punct-en", model_name).warmup()
    print(f"Ready: {model_name}")


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:bz2") as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(destination)):
                raise RuntimeError(f"Refusing to extract path outside destination: {member.name}")
        archive.extractall(destination)


if __name__ == "__main__":
    main()
