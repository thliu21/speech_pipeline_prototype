#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download Sherpa-ONNX streaming ASR models.")
    parser.add_argument("--model", default=DEFAULT_MODEL_KEY, choices=sorted(MODEL_URLS))
    parser.add_argument("--url", default=None, help="Override the model URL. The directory name is inferred from the archive.")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--force", action="store_true", help="Re-download and replace an existing model directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    url = args.url or MODEL_URLS[args.model]
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
