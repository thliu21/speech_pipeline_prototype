#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

DEFAULT_MODEL_NAME = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
DEFAULT_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{DEFAULT_MODEL_NAME}.tar.bz2"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the default Sherpa-ONNX bilingual streaming model.")
    parser.add_argument("--url", default=DEFAULT_MODEL_URL)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--force", action="store_true", help="Re-download and replace an existing model directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / Path(args.url).name
    model_dir = out_dir / DEFAULT_MODEL_NAME

    if model_dir.exists() and not args.force:
        print(f"Model already exists: {model_dir}")
        return

    if model_dir.exists() and args.force:
        shutil.rmtree(model_dir)

    print(f"Downloading {args.url}")
    urllib.request.urlretrieve(args.url, archive_path)
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

