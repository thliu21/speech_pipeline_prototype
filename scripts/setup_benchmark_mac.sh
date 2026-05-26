#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

UV_BIN="${UV_BIN:-/Users/arthurliu/.local/bin/uv}"
BENCH_VENV="${BENCH_VENV:-.venv-bench}"

"$UV_BIN" venv --python 3.12 "$BENCH_VENV"
"$UV_BIN" pip install --python "$BENCH_VENV/bin/python" -e ".[dev]"

cat <<EOF
Benchmark environment ready:

  source $BENCH_VENV/bin/activate
  python -m speech_proto.benchmark_suite --help

EOF
