# Speech Transcript Pipeline

Offline speech-to-text prototype for Raspberry Pi 5 and macOS. It has two paths:

- Real-time Web UI for microphone capture and pipeline diagnostics.
- Offline WAV replay/benchmark path that reuses the same audio, preprocessing,
  VAD, ASR, and metrics modules.

The intended runtime pipeline is:

```text
selected mic or WAV -> 16 kHz mono 10 ms frames -> light energy VAD
  -> sentence-level VAD endpointing -> Sherpa-ONNX English streaming ASR
  -> WebSocket metrics/transcript events
```

## Setup

On Raspberry Pi OS:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev portaudio19-dev libsndfile1 build-essential
cd /Users/arthurliu/dev/speech_transcript_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

On macOS, install PortAudio first if `sounddevice` cannot find devices:

```bash
brew install portaudio libsndfile
```

## Download The Default ASR Model

```bash
python scripts/download_models.py
```

This downloads:

```text
models/sherpa-onnx-streaming-zipformer-en-2023-06-21
```

The pipeline is currently tuned for English-only recognition. The default model
is the Sherpa-ONNX English streaming Zipformer trained on LibriSpeech and
GigaSpeech. For a smaller English model, use:

```bash
python scripts/download_models.py --model english-fast
```

The previous Chinese/English bilingual model is still available for experiments:

```bash
python scripts/download_models.py --model bilingual
```

## Run The Web UI

For a model-free smoke test:

```bash
python -m speech_proto.app --asr mock --denoise off --host 0.0.0.0 --port 8080
```

For real local ASR:

```bash
python -m speech_proto.app --asr sherpa --denoise off --host 0.0.0.0 --port 8080 --num-threads 2
```

For a noisy room, try `--denoise webrtc`, but clean USB microphones often keep
English suffixes more accurately with denoise disabled.

Open:

```text
http://<pi5-host>:8080
```

The UI supports microphone refresh, microphone selection, start/stop, WAV replay,
per-stage latency cards, VAD state, audio level, partial/final transcript, and
recent events.

Microphone switching is intentionally blocked while the pipeline is running.
Stop first, select another microphone, then start again.

## Run WAV Replay / Benchmark

Mock smoke test:

```bash
python -m speech_proto.cli --wav samples/example.wav --asr mock --denoise off
```

Real ASR benchmark with JSONL event log and reference transcript:

```bash
python -m speech_proto.cli \
  --wav samples/noisy_room_001.wav \
  --reference "I live in Sunnyvale and I am testing the microphone" \
  --asr sherpa \
  --denoise off \
  --jsonl-log logs/noisy_room_001.jsonl
```

The summary includes wall-clock RTF, final transcript, per-stage latency
snapshots, and optional CER/WER.

## Run Streaming Benchmark Suites

The benchmark suite feeds public and local audio through the same streaming
pipeline used by the Web UI, then reports WER, boundary quality, endpoint
latency, partial churn, and first/last word loss:

```bash
bash scripts/setup_benchmark_mac.sh
source .venv-bench/bin/activate

python -m speech_proto.benchmark_suite prepare \
  --include local,librispeech,ljspeech \
  --download \
  --max-per-dataset 20 \
  --synthetic-count 8 \
  --out benchmarks/cache/manifest.jsonl

python -m speech_proto.benchmark_suite run-stream \
  --manifest benchmarks/cache/manifest.jsonl \
  --config benchmarks/configs/current.json \
  --out benchmarks/runs/current

python -m speech_proto.benchmark_suite compare benchmarks/runs/current
```

Common Voice requires accepting Mozilla's dataset terms first; after downloading
it manually, pass `--include commonvoice --commonvoice-dir <path>`.

## Benchmark Guidance

Use public datasets for sanity checks, but decide hardware/model tradeoffs with
your own noisy-room recordings:

- Accuracy: WER for English.
- Real time: RTF p50/p95, first partial latency, speech-end-to-final latency.
- VAD: false alarms, missed speech, start clipping, utterance fragmentation.
- Device: CPU, RAM, temperature, throttling, long-run stability.
- Denoising: compare `raw -> ASR` against `denoised -> ASR`; better sounding
  audio is not always better for ASR.

The current default uses 800 ms ASR context padding, 800 ms VAD pre-roll, and a
1400 ms final endpoint with a 700 ms soft endpoint. In Mac stream benchmarks this
reduced first/last word loss while improving balanced boundary F1 versus the
older 500 ms padding baseline.

Suggested recording matrix:

```text
near/far speaker x quiet/crowd/music/fan/device-speaker x mic position
```

Keep the raw audio, processed audio if you add dumps later, human transcript,
device ID, mic name, and JSONL event log together.

## Tests

```bash
python -m pytest
```

The unit tests do not require real microphones or model files.

## Troubleshooting

- `sounddevice is required`: install project dependencies and PortAudio.
- No microphone devices: check OS permissions and `python -c "import sounddevice as sd; print(sd.query_devices())"`.
- `webrtc-noise-gain` build failure on Pi: ensure `python3-dev` and `build-essential` are installed, or run with `--denoise off`.
- Missing model files: run `python scripts/download_models.py`, or pass `--model-dir`.
- UI loads but real ASR fails: try `--asr mock --denoise off` first to isolate microphone/UI issues from model setup.
