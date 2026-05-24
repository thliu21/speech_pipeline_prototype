# Speech Transcript Pipeline

Offline speech-to-text prototype for Raspberry Pi 5 and macOS. It has two paths:

- Real-time Web UI for microphone capture and pipeline diagnostics.
- Offline WAV replay/benchmark path that reuses the same audio, preprocessing,
  VAD, ASR, and metrics modules.

The intended runtime pipeline is:

```text
selected mic or WAV -> 16 kHz mono 10 ms frames -> WebRTC NS/AGC/VAD
  -> VAD smoothing + 500 ms pre-roll -> Sherpa-ONNX streaming ASR
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
models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
```

The earlier "small 2023-02-16" model name is mainly useful for RKNN/Rockchip
experiments. For the Python CPU prototype, this repo defaults to the official
Sherpa-ONNX bilingual Chinese/English transducer model used in their microphone
example.

## Run The Web UI

For a model-free smoke test:

```bash
python -m speech_proto.app --asr mock --denoise off --host 0.0.0.0 --port 8080
```

For real local ASR:

```bash
python -m speech_proto.app --asr sherpa --denoise webrtc --host 0.0.0.0 --port 8080
```

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
  --reference "打开客厅的灯 and start recording" \
  --asr sherpa \
  --denoise webrtc \
  --jsonl-log logs/noisy_room_001.jsonl
```

The summary includes wall-clock RTF, final transcript, per-stage latency
snapshots, and optional CER/WER.

## Benchmark Guidance

Use public datasets for sanity checks, but decide hardware/model tradeoffs with
your own noisy-room recordings:

- Accuracy: CER for Chinese, WER for English, both for mixed Chinese/English.
- Real time: RTF p50/p95, first partial latency, speech-end-to-final latency.
- VAD: false alarms, missed speech, start clipping, utterance fragmentation.
- Device: CPU, RAM, temperature, throttling, long-run stability.
- Denoising: compare `raw -> ASR` against `denoised -> ASR`; better sounding
  audio is not always better for ASR.

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

