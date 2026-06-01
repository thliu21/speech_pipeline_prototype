from __future__ import annotations

import argparse
import asyncio
import csv
import json
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .asr import DEFAULT_MODEL_NAME
from .audio_source import WavSource
from .audio_utils import SAMPLE_RATE, require_numpy, resample_float32
from .benchmark import edit_distance, normalize_words, score_transcript
from .events import Event, EventHub
from .pipeline import PipelineConfig, PipelineRunner


LIBRISPEECH_URLS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
}
LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


@dataclass(frozen=True)
class ExpectedSegment:
    text: str
    start_ms: int | None = None
    end_ms: int | None = None

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ExpectedSegment":
        return ExpectedSegment(
            text=str(data.get("text", "")),
            start_ms=_optional_int(data.get("start_ms")),
            end_ms=_optional_int(data.get("end_ms")),
        )


@dataclass(frozen=True)
class ManifestItem:
    id: str
    dataset: str
    audio_path: str
    reference_text: str
    group: str = "default"
    expected_segments: list[ExpectedSegment] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["expected_segments"] = [asdict(segment) for segment in self.expected_segments]
        return data

    @staticmethod
    def from_json(data: dict[str, Any]) -> "ManifestItem":
        return ManifestItem(
            id=str(data["id"]),
            dataset=str(data["dataset"]),
            audio_path=str(data["audio_path"]),
            reference_text=str(data.get("reference_text", "")),
            group=str(data.get("group", "default")),
            expected_segments=[
                ExpectedSegment.from_dict(item) for item in data.get("expected_segments", [])
            ],
        )


@dataclass(frozen=True)
class StreamResult:
    id: str
    dataset: str
    group: str
    audio_duration_sec: float
    wall_time_sec: float
    wall_rtf: float | None
    transcript: str
    cer: float | None
    wer: float | None
    segment_wer: float | None
    segment_count: int
    sentence_count: int
    expected_segment_count: int
    boundary_precision: float | None
    boundary_recall: float | None
    boundary_f1: float | None
    mean_endpoint_latency_ms: float | None
    p95_endpoint_latency_ms: float | None
    partial_count: int
    partial_churn_avg: float
    first_word_missing: bool
    last_word_missing: bool
    tail_s_loss: bool
    metrics: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and run streaming ASR benchmark suites.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build a JSONL manifest from public and local datasets.")
    prepare.add_argument("--out", default="benchmarks/cache/manifest.jsonl")
    prepare.add_argument("--cache-dir", default="benchmarks/cache")
    prepare.add_argument("--include", default="local", help="Comma-separated: local,librispeech,ljspeech,commonvoice")
    prepare.add_argument("--max-per-dataset", type=int, default=20)
    prepare.add_argument("--download", action="store_true", help="Download supported public archives if missing.")
    prepare.add_argument("--local-recordings", default="recordings")
    prepare.add_argument("--librispeech-dir", default=None)
    prepare.add_argument("--ljspeech-dir", default=None)
    prepare.add_argument("--commonvoice-dir", default=None)
    prepare.add_argument("--synthetic-count", type=int, default=0)
    prepare.add_argument("--synthetic-gap-ms", default="300,800,1600,2500")

    run_stream = subparsers.add_parser("run-stream", help="Run a manifest through the real streaming pipeline.")
    run_stream.add_argument("--manifest", default="benchmarks/cache/manifest.jsonl")
    run_stream.add_argument("--config", default=None, help="JSON benchmark config. Missing means current defaults.")
    run_stream.add_argument("--out", default="benchmarks/runs/latest")
    run_stream.add_argument("--limit", type=int, default=None)
    run_stream.add_argument("--realtime", action="store_true", help="Sleep between 10ms frames.")
    run_stream.add_argument("--save-events", action="store_true")

    compare = subparsers.add_parser("compare", help="Compare one or more run summary JSON files or directories.")
    compare.add_argument("runs", nargs="+")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        manifest = prepare_manifest(args)
        write_manifest(manifest, Path(args.out))
        print(json.dumps({"items": len(manifest), "out": args.out}, indent=2))
    elif args.command == "run-stream":
        summary = asyncio.run(run_stream_suite(args))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "compare":
        print(compare_runs(args.runs))


def prepare_manifest(args: argparse.Namespace) -> list[ManifestItem]:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    requested = {item.strip().lower() for item in args.include.split(",") if item.strip()}
    items: list[ManifestItem] = []

    if "local" in requested:
        items.extend(load_local_recordings(Path(args.local_recordings), args.max_per_dataset))
    if "librispeech" in requested:
        root = _prepare_librispeech(cache_dir, args.download, args.librispeech_dir)
        items.extend(load_librispeech(root, args.max_per_dataset))
    if "ljspeech" in requested:
        root = _prepare_ljspeech(cache_dir, args.download, args.ljspeech_dir)
        items.extend(load_ljspeech(root, args.max_per_dataset))
    if "commonvoice" in requested:
        if not args.commonvoice_dir:
            raise ValueError("Common Voice requires --commonvoice-dir after accepting Mozilla's dataset terms.")
        items.extend(load_commonvoice(Path(args.commonvoice_dir), args.max_per_dataset))
    if args.synthetic_count:
        gaps = [int(value) for value in args.synthetic_gap_ms.split(",") if value.strip()]
        items.extend(build_synthetic_streams(items, cache_dir / "synthetic", args.synthetic_count, gaps))
    return items


def load_local_recordings(recordings_dir: Path, limit: int) -> list[ManifestItem]:
    items: list[ManifestItem] = []
    if not recordings_dir.is_dir():
        return items
    for path in sorted(recordings_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        reference = str(data.get("reference_text", "")).strip()
        audio_path = recordings_dir / Path(str(data.get("path", ""))).name
        candidate = Path(str(data.get("path", "")))
        if candidate.is_file():
            audio_path = candidate
        elif (recordings_dir / "imports" / candidate.name).is_file():
            audio_path = recordings_dir / "imports" / candidate.name
        if not reference or not audio_path.is_file():
            continue
        items.append(
            ManifestItem(
                id=str(data.get("id", path.stem)),
                dataset="local",
                audio_path=str(audio_path),
                reference_text=reference,
                group="recordings",
            )
        )
        if len(items) >= limit:
            break
    return items


def load_librispeech(root: Path, limit: int) -> list[ManifestItem]:
    base = root / "LibriSpeech" if (root / "LibriSpeech").is_dir() else root
    items: list[ManifestItem] = []
    group_counts: dict[str, int] = {}
    for transcript_path in sorted(base.glob("*/*/*/*.trans.txt")):
        group = transcript_path.parts[-4]
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            if group_counts.get(group, 0) >= limit:
                break
            utterance_id, reference = line.split(" ", 1)
            audio_path = transcript_path.parent / f"{utterance_id}.flac"
            if audio_path.is_file():
                items.append(
                    ManifestItem(
                        id=utterance_id,
                        dataset="librispeech",
                        audio_path=str(audio_path),
                        reference_text=reference,
                        group=group,
                    )
                )
                group_counts[group] = group_counts.get(group, 0) + 1
    return items


def load_ljspeech(root: Path, limit: int) -> list[ManifestItem]:
    base = root / "LJSpeech-1.1" if (root / "LJSpeech-1.1").is_dir() else root
    metadata = base / "metadata.csv"
    if not metadata.is_file():
        return []
    items: list[ManifestItem] = []
    for line in metadata.read_text(encoding="utf-8").splitlines():
        parts = line.split("|")
        if len(parts) < 2:
            continue
        clip_id = parts[0]
        reference = parts[2] if len(parts) >= 3 and parts[2].strip() else parts[1]
        audio_path = base / "wavs" / f"{clip_id}.wav"
        if audio_path.is_file():
            items.append(
                ManifestItem(
                    id=clip_id,
                    dataset="ljspeech",
                    audio_path=str(audio_path),
                    reference_text=reference,
                    group="ljspeech",
                )
            )
        if len(items) >= limit:
            break
    return items


def load_commonvoice(root: Path, limit: int) -> list[ManifestItem]:
    tsv = _first_existing(root, ["test.tsv", "dev.tsv", "validated.tsv"])
    if tsv is None:
        return []
    items: list[ManifestItem] = []
    with tsv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            clip = str(row.get("path", ""))
            reference = str(row.get("sentence", "")).strip()
            audio_path = root / "clips" / clip
            if reference and audio_path.is_file():
                items.append(
                    ManifestItem(
                        id=Path(clip).stem,
                        dataset="commonvoice",
                        audio_path=str(audio_path),
                        reference_text=reference,
                        group="commonvoice",
                    )
                )
            if len(items) >= limit:
                break
    return items


def build_synthetic_streams(
    source_items: list[ManifestItem],
    out_dir: Path,
    count: int,
    gap_ms_values: list[int],
    segments_per_stream: int = 3,
) -> list[ManifestItem]:
    if len(source_items) < segments_per_stream:
        return []
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to build synthetic stream fixtures") from exc
    np = require_numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    synthetic: list[ManifestItem] = []
    for index in range(count):
        start = (index * segments_per_stream) % (len(source_items) - segments_per_stream + 1)
        parts = source_items[start : start + segments_per_stream]
        gap_ms = gap_ms_values[index % len(gap_ms_values)] if gap_ms_values else 800
        expected: list[ExpectedSegment] = []
        chunks = []
        cursor_ms = 0
        for part_index, part in enumerate(parts):
            samples, sample_rate = sf.read(part.audio_path, dtype="float32", always_2d=False)
            samples_16k = resample_float32(samples, int(sample_rate), SAMPLE_RATE)
            duration_ms = round(len(samples_16k) * 1000 / SAMPLE_RATE)
            start_ms = cursor_ms
            end_ms = start_ms + duration_ms
            expected.append(ExpectedSegment(part.reference_text, start_ms, end_ms))
            chunks.append(samples_16k)
            cursor_ms = end_ms
            if part_index < len(parts) - 1:
                chunks.append(np.zeros(SAMPLE_RATE * gap_ms // 1000, dtype="float32"))
                cursor_ms += gap_ms
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype="float32")
        synthetic_id = f"synthetic-{index + 1:04d}-gap-{gap_ms}"
        audio_path = out_dir / f"{synthetic_id}.wav"
        sf.write(audio_path, audio, SAMPLE_RATE)
        synthetic.append(
            ManifestItem(
                id=synthetic_id,
                dataset="synthetic",
                audio_path=str(audio_path),
                reference_text=" ".join(segment.text for segment in expected),
                group=f"gap-{gap_ms}",
                expected_segments=expected,
            )
        )
    return synthetic


async def run_stream_suite(args: argparse.Namespace) -> dict[str, Any]:
    manifest = read_manifest(Path(args.manifest))
    config_data = read_config(Path(args.config)) if args.config else {}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_dir = out_dir / "events"
    if args.save_events:
        events_dir.mkdir(exist_ok=True)

    results = []
    selected = manifest[: args.limit] if args.limit is not None else manifest
    for item in selected:
        result, events = await run_stream_item(item, config_data, realtime=args.realtime)
        results.append(result)
        if args.save_events:
            write_jsonl(events_dir / f"{item.id}.jsonl", events)

    result_dicts = [asdict(result) for result in results]
    write_jsonl(out_dir / "results.jsonl", result_dicts)
    summary = summarize_results(result_dicts, config_data)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


async def run_stream_item(
    item: ManifestItem,
    config_data: dict[str, Any],
    realtime: bool = False,
) -> tuple[StreamResult, list[Event]]:
    hub = EventHub(max_queue_size=10000)
    queue = await hub.subscribe()
    config = pipeline_config_for_item(item, config_data, realtime=realtime)
    runner = PipelineRunner(config, hub)
    task = asyncio.create_task(runner.run())
    events: list[Event] = []
    started = time.perf_counter()
    while not task.done() or not queue.empty():
        try:
            events.append(await asyncio.wait_for(queue.get(), timeout=0.05))
        except TimeoutError:
            continue
    summary = (await task).as_dict()
    await hub.unsubscribe(queue)
    wall_time = time.perf_counter() - started
    result = result_from_events(item, summary, events, wall_time)
    return result, events


def pipeline_config_for_item(item: ManifestItem, config_data: dict[str, Any], realtime: bool = False) -> PipelineConfig:
    values = dict(config_data.get("pipeline", config_data))
    values.pop("name", None)
    values.setdefault("source", "wav")
    values["wav_path"] = item.audio_path
    values["wav_realtime"] = bool(realtime)
    values["reference_text"] = item.reference_text
    values.setdefault("asr_mode", values.pop("asr", "sherpa"))
    values.setdefault("model_dir", str(Path("models") / DEFAULT_MODEL_NAME))
    allowed = set(PipelineConfig.__dataclass_fields__)
    return PipelineConfig(**{key: value for key, value in values.items() if key in allowed})


def result_from_events(
    item: ManifestItem,
    summary: dict[str, Any],
    events: list[Event],
    wall_time_sec: float,
) -> StreamResult:
    transcript_events = [event for event in events if event["type"] == "transcript"]
    partials = [event["payload"] for event in transcript_events if event["payload"].get("type") == "partial"]
    finals = final_segments_from_events(
        [event["payload"] for event in transcript_events if event["payload"].get("type") == "final"]
    )
    transcript = str(summary.get("transcript", ""))
    score = score_transcript(item.reference_text, transcript)
    expected = expected_segments_for_item(item)
    boundary = boundary_metrics(expected, finals)
    endpoint_latencies = endpoint_latencies_ms(expected, finals)
    tail = tail_loss_flags(item.reference_text, transcript)
    audio_duration_sec = float(summary.get("audio_duration_sec", 0.0) or 0.0)
    return StreamResult(
        id=item.id,
        dataset=item.dataset,
        group=item.group,
        audio_duration_sec=audio_duration_sec,
        wall_time_sec=round(wall_time_sec, 3),
        wall_rtf=round(wall_time_sec / audio_duration_sec, 3) if audio_duration_sec else None,
        transcript=transcript,
        cer=score.cer,
        wer=score.wer,
        segment_wer=segment_wer(expected, finals),
        segment_count=len(finals),
        sentence_count=len(finals),
        expected_segment_count=len(expected),
        boundary_precision=boundary["precision"],
        boundary_recall=boundary["recall"],
        boundary_f1=boundary["f1"],
        mean_endpoint_latency_ms=_mean(endpoint_latencies),
        p95_endpoint_latency_ms=_percentile(endpoint_latencies, 95),
        partial_count=len(partials),
        partial_churn_avg=partial_churn_avg(partials),
        first_word_missing=tail["first_word_missing"],
        last_word_missing=tail["last_word_missing"],
        tail_s_loss=tail["tail_s_loss"],
        metrics=dict(summary.get("metrics", {})),
    )


def expected_segments_for_item(item: ManifestItem) -> list[ExpectedSegment]:
    if item.expected_segments:
        return item.expected_segments
    return [ExpectedSegment(text=item.reference_text)]


def boundary_metrics(
    expected: list[ExpectedSegment],
    finals: list[dict[str, Any]],
    tolerance_ms: int = 800,
) -> dict[str, float | None]:
    expected_boundaries = _expected_boundary_windows(expected)
    predicted_boundaries = [int(final.get("end_ms", 0)) for final in finals[:-1]]
    if not expected_boundaries and not predicted_boundaries:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not expected_boundaries or not predicted_boundaries:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    matched: set[int] = set()
    true_positive = 0
    for predicted in predicted_boundaries:
        for index, (start_ms, end_ms) in enumerate(expected_boundaries):
            if index in matched:
                continue
            if start_ms - tolerance_ms <= predicted <= end_ms + tolerance_ms:
                matched.add(index)
                true_positive += 1
                break
    precision = true_positive / len(predicted_boundaries)
    recall = true_positive / len(expected_boundaries)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def final_segments_from_events(finals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_segment: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    fallback_id = 0
    for payload in finals:
        segment_id = payload.get("segment_id")
        if segment_id is None:
            fallback_id += 1
            segment_key = f"event-{fallback_id}"
        else:
            segment_key = str(segment_id)
        if segment_key not in latest_by_segment:
            order.append(segment_key)
        previous = latest_by_segment.get(segment_key)
        if previous is None or int(payload.get("revision", 0)) >= int(previous.get("revision", 0)):
            latest_by_segment[segment_key] = payload
    return [latest_by_segment[segment_key] for segment_key in order]


def _expected_boundary_windows(expected: list[ExpectedSegment]) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for index, segment in enumerate(expected[:-1]):
        next_segment = expected[index + 1]
        if segment.end_ms is None and next_segment.start_ms is None:
            continue
        if segment.end_ms is None:
            start_ms = end_ms = int(next_segment.start_ms or 0)
        elif next_segment.start_ms is None:
            start_ms = end_ms = int(segment.end_ms)
        else:
            start_ms = min(int(segment.end_ms), int(next_segment.start_ms))
            end_ms = max(int(segment.end_ms), int(next_segment.start_ms))
        windows.append((start_ms, end_ms))
    return windows


def endpoint_latencies_ms(expected: list[ExpectedSegment], finals: list[dict[str, Any]]) -> list[float]:
    latencies = []
    for expected_segment, final in zip(expected, finals):
        if expected_segment.end_ms is None:
            continue
        latencies.append(float(final.get("end_ms", 0)) - expected_segment.end_ms)
    return latencies


def segment_wer(expected: list[ExpectedSegment], finals: list[dict[str, Any]]) -> float | None:
    reference_words = 0
    edits = 0
    for expected_segment, final in zip_longest(expected, finals):
        ref_words = normalize_words(expected_segment.text) if expected_segment is not None else []
        hyp_words = normalize_words(str(final.get("text", ""))) if final is not None else []
        reference_words += len(ref_words)
        edits += edit_distance(ref_words, hyp_words)
    return edits / reference_words if reference_words else None


def partial_churn_avg(partials: list[dict[str, Any]]) -> float:
    if len(partials) < 2:
        return 0.0
    distances = []
    previous = normalize_words(str(partials[0].get("text", "")))
    for payload in partials[1:]:
        current = normalize_words(str(payload.get("text", "")))
        denominator = max(1, len(previous), len(current))
        distances.append(edit_distance(previous, current) / denominator)
        previous = current
    return round(sum(distances) / len(distances), 4) if distances else 0.0


def tail_loss_flags(reference: str, hypothesis: str) -> dict[str, bool]:
    ref_words = normalize_words(reference)
    hyp_words = normalize_words(hypothesis)
    first_missing = bool(ref_words) and (not hyp_words or hyp_words[0] != ref_words[0])
    last_missing = bool(ref_words) and (not hyp_words or hyp_words[-1] != ref_words[-1])
    tail_s_loss = (
        bool(ref_words)
        and bool(hyp_words)
        and ref_words[-1].endswith("s")
        and hyp_words[-1] == ref_words[-1][:-1]
    )
    return {
        "first_word_missing": first_missing,
        "last_word_missing": last_missing,
        "tail_s_loss": tail_s_loss,
    }


def summarize_results(results: list[dict[str, Any]], config_data: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        groups.setdefault(str(item["dataset"]), []).append(item)
    return {
        "config": config_data,
        "count": len(results),
        "overall": summarize_group(results),
        "by_dataset": {name: summarize_group(items) for name, items in sorted(groups.items())},
    }


def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
    wers = [item["wer"] for item in items if item.get("wer") is not None]
    boundary_f1 = [item["boundary_f1"] for item in items if item.get("boundary_f1") is not None]
    rtfs = [item["wall_rtf"] for item in items if item.get("wall_rtf") is not None]
    endpoint = [item["mean_endpoint_latency_ms"] for item in items if item.get("mean_endpoint_latency_ms") is not None]
    return {
        "count": len(items),
        "wer": _mean(wers),
        "segment_wer": _mean([item["segment_wer"] for item in items if item.get("segment_wer") is not None]),
        "sentence_count": _mean([item["sentence_count"] for item in items if item.get("sentence_count") is not None]),
        "boundary_f1": _mean(boundary_f1),
        "wall_rtf": _mean(rtfs),
        "mean_endpoint_latency_ms": _mean(endpoint),
        "first_word_missing_rate": _rate(items, "first_word_missing"),
        "last_word_missing_rate": _rate(items, "last_word_missing"),
        "tail_s_loss_rate": _rate(items, "tail_s_loss"),
        "partial_churn_avg": _mean([item["partial_churn_avg"] for item in items]),
    }


def compare_runs(runs: list[str]) -> str:
    rows = []
    for run in runs:
        path = Path(run)
        summary_path = path / "summary.json" if path.is_dir() else path
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        overall = data.get("overall", {})
        rows.append(
            [
                str(path),
                overall.get("count"),
                _format_float(overall.get("wer")),
                _format_float(overall.get("boundary_f1")),
                _format_float(overall.get("wall_rtf")),
                _format_float(overall.get("last_word_missing_rate")),
            ]
        )
    header = ["run", "n", "wer", "boundary_f1", "rtf", "tail_missing"]
    widths = [max(len(str(row[i])) for row in [header, *rows]) for i in range(len(header))]
    lines = ["  ".join(str(value).ljust(widths[i]) for i, value in enumerate(header))]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))
    return "\n".join(lines)


def read_manifest(path: Path) -> list[ManifestItem]:
    return [ManifestItem.from_json(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_manifest(items: list[ManifestItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(path, [item.to_json() for item in items])


def read_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _prepare_librispeech(cache_dir: Path, download: bool, provided: str | None) -> Path:
    if provided:
        return Path(provided)
    root = cache_dir / "librispeech"
    if download:
        for split, url in LIBRISPEECH_URLS.items():
            archive = root / f"{split}.tar.gz"
            if not archive.exists():
                _download(url, archive)
            _safe_extract(archive, root)
    return root


def _prepare_ljspeech(cache_dir: Path, download: bool, provided: str | None) -> Path:
    if provided:
        return Path(provided)
    root = cache_dir / "ljspeech"
    if download:
        archive = root / "LJSpeech-1.1.tar.bz2"
        if not archive.exists():
            _download(LJSPEECH_URL, archive)
        _safe_extract(archive, root)
    return root


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(destination)


def _safe_extract(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    mode = "r:bz2" if archive_path.name.endswith(".bz2") else "r:gz"
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, mode) as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not str(target).startswith(str(destination_resolved)):
                raise RuntimeError(f"Refusing to extract path outside destination: {member.name}")
        archive.extractall(destination)


def _first_existing(root: Path, names: list[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    return None


def _optional_int(value) -> int | None:
    return int(value) if value is not None else None


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((percentile / 100) * (len(ordered) - 1))))
    return round(ordered[index], 3)


def _rate(items: list[dict[str, Any]], key: str) -> float | None:
    if not items:
        return None
    return round(sum(1 for item in items if item.get(key)) / len(items), 4)


def _format_float(value) -> str:
    return "-" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    main()
