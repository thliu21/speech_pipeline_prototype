from speech_proto.asr import (
    DEFAULT_MODEL_NAME,
    ENGLISH_MODEL_NAME,
    SherpaOnnxAsrEngine,
    resolve_transducer_model_paths,
)


class FakeStream:
    def __init__(self) -> None:
        self.finished = False
        self.waveforms: list[tuple[int, int]] = []

    def accept_waveform(self, sample_rate: int, samples) -> None:
        self.waveforms.append((sample_rate, len(samples)))

    def input_finished(self) -> None:
        self.finished = True


class FakeRecognizer:
    def __init__(self) -> None:
        self.decode_calls = 0
        self.ready_checks = 0
        self.streams = [FakeStream(), FakeStream()]
        self.created_streams: list[FakeStream] = []

    def is_ready(self, stream: FakeStream) -> bool:
        assert stream.finished is True
        self.ready_checks += 1
        return self.ready_checks == 1

    def decode_stream(self, stream: FakeStream) -> None:
        assert stream.finished is True
        self.decode_calls += 1

    def get_result(self, stream: FakeStream) -> str:
        assert stream.finished is True
        return "sunnyvale"

    def create_stream(self) -> FakeStream:
        stream = self.streams.pop(0)
        self.created_streams.append(stream)
        return stream


def test_sherpa_finalize_marks_input_finished_before_final_decode():
    recognizer = FakeRecognizer()
    engine = SherpaOnnxAsrEngine.__new__(SherpaOnnxAsrEngine)
    engine._recognizer = recognizer
    engine._stream = recognizer.create_stream()
    engine._last_decode_ms = 1000
    engine._last_text = "sunny"

    transcript = engine.finalize(100, 900)

    assert transcript is not None
    assert transcript.type == "final"
    assert transcript.text == "sunnyvale"
    assert recognizer.decode_calls == 1
    assert engine._stream is recognizer.created_streams[1]
    assert engine._last_decode_ms == 0
    assert engine._last_text == ""


def test_sherpa_context_padding_wraps_each_stream():
    recognizer = FakeRecognizer()
    engine = SherpaOnnxAsrEngine.__new__(SherpaOnnxAsrEngine)
    engine._recognizer = recognizer
    engine._context_padding_samples = [0.0] * 8000
    engine._stream = engine._create_stream()
    engine._last_decode_ms = 1000
    engine._last_text = "sunny"

    old_stream = engine._stream
    engine.finalize(100, 900)

    assert old_stream.waveforms == [(16000, 8000), (16000, 8000)]
    assert engine._stream.waveforms == [(16000, 8000)]


def test_default_model_is_english_only():
    assert DEFAULT_MODEL_NAME == ENGLISH_MODEL_NAME


def test_resolve_transducer_prefers_fp32_decoder_with_int8_encoder_joiner(tmp_path):
    names = [
        "encoder-epoch-99-avg-1.int8.onnx",
        "encoder-epoch-99-avg-1.onnx",
        "decoder-epoch-99-avg-1.int8.onnx",
        "decoder-epoch-99-avg-1.onnx",
        "joiner-epoch-99-avg-1.int8.onnx",
        "joiner-epoch-99-avg-1.onnx",
        "tokens.txt",
    ]
    for name in names:
        (tmp_path / name).write_text("x")

    paths = resolve_transducer_model_paths(tmp_path, use_int8=True)

    assert paths["encoder"].name == "encoder-epoch-99-avg-1.int8.onnx"
    assert paths["decoder"].name == "decoder-epoch-99-avg-1.onnx"
    assert paths["joiner"].name == "joiner-epoch-99-avg-1.int8.onnx"


def test_resolve_transducer_supports_chunked_english_model_names(tmp_path):
    names = [
        "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        "tokens.txt",
    ]
    for name in names:
        (tmp_path / name).write_text("x")

    paths = resolve_transducer_model_paths(tmp_path, use_int8=True)

    assert paths["encoder"].name == "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
    assert paths["decoder"].name == "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"
    assert paths["joiner"].name == "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
