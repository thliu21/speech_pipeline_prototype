from speech_proto.benchmark import edit_distance, normalize_chars, normalize_words, score_transcript


def test_edit_distance():
    assert edit_distance(list("kitten"), list("sitting")) == 3


def test_mixed_language_normalization():
    assert normalize_chars("打开 light!") == list("打开light")
    assert normalize_words("打开 light!") == ["打", "开", "light"]


def test_score_transcript_returns_cer_and_wer():
    score = score_transcript("打开 light", "打开 right")

    assert score.cer is not None
    assert score.wer is not None
    assert score.reference_chars == 7
    assert score.reference_words == 3

