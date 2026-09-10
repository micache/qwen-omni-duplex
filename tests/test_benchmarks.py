from duplex.metrics import FULL_DUPLEX_BENCH_VARIANT, OFFICIAL_SPEECH_OUTPUT_SCORE


def test_full_duplex_bench_is_labeled_as_adaptation() -> None:
    assert FULL_DUPLEX_BENCH_VARIANT == "text-timeline adaptation"
    assert OFFICIAL_SPEECH_OUTPUT_SCORE is False
