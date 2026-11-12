import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from anpr.recognize import PlateRead
from anpr.types import PlateBox
from anpr.vote import PlateVoteTracker


def _read(
    text: str,
    confidence: float,
    *,
    x: int = 10,
    margins: tuple[float, ...] | None = None,
) -> PlateRead:
    return PlateRead(
        text=text,
        confidence=confidence,
        box=PlateBox(x, 20, 100, 40, 0.8),
        char_confidences=tuple(confidence for _ in text),
        char_margins=margins or tuple(0.2 for _ in text),
        blank_uncertainty=0.0,
        laplacian_var=80.0,
        brightness=120.0,
        contrast=40.0,
    )


def _feed(tracker: PlateVoteTracker, text: str, confidences: tuple[float, ...], **kw):
    return [tracker.consider("cam", _read(text, c, **kw)) for c in confidences]


def _settled(tracker: PlateVoteTracker, text: str = "AXKW568") -> None:
    out = _feed(tracker, text, (0.8, 0.8))
    assert out[0] is None and out[1] is not None and out[1].text == text


def test_release_needs_second_agreeing_sample() -> None:
    tracker = PlateVoteTracker()
    assert tracker.consider("cam", _read("AXKW568", 0.9)) is None
    out = tracker.consider("cam", _read("AXKW568", 0.9))
    assert out is not None
    assert out.text == "AXKW568"


def test_two_weak_agreeing_reads_do_not_release() -> None:
    tracker = PlateVoteTracker()
    assert _feed(tracker, "AXKW568", (0.5, 0.5)) == [None, None]


def test_unsure_margin_read_is_down_weighted() -> None:
    tracker = PlateVoteTracker()
    unsure = (0.02, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)
    assert _feed(tracker, "AXKW568", (0.9, 0.9), margins=unsure) == [None, None]


def test_longer_spelling_does_not_win_by_length() -> None:
    tracker = PlateVoteTracker()
    _settled(tracker, "AKW568")
    # A spurious leading character is a different spelling, not an upgrade.
    assert _feed(tracker, "JAKW568", (0.8, 0.8)) == [None, None]
    out = tracker.consider("cam", _read("AKW568", 0.8))
    assert out is not None and out.text == "AKW568"


def test_one_off_substitution_does_not_override_sticky() -> None:
    tracker = PlateVoteTracker()
    _settled(tracker)
    assert tracker.consider("cam", _read("CAKW568", 0.71)) is None
    assert tracker.consider("cam", _read("CAKW568", 0.71)) is None
    out = tracker.consider("cam", _read("AXKW568", 0.78))
    assert out is not None and out.text == "AXKW568"


def test_truncation_does_not_replace_sticky() -> None:
    tracker = PlateVoteTracker()
    _feed(tracker, "AKW568", (0.9, 0.9, 0.9, 0.9))
    assert _feed(tracker, "AKW58", (0.99, 0.99, 0.99)) == [None, None, None]


def test_higher_confidence_alone_does_not_flip_the_plate() -> None:
    tracker = PlateVoteTracker()
    _feed(tracker, "AXKW568", (0.78, 0.78, 0.78, 0.78))
    assert _feed(tracker, "CAKW568", (0.95, 0.95)) == [None, None]


def test_sustained_challenger_switches() -> None:
    tracker = PlateVoteTracker()
    _settled(tracker, "AKW568")
    outs = _feed(tracker, "AMW568", (0.95,) * 4)
    assert outs[:2] == [None, None]
    assert outs[-1] is not None and outs[-1].text == "AMW568"


def test_stream_sticky_survives_box_jump() -> None:
    tracker = PlateVoteTracker()
    _settled(tracker, "AKW568")
    # Far box starts a new track; the stream sticky still blocks a one-off typo.
    assert tracker.consider("cam", _read("AMW568", 0.99, x=400)) is None
    out = tracker.consider("cam", _read("AKW568", 0.9, x=400))
    assert out is not None and out.text == "AKW568"


def _fused_read(text: str, x: int = 10, sure: float = 0.999) -> PlateRead:
    from test_fusion import ALPHABET, _frame

    steps = tuple(tuple(step) for step in _frame(text, sure=sure))
    assert len(ALPHABET) == 37
    base = _read(text, 0.9, x=x)
    from dataclasses import replace

    return replace(base, step_probs=steps)


def test_fusion_releases_fast_and_ignores_a_garbage_frame() -> None:
    from anpr.fusion import FusionSettings

    tracker = PlateVoteTracker(FusionSettings())
    assert tracker.consider("cam", _fused_read("CAKW568")) is None  # one frame is not enough
    out = tracker.consider("cam", _fused_read("CAKW568"))
    assert out is not None and out.text == "CAKW568"
    tracker.consider("cam", _fused_read("CAKW568"))
    tracker.consider("cam", _fused_read("CAKW568"))
    # An uncertain truncated frame is reported as the fused plate, not as itself.
    out = tracker.consider("cam", _fused_read("AKW58", sure=0.6))
    assert out is not None and out.text == "CAKW568"
    assert out.confidence >= 0.9


def test_fusion_crisp_contradiction_withholds_instead_of_flipping() -> None:
    from anpr.fusion import FusionSettings

    tracker = PlateVoteTracker(FusionSettings())
    for _ in range(2):
        tracker.consider("cam", _fused_read("CAKW568"))
    assert tracker.consider("cam", _fused_read("AKW58")) is None
    for _ in range(3):
        out = tracker.consider("cam", _fused_read("CAKW568"))
    assert out is not None and out.text == "CAKW568"


def test_fusion_falls_back_to_text_vote_without_probabilities() -> None:
    from anpr.fusion import FusionSettings

    tracker = PlateVoteTracker(FusionSettings())
    assert tracker.consider("cam", _read("AKW568", 0.95)) is None
    out = tracker.consider("cam", _read("AKW568", 0.96))
    assert out is not None and out.text == "AKW568"


def test_publisher_reset_drops_vote_tracks() -> None:
    from anpr.publisher import ReadPublisher
    from anpr.results import ResultStore

    store = ResultStore(4)
    publisher = ReadPublisher(store, 0.5, 0.5, 80)
    publisher._voter.consider("cam", _read("AKW568", 0.95))
    assert publisher._voter._tracks
    store.reset()
    assert not publisher._voter._tracks
