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
    chars = tuple(confidence for _ in text)
    return PlateRead(
        text=text,
        confidence=confidence,
        box=PlateBox(x, 20, 100, 40, 0.8),
        char_confidences=chars,
        char_margins=margins or tuple(0.2 for _ in text),
        blank_uncertainty=0.0,
        laplacian_var=80.0,
        brightness=120.0,
        contrast=40.0,
    )


def test_high_trust_needs_second_agreeing_sample() -> None:
    tracker = PlateVoteTracker()
    assert tracker.consider("cam", _read("CAKW568", 0.85)) is None
    out = tracker.consider("cam", _read("CAKW568", 0.88))
    assert out is not None
    assert out.text == "CAKW568"


def test_majority_prefers_longer_suffix() -> None:
    tracker = PlateVoteTracker()
    weak = (0.05, 0.2, 0.2, 0.2, 0.2, 0.2)
    assert tracker.consider("cam", _read("AKW568", 0.2, margins=weak)) is None
    out = tracker.consider(
        "cam",
        _read("CAKW568", 0.2, margins=weak + (0.2,)),
    )
    assert out is not None
    assert out.text == "CAKW568"


def test_low_margin_does_not_early_release() -> None:
    tracker = PlateVoteTracker()
    out = tracker.consider(
        "cam",
        _read("CAKW568", 0.9, margins=(0.02, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2)),
    )
    assert out is None


def test_single_frame_near_duplicate_does_not_override_sticky() -> None:
    tracker = PlateVoteTracker()
    assert tracker.consider("cam", _read("AKW568", 0.95)) is None
    first = tracker.consider("cam", _read("AKW568", 0.96))
    assert first is not None and first.text == "AKW568"
    assert tracker.consider("cam", _read("AMW568", 0.99)) is None
    assert tracker.consider("cam", _read("AMW568", 0.98)) is None
    switched = tracker.consider("cam", _read("AMW568", 0.97))
    assert switched is not None
    assert switched.text == "AMW568"


def test_sticky_blocks_shorter_suffix_high_trust() -> None:
    tracker = PlateVoteTracker()
    assert tracker.consider("cam", _read("CAKW568", 0.9)) is None
    assert tracker.consider("cam", _read("CAKW568", 0.91)).text == "CAKW568"
    # Shorter suffix must not replace sticky; majority may reaffirm the longer text.
    out = tracker.consider("cam", _read("AKW568", 0.99))
    assert out is None or out.text == "CAKW568"
    out = tracker.consider("cam", _read("AKW568", 0.99))
    assert out is None or out.text == "CAKW568"


def test_middle_insertion_does_not_beat_sticky() -> None:
    tracker = PlateVoteTracker()
    assert tracker.consider("cam", _read("AKW568", 0.95)) is None
    assert tracker.consider("cam", _read("AKW568", 0.96)).text == "AKW568"
    assert tracker.consider("cam", _read("AXKW568", 0.99)) is None
    assert tracker.consider("cam", _read("AXKW568", 0.98)) is None


def test_stream_sticky_survives_box_jump() -> None:
    tracker = PlateVoteTracker()
    assert tracker.consider("cam", _read("AKW568", 0.95, x=10)) is None
    assert tracker.consider("cam", _read("AKW568", 0.96, x=10)).text == "AKW568"
    # Far box → new track, but stream sticky still blocks a one-off typo.
    assert tracker.consider("cam", _read("AMW568", 0.99, x=400)) is None


def test_longer_prefix_upgrades_sticky() -> None:
    tracker = PlateVoteTracker()
    assert tracker.consider("cam", _read("AKW568", 0.95)) is None
    assert tracker.consider("cam", _read("AKW568", 0.96)).text == "AKW568"
    out = tracker.consider("cam", _read("CAKW568", 0.80))
    assert out is not None
    assert out.text == "CAKW568"


def test_truncation_does_not_replace_sticky() -> None:
    tracker = PlateVoteTracker()
    assert tracker.consider("cam", _read("AKW568", 0.95)) is None
    assert tracker.consider("cam", _read("AKW568", 0.96)).text == "AKW568"
    for conf in (0.99, 0.98, 0.97):
        out = tracker.consider("cam", _read("AKW58", conf))
        assert out is None or out.text == "AKW568"
