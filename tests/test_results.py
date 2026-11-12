import pytest

from anpr.recognize import PlateRead
from anpr.results import FrameTiming, ResultStore, encode_jpeg
from anpr.types import Frame, PlateBox

JPEG = b"\xff\xd8\xff\xd9"


def test_record_keeps_two_distinct_slots() -> None:
    store = ResultStore(2)
    seen: dict[str, set[str]] = {}
    for index, text in enumerate(["AAA111", "BBB222", "CCC333"]):
        fresh = store.record(
            _frame(index),
            [_plate(text)],
            min_confidence=0.07,
            jpeg=JPEG + bytes([index]),
            last_texts=seen,
        )
        assert [item.text for item in fresh] == [text]
    history = store.history()
    assert [item.text for item in history] == ["CCC333", "BBB222"]
    latest = store.latest()
    assert latest is not None
    assert latest.frame_id == 3
    assert latest.plates[0].text == "CCC333"


def test_partial_match_upgrades_same_slot() -> None:
    store = ResultStore(2)
    seen: dict[str, set[str]] = {}
    for text, confidence in [
        ("AKW568", 0.90),
        ("CAKW568", 0.80),
        ("AKW568", 0.95),
        ("AKW568", 0.99),
    ]:
        store.record(
            _frame(),
            [_plate(text, confidence=confidence)],
            min_confidence=0.07,
            jpeg=JPEG,
            last_texts=seen,
        )
    history = store.history()
    assert len(history) == 1
    assert history[0].text == "CAKW568"
    assert history[0].confidence == 0.99


def test_higher_conf_correct_spelling_beats_garbage() -> None:
    store = ResultStore(2)
    seen: dict[str, set[str]] = {}
    store.record(
        _frame(1),
        [_plate("AXKW568", confidence=0.77)],
        min_confidence=0.07,
        jpeg=JPEG,
        last_texts=seen,
    )
    store.record(
        _frame(2),
        [_plate("CAKW568", confidence=0.94)],
        min_confidence=0.07,
        jpeg=JPEG + b"\x01",
        last_texts=seen,
    )
    store.record(
        _frame(3),
        [_plate("AXKW568", confidence=0.80)],
        min_confidence=0.07,
        jpeg=JPEG + b"\x02",
        last_texts=seen,
    )
    history = store.history()
    assert len(history) == 1
    assert history[0].text == "CAKW568"
    assert history[0].confidence == 0.94


def test_near_duplicate_does_not_open_second_slot() -> None:
    store = ResultStore(2)
    seen: dict[str, set[str]] = {}
    store.record(
        _frame(1),
        [_plate("AKW568", confidence=0.8)],
        min_confidence=0.07,
        jpeg=JPEG,
        last_texts=seen,
    )
    store.record(
        _frame(2),
        [_plate("AMW568", confidence=0.95)],
        min_confidence=0.07,
        jpeg=JPEG + b"\x01",
        last_texts=seen,
    )
    assert [item.text for item in store.history()] == ["AMW568"]


def test_weak_character_is_not_published() -> None:
    store = ResultStore(2)
    seen: dict[str, set[str]] = {}
    fresh = store.record(
        _frame(),
        [
            PlateRead(
                "AB",
                0.5,
                PlateBox(4, 6, 30, 12, 0.8),
                (0.9, 0.02),
            )
        ],
        min_confidence=0.07,
        min_char_confidence=0.07,
        jpeg=JPEG,
        last_texts=seen,
    )
    assert fresh == []
    assert store.latest() is not None
    assert store.latest().plates == ()


def test_confidence_floor_is_inclusive() -> None:
    store = ResultStore(2)
    seen: dict[str, set[str]] = {}
    store.record(
        _frame(),
        [_plate("ABC123", confidence=0.07)],
        min_confidence=0.07,
        jpeg=JPEG,
        last_texts=seen,
    )
    assert store.history()[0].text == "ABC123"


def test_two_plates_keep_both_boxes() -> None:
    store = ResultStore(2)
    seen: dict[str, set[str]] = {}
    fresh = store.record(
        _frame(),
        [_plate("AAA", confidence=0.2), _plate("BBB", confidence=0.9, x=10)],
        min_confidence=0.07,
        jpeg=JPEG,
        last_texts=seen,
    )
    assert [item.text for item in fresh] == ["AAA", "BBB"]
    latest = store.latest()
    assert latest is not None
    plates = latest.json()["plates"]
    assert [plate["text"] for plate in plates] == ["BBB", "AAA"]
    assert plates[0]["characters"] == [
        {"char": "B", "confidence": 0.9},
        {"char": "B", "confidence": 0.9},
        {"char": "B", "confidence": 0.9},
    ]
    assert [item.text for item in store.history()] == ["BBB", "AAA"]
    assert latest.plates[0].box.width == 30


def test_frame_timing_is_published_with_the_plate() -> None:
    store = ResultStore(2)
    seen: dict[str, set[str]] = {}
    store.record(
        _frame(),
        [_plate("ABC123")],
        min_confidence=0.07,
        jpeg=JPEG,
        last_texts=seen,
        timing=FrameTiming(977.24, 189.06, "npu"),
    )
    latest = store.latest()
    assert latest is not None
    body = latest.json()
    assert body["detector_ms"] == 977.2
    assert body["reader_ms"] == 189.1
    assert body["runtime"] == "npu"
    assert store.history()[0].json()["detector_ms"] == 977.2
    assert store.history()[0].json()["reader_ms"] == 189.1
    assert store.history()[0].json()["runtime"] == "npu"


def test_history_image_is_the_plate_crop() -> None:
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    image = numpy.zeros((8, 12, 3), dtype=numpy.uint8)
    image[2:6, 3:9] = (20, 40, 60)
    frame = Frame("videoCapture", image.tobytes(), 1, 12, 8)
    store = ResultStore(2)
    store.record(
        frame,
        [PlateRead("ABC123", 0.9, PlateBox(3, 2, 6, 4, 0.8))],
        min_confidence=0.07,
        jpeg=b"full-frame",
        last_texts={},
        jpeg_quality=95,
    )
    item = store.history()[0]
    assert item.width == 6
    assert item.height == 4
    assert item.jpeg != b"full-frame"
    decoded = cv2.imdecode(numpy.frombuffer(item.jpeg, dtype=numpy.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (4, 6, 3)
    latest = store.latest()
    assert latest is not None
    assert latest.jpeg == b"full-frame"
    assert latest.width == 12


def test_history_image_prefers_pre_ocr_enhancer_output() -> None:
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")
    frame_image = numpy.zeros((8, 12, 3), dtype=numpy.uint8)
    frame_image[2:6, 3:9] = (20, 40, 60)
    pre_ocr = numpy.full((5, 7, 3), (90, 100, 110), dtype=numpy.uint8)
    frame = Frame("videoCapture", frame_image.tobytes(), 1, 12, 8)
    store = ResultStore(2)
    store.record(
        frame,
        [
            PlateRead(
                "ABC123",
                0.9,
                PlateBox(3, 2, 6, 4, 0.8),
                pre_ocr_bgr=pre_ocr.tobytes(),
                pre_ocr_width=7,
                pre_ocr_height=5,
            )
        ],
        min_confidence=0.07,
        jpeg=b"full-frame",
        last_texts={},
        jpeg_quality=95,
    )
    item = store.history()[0]
    assert item.width == 7
    assert item.height == 5
    decoded = cv2.imdecode(numpy.frombuffer(item.jpeg, dtype=numpy.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (5, 7, 3)
    assert int(decoded.mean()) == pytest.approx(100, abs=5)


def test_encode_jpeg_round_trip() -> None:
    cv2 = pytest_cv2()
    numpy = pytest_numpy()
    image = numpy.zeros((4, 6, 3), dtype=numpy.uint8)
    image[:, :] = (10, 20, 30)
    frame = Frame("videoCapture", image.tobytes(), 1, 6, 4)
    payload = encode_jpeg(frame, 95)
    assert payload is not None
    assert payload.startswith(b"\xff\xd8")
    decoded = cv2.imdecode(numpy.frombuffer(payload, dtype=numpy.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (4, 6, 3)


def test_better_crop_replaces_slot_image() -> None:
    store = ResultStore(2)
    seen: dict[str, set[str]] = {}
    store.record(
        _frame(1),
        [_plate("AKW568", confidence=0.7)],
        min_confidence=0.07,
        jpeg=JPEG + b"\x01",
        last_texts=seen,
    )
    first = store.history()[0].jpeg
    store.record(
        _frame(2),
        [_plate("AKW568", confidence=0.95)],
        min_confidence=0.07,
        jpeg=JPEG + b"\x02",
        last_texts=seen,
    )
    assert len(store.history()) == 1
    assert store.history()[0].confidence == 0.95
    assert store.history()[0].jpeg != first


def _frame(stamp: int = 1) -> Frame:
    return Frame("videoCapture", b"raw", stamp, 100, 50)


def _plate(text: str, confidence: float = 0.9, x: int = 4) -> PlateRead:
    return PlateRead(
        text,
        confidence,
        PlateBox(x, 6, 30, 12, 0.8),
        tuple(confidence for _ in text),
    )


def pytest_cv2() -> object:
    return pytest.importorskip("cv2")


def pytest_numpy() -> object:
    return pytest.importorskip("numpy")


def test_store_reports_published_frames_per_second() -> None:
    store = ResultStore(4)
    assert store._measure_fps(0.0) is None  # one sample is not a rate
    assert store._measure_fps(0.5) == pytest.approx(2.0)
    for tick in range(1, 6):
        fps = store._measure_fps(0.5 + tick * 0.1)
    assert fps == pytest.approx(6.0)  # 7 samples over 1.0 s = 6 intervals
    # Frames older than the window drop out, so a long pause is not averaged in.
    assert store._measure_fps(100.0) is None
    assert store._measure_fps(100.1) == pytest.approx(10.0)


def test_latest_json_carries_fps_after_two_frames() -> None:
    store = ResultStore(4)
    frame = Frame("videoCapture", b"raw", 5, 100, 50)
    for _ in range(2):
        store.record(frame, [], min_confidence=0.07, jpeg=b"\xff\xd8\xff\xd9", last_texts={})
    latest = store.latest()
    assert latest is not None
    assert latest.json()["fps"] > 0
