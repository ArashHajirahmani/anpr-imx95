import threading
import time

from anpr.pipeline import LatestFrameSlot
from anpr.recognize import DetectedFrame, OcrOutcome, PlateRead
from anpr.staged import StagedRecognizer
from anpr.types import Frame, PlateBox

BOX = ((0, 0, 4, 4, 0.9),)


class _Reader:
    """Boxes on odd frames; OCR is slow for early frames so workers finish out of order."""

    def __init__(self, fail_on: int | None = None) -> None:
        self.fail_on = fail_on
        self.detect_thread: set[str] = set()
        self.ocr_threads: set[str] = set()

    def detect(self, frame: Frame) -> DetectedFrame:
        self.detect_thread.add(threading.current_thread().name)
        boxes = BOX if frame.monotonic_ns % 2 else ()
        return DetectedFrame(frame, None, boxes, "cpu", 1.0, 0.0)

    def recognize(self, detected: DetectedFrame) -> OcrOutcome:
        self.ocr_threads.add(threading.current_thread().name)
        number = detected.frame.monotonic_ns
        if number == self.fail_on:
            raise RuntimeError("boom")
        time.sleep(0.05 if number % 3 == 0 else 0.005)
        plate = PlateRead("ABC123", 0.9, PlateBox(0, 0, 4, 4, 0.9))
        return OcrOutcome([plate], 2.0, 0.0)


class _Publisher:
    def __init__(self) -> None:
        self.order: list[int] = []
        self.reads: list[int] = []

    def handle(self, name: str, frame: Frame, reads: list[object], stats: object) -> None:
        self.order.append(frame.monotonic_ns)
        self.reads.append(len(reads))


def _run(reader: _Reader, frames: int, workers: int = 3) -> _Publisher:
    slot = LatestFrameSlot()
    stop = threading.Event()
    publisher = _Publisher()
    staged = StagedRecognizer({"cam": slot}, stop, reader, 1000, workers, publisher)
    threads = staged.threads()
    for thread in threads:
        thread.start()
    for number in range(1, frames + 1):
        slot.publish(Frame("cam", b"x", number, 4, 4))
        time.sleep(0.01)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and len(publisher.order) < 3:
        time.sleep(0.02)
    time.sleep(0.4)
    stop.set()
    for thread in threads:
        thread.join(timeout=2)
    return publisher


def test_frames_publish_in_capture_order_despite_out_of_order_ocr() -> None:
    reader = _Reader()
    publisher = _run(reader, 30)
    assert len(publisher.order) >= 10
    assert publisher.order == sorted(publisher.order)
    assert len(set(publisher.order)) == len(publisher.order)
    # Frames without a box still reach the publisher, with no reads.
    empty = [n for n, count in zip(publisher.order, publisher.reads, strict=True) if count == 0]
    assert empty and all(n % 2 == 0 for n in empty)


def test_detector_and_ocr_run_on_separate_threads() -> None:
    reader = _Reader()
    _run(reader, 20)
    assert reader.detect_thread == {"plate-detector"}
    assert reader.ocr_threads
    assert all(name.startswith("plate-ocr-") for name in reader.ocr_threads)
    assert len(reader.ocr_threads) > 1


def test_failed_frame_does_not_stall_later_frames() -> None:
    reader = _Reader(fail_on=5)
    publisher = _run(reader, 30)
    assert 5 not in publisher.order
    assert any(n > 5 for n in publisher.order)
    assert publisher.order == sorted(publisher.order)
