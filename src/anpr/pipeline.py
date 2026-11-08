"""Latest-frame slots and the capture threads that fill them."""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from anpr.capture import pipeline_for_stream
from anpr.config import AppConfig
from anpr.plates import ocr_floor
from anpr.results import FrameTiming, ResultStore, publish_reads
from anpr.types import Frame

logger = logging.getLogger("anpr.pipeline")
_OPEN_RETRY_SECONDS = 0.5
_MAX_RETRY_SECONDS = 5.0


class LatestFrameSlot:
    """Holds one frame. A newer publish replaces the frame still waiting."""

    def __init__(self) -> None:
        self._frame: Frame | None = None
        self._lock = threading.Lock()

    def publish(self, frame: Frame) -> None:
        with self._lock:
            self._frame = frame

    def take(self) -> Frame | None:
        with self._lock:
            frame = self._frame
            self._frame = None
            return frame


@dataclass(frozen=True)
class WiredPipeline:
    pipelines: dict[str, str]
    slots: dict[str, LatestFrameSlot]


def wire(config: AppConfig) -> WiredPipeline:
    """Build one pipeline string and one slot for each enabled stream."""
    pipelines: dict[str, str] = {}
    slots: dict[str, LatestFrameSlot] = {}
    for stream in config.enabled_streams():
        pipelines[stream.name] = pipeline_for_stream(stream)
        slots[stream.name] = LatestFrameSlot()
    return WiredPipeline(pipelines=pipelines, slots=slots)


def run_streams(
    config: AppConfig,
    stop: threading.Event,
    open_pipeline: Callable[[str], object] | None = None,
    *,
    results: ResultStore | None = None,
    reader: object | None = None,
) -> WiredPipeline:
    """Capture until stop is set. Each thread publishes only its latest frame."""
    if open_pipeline is None:
        from anpr.capture import GStreamerCapture

        open_pipeline = GStreamerCapture
    wired = wire(config)
    for name, description in wired.pipelines.items():
        logger.info("stream %s pipeline: %s", name, description)
    threads = [
        threading.Thread(
            target=_capture_loop,
            name=f"capture-{name}",
            args=(name, description, wired.slots[name], stop, open_pipeline),
            daemon=True,
        )
        for name, description in wired.pipelines.items()
    ]
    plate_reader = reader if reader is not None else _load_reader(config)
    if plate_reader is not None:
        threads.append(
            threading.Thread(
                target=_infer_loop,
                name="plate-reader",
                args=(
                    wired.slots,
                    stop,
                    plate_reader,
                    config.service.inference_fps_limit,
                    ocr_floor(config.recognizer.min_confidence, config.service.runtime),
                    ocr_floor(config.recognizer.min_char_confidence, config.service.runtime),
                    results,
                    config.service.jpeg_quality,
                ),
                daemon=True,
            )
        )
    for thread in threads:
        thread.start()
    try:
        while not stop.is_set():
            stop.wait(0.5)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
    return wired


def _capture_loop(
    name: str,
    description: str,
    slot: LatestFrameSlot,
    stop: threading.Event,
    open_pipeline: Callable[[str], object],
) -> None:
    logged_frame = False
    reported_failure = False
    retry_seconds = _OPEN_RETRY_SECONDS
    while not stop.is_set():
        try:
            capture = open_pipeline(description)
        except Exception as exc:
            reported_failure, retry_seconds = _capture_failed(
                name, "open", exc, reported_failure, retry_seconds, stop
            )
            continue
        read_error: Exception | None = None
        try:
            while not stop.is_set():
                pulled = capture.pull()
                if pulled is None:
                    continue
                payload, timestamp, width, height = pulled
                slot.publish(Frame(name, payload, int(timestamp), int(width), int(height)))
                if not logged_frame or reported_failure:
                    logger.info("stream %s received a frame (%d bytes)", name, len(payload))
                logged_frame = True
                reported_failure = False
                retry_seconds = _OPEN_RETRY_SECONDS
        except Exception as exc:
            read_error = exc
        finally:
            capture.close()
        if read_error is not None:
            reported_failure, retry_seconds = _capture_failed(
                name, "read", read_error, reported_failure, retry_seconds, stop
            )


def _capture_failed(
    name: str,
    action: str,
    exc: Exception,
    reported: bool,
    retry_seconds: float,
    stop: threading.Event,
) -> tuple[bool, float]:
    """Log a capture failure once, then wait longer between quiet retries."""
    if not reported:
        logger.warning("capture %s failed for %s: %s", action, name, exc)
    if stop.wait(retry_seconds):
        return True, retry_seconds
    return True, min(retry_seconds * 2, _MAX_RETRY_SECONDS)


def _load_reader(config: AppConfig) -> object | None:
    if not config.recognizer.enabled:
        logger.info("plate reader disabled")
        return None
    if config.service.runtime == "npu":
        detector_path = config.detector.npu_path
        ocr_path = config.recognizer.npu_path
    else:
        detector_path = config.detector.path
        ocr_path = config.recognizer.path
    if not detector_path or not ocr_path:
        logger.error("plate reader needs model paths for runtime %s", config.service.runtime)
        return None
    try:
        from anpr.recognize import PlateReader

        return PlateReader(
            detector_path=Path(detector_path),
            ocr_path=Path(ocr_path),
            alphabet=config.recognizer.charset,
            threshold=config.detector.conf_threshold,
            runtime=config.service.runtime,
            quality=config.recognizer.quality,
            enhancer=config.recognizer.enhancer,
        )
    except Exception:
        logger.exception("plate reader failed to load")
        return None


def _frame_timing(stats: object) -> FrameTiming | None:
    detector_ms = getattr(stats, "det_ms", None)
    reader_ms = getattr(stats, "ocr_ms", None)
    if isinstance(detector_ms, bool) or isinstance(reader_ms, bool):
        return None
    if not isinstance(detector_ms, int | float) or not isinstance(reader_ms, int | float):
        return None
    return FrameTiming(float(detector_ms), float(reader_ms))


def _reader_outcome(name: str, outcome: object) -> tuple[list[object] | None, object]:
    if not isinstance(outcome, tuple) or len(outcome) != 2 or not isinstance(outcome[0], list):
        logger.warning("plate reader for %s returned %s", name, type(outcome).__name__)
        return None, None
    return outcome


def _infer_loop(
    slots: dict[str, LatestFrameSlot],
    stop: threading.Event,
    reader: object,
    fps_limit: int,
    min_confidence: float,
    min_char_confidence: float,
    results: ResultStore | None,
    jpeg_quality: int,
) -> None:
    interval = 1.0 / fps_limit
    next_tick = 0.0
    last_texts: dict[str, set[str]] = {}
    while not stop.is_set():
        now = time.monotonic()
        if now < next_tick:
            if stop.wait(next_tick - now):
                return
            continue
        next_tick = now + interval
        for name, slot in slots.items():
            frame = slot.take()
            if frame is None or frame.width <= 0 or frame.height <= 0:
                continue
            try:
                outcome = reader.read(frame)
            except Exception as exc:
                logger.warning("plate read failed for %s: %s", name, exc)
                continue
            reads, stats = _reader_outcome(name, outcome)
            if reads is None or stats is None:
                continue
            try:
                fresh = publish_reads(
                    results,
                    frame,
                    reads,
                    min_confidence=min_confidence,
                    min_char_confidence=min_char_confidence,
                    jpeg_quality=jpeg_quality,
                    last_texts=last_texts,
                    timing=_frame_timing(stats),
                )
            except Exception:
                logger.exception("plate publish failed for %s", name)
                continue
            for read in fresh:
                box = read.box
                logger.info(
                    "plate %s text=%s confidence=%.2f runtime=%s det_ms=%.1f ocr_ms=%.1f "
                    "cpu=%.0f rss_mb=%.0f box=%d,%d %dx%d det=%.2f",
                    name,
                    read.text,
                    read.confidence,
                    stats.runtime,
                    stats.det_ms,
                    stats.ocr_ms,
                    stats.cpu_percent,
                    stats.rss_mb,
                    box.x,
                    box.y,
                    box.width,
                    box.height,
                    box.confidence,
                )
