"""Latest-frame slots and the capture threads that fill them."""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from anpr.capture import pipeline_for_stream
from anpr.config import AppConfig
from anpr.plates import accept_plate
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
    reader = _load_reader(config)
    if reader is not None:
        threads.append(
            threading.Thread(
                target=_infer_loop,
                name="plate-reader",
                args=(
                    wired.slots,
                    stop,
                    reader,
                    config.service.inference_fps_limit,
                    config.recognizer.min_confidence,
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
    if not config.detector.path or not config.recognizer.path:
        logger.error("plate reader needs detector.path and recognizer.path")
        return None
    try:
        from anpr.recognize import PlateReader

        return PlateReader(
            detector_path=Path(config.detector.path),
            ocr_path=Path(config.recognizer.path),
            alphabet=config.recognizer.charset,
            threshold=config.detector.conf_threshold,
        )
    except Exception:
        logger.exception("plate reader failed to load")
        return None


def _infer_loop(
    slots: dict[str, LatestFrameSlot],
    stop: threading.Event,
    reader: object,
    fps_limit: int,
    min_confidence: float,
) -> None:
    interval = 1.0 / fps_limit
    next_tick = 0.0
    last_text: dict[str, str] = {}
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
                reads = reader.read(frame)
            except Exception as exc:
                logger.warning("plate read failed for %s: %s", name, exc)
                continue
            for read in reads:
                if not accept_plate(read.text, read.confidence, min_confidence):
                    continue
                if last_text.get(name) == read.text:
                    continue
                last_text[name] = read.text
                box = read.box
                logger.info(
                    "plate %s text=%s confidence=%.2f box=%d,%d %dx%d det=%.2f",
                    name,
                    read.text,
                    read.confidence,
                    box.x,
                    box.y,
                    box.width,
                    box.height,
                    box.confidence,
                )
