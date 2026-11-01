"""Latest-frame slots and the capture threads that fill them."""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass

from anpr.capture import pipeline_for_stream
from anpr.config import AppConfig
from anpr.types import Frame

logger = logging.getLogger("anpr.pipeline")
_OPEN_RETRY_SECONDS = 0.5


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
    while not stop.is_set():
        try:
            capture = open_pipeline(description)
        except Exception:
            logger.exception("capture open failed for %s", name)
            if stop.wait(_OPEN_RETRY_SECONDS):
                return
            continue
        try:
            while not stop.is_set():
                pulled = capture.pull()
                if pulled is None:
                    continue
                payload, timestamp = pulled
                slot.publish(Frame(name, payload, int(timestamp)))
                if not logged_frame:
                    logger.info("stream %s received a frame (%d bytes)", name, len(payload))
                    logged_frame = True
        except Exception:
            logger.exception("capture read failed for %s", name)
        finally:
            capture.close()
        if stop.wait(_OPEN_RETRY_SECONDS):
            return
