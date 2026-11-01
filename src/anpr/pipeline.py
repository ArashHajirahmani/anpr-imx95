"""Latest-frame slots and the scaffold wiring step."""

import threading
from dataclasses import dataclass

from anpr.capture import pipeline_for_stream
from anpr.config import AppConfig
from anpr.types import Frame


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
    """Build per-stream pipelines and slots. This scaffold does not start capture."""
    pipelines: dict[str, str] = {}
    slots: dict[str, LatestFrameSlot] = {}
    for stream in config.enabled_streams():
        pipelines[stream.name] = pipeline_for_stream(stream)
        slots[stream.name] = LatestFrameSlot()
    return WiredPipeline(pipelines=pipelines, slots=slots)
