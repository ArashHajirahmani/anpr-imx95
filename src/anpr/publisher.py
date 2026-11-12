"""Vote, accept, and publish plate reads for one frame.

Both the single-threaded loop and the staged pipeline call `ReadPublisher.handle` with
frames in capture order, so voting state, history, and the UI see the same sequence.
"""

import logging
import threading

from anpr.fusion import FusionSettings
from anpr.plates import PLATE_ALPHABET
from anpr.results import FrameTiming, ResultStore, publish_reads
from anpr.types import Frame
from anpr.vote import PlateVoteTracker

logger = logging.getLogger("anpr.pipeline")


def frame_timing(stats: object) -> FrameTiming | None:
    detector_ms = getattr(stats, "det_ms", None)
    reader_ms = getattr(stats, "ocr_ms", None)
    if isinstance(detector_ms, bool) or isinstance(reader_ms, bool):
        return None
    if not isinstance(detector_ms, int | float) or not isinstance(reader_ms, int | float):
        return None
    runtime = getattr(stats, "runtime", "cpu")
    if not isinstance(runtime, str) or runtime not in {"cpu", "npu"}:
        runtime = "cpu"
    return FrameTiming(float(detector_ms), float(reader_ms), runtime)


class ReadPublisher:
    """Stateful: owns the temporal voter and the per-stream last-text table."""

    def __init__(
        self,
        results: ResultStore | None,
        min_confidence: float,
        min_char_confidence: float,
        jpeg_quality: int,
        vote: FusionSettings | None = None,
        alphabet: str = PLATE_ALPHABET,
    ) -> None:
        self._results = results
        self._min_confidence = min_confidence
        self._min_char_confidence = min_char_confidence
        self._jpeg_quality = jpeg_quality
        self._last_texts: dict[str, set[str]] = {}
        self._vote = vote
        self._alphabet = alphabet
        self._lock = threading.Lock()
        self._voter = PlateVoteTracker(vote, alphabet)
        if results is not None:
            results.on_reset(self.reset)
        # Admit weak OCR into the vote buffer; high-trust / majority decide release.
        self._admit_floor = min(0.05, float(min_confidence), float(min_char_confidence))

    def reset(self) -> None:
        """Drop the temporal vote tracks and the last-text table."""
        with self._lock:
            self._voter = PlateVoteTracker(self._vote, self._alphabet)
            self._last_texts.clear()
        logger.info("vote state reset")

    def handle(self, name: str, frame: Frame, reads: list[object], stats: object) -> None:
        with self._lock:
            self._handle(name, frame, reads, stats)

    def _handle(self, name: str, frame: Frame, reads: list[object], stats: object) -> None:
        min_char_confidence = self._min_char_confidence
        released: list[object] = []
        for read in reads:
            min_char = (
                min(read.char_confidences)
                if getattr(read, "char_confidences", None)
                else read.confidence
            )
            if float(min_char) < self._admit_floor:
                logger.info(
                    "ocr reject %s text=%s min_char=%.2f admit_floor=%.2f",
                    name,
                    read.text,
                    float(min_char),
                    self._admit_floor,
                )
                continue
            logger.info(
                "ocr candidate %s text=%s confidence=%.2f blank=%.2f box=%d,%d %dx%d",
                name,
                read.text,
                float(read.confidence),
                float(getattr(read, "blank_uncertainty", 0.0)),
                read.box.x,
                read.box.y,
                read.box.width,
                read.box.height,
            )
            try:
                voted = self._voter.consider(name, read)
            except Exception:
                logger.exception("plate vote failed for %s", name)
                voted = read if float(min_char) >= float(min_char_confidence) else None
            if voted is None:
                continue
            voted_min = (
                min(voted.char_confidences)
                if getattr(voted, "char_confidences", None)
                else voted.confidence
            )
            if float(voted_min) < float(min_char_confidence):
                continue
            released.append(voted)
        try:
            fresh = publish_reads(
                self._results,
                frame,
                released,
                min_confidence=self._min_confidence,
                min_char_confidence=min_char_confidence,
                jpeg_quality=self._jpeg_quality,
                last_texts=self._last_texts,
                timing=frame_timing(stats),
            )
        except Exception:
            logger.exception("plate publish failed for %s", name)
            return
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
