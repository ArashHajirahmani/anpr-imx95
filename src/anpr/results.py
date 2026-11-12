"""Latest plate snapshot and sticky recognition history.

The inference thread records one JPEG plus the plates on that frame. History
keeps a few distinct plate slots. Partial OCR forms of the same plate update
the matching slot (best text + best crop) instead of opening a new entry.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime

from anpr.plate_match import MATCH_ALIASES, merge_text, same_identity
from anpr.plates import accept_plate
from anpr.recognize import PlateRead
from anpr.types import Frame, PlateBox

logger = logging.getLogger("anpr.results")

_FPS_WINDOW_SECONDS = 3.0


@dataclass(frozen=True)
class FrameTiming:
    """Detector and plate-read durations for one frame, in milliseconds."""

    detector_ms: float
    reader_ms: float
    runtime: str = "cpu"


@dataclass(frozen=True)
class LatestView:
    stream: str
    frame_id: int
    width: int
    height: int
    monotonic_ns: int
    time: str
    jpeg: bytes
    plates: tuple[PlateRead, ...]
    timing: FrameTiming | None = None
    # Frames published per second over the last few seconds (backend-measured).
    fps: float | None = None

    def json(self) -> dict[str, object]:
        ordered = sorted(self.plates, key=lambda read: read.confidence, reverse=True)
        payload: dict[str, object] = {
            "stream": self.stream,
            "frame_id": self.frame_id,
            "width": self.width,
            "height": self.height,
            "monotonic_ns": self.monotonic_ns,
            "time": self.time,
            "plates": [_plate_json(read) for read in ordered],
        }
        payload.update(_timing_json(self.timing))
        if self.fps is not None:
            payload["fps"] = round(self.fps, 1)
        return payload


@dataclass(frozen=True)
class HistoryView:
    id: int
    stream: str
    frame_id: int
    width: int
    height: int
    time: str
    text: str
    confidence: float
    box: PlateBox
    jpeg: bytes
    timing: FrameTiming | None = None
    characters: tuple[tuple[str, float], ...] = ()

    def json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "stream": self.stream,
            "frame_id": self.frame_id,
            "width": self.width,
            "height": self.height,
            "time": self.time,
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "characters": [
                {"char": character, "confidence": round(score, 4)}
                for character, score in self.characters
            ],
            "box": _box_json(self.box),
        }
        payload.update(_timing_json(self.timing))
        return payload


@dataclass
class _PlateSlot:
    view: HistoryView
    aliases: deque[str] = field(default_factory=lambda: deque(maxlen=MATCH_ALIASES))
    best_by_text: dict[str, float] = field(default_factory=dict)
    quality_by_text: dict[str, float] = field(default_factory=dict)
    best_quality: float = 0.0


class ResultStore:
    """Thread-safe latest frame and sticky distinct-plate history."""

    def __init__(self, history_limit: int) -> None:
        if history_limit <= 0:
            raise ValueError("history_limit must be positive")
        self._limit = history_limit
        self._lock = threading.Lock()
        self._frame_id = 0
        self._event_id = 0
        self._latest: LatestView | None = None
        self._slots: list[_PlateSlot] = []
        self._frame_times: deque[float] = deque()

    @property
    def history_limit(self) -> int:
        return self._limit

    def latest(self) -> LatestView | None:
        with self._lock:
            return self._latest

    def _measure_fps(self, now: float) -> float | None:
        """Published frames per second over the last `_FPS_WINDOW_SECONDS` (lock held)."""
        times = self._frame_times
        times.append(now)
        while times and now - times[0] > _FPS_WINDOW_SECONDS:
            times.popleft()
        if len(times) < 2 or now <= times[0]:
            return None
        return (len(times) - 1) / (now - times[0])

    def history(self) -> tuple[HistoryView, ...]:
        with self._lock:
            return tuple(slot.view for slot in self._slots)

    def record(
        self,
        frame: Frame,
        reads: list[PlateRead],
        *,
        min_confidence: float,
        jpeg: bytes,
        last_texts: dict[str, set[str]],
        timing: FrameTiming | None = None,
        jpeg_quality: int = 80,
        min_char_confidence: float | None = None,
    ) -> list[PlateRead]:
        """Store a frame. Return plate reads whose text was not in the previous set."""
        accepted, fresh = classify_reads(
            frame.stream_name,
            reads,
            min_confidence,
            last_texts,
            min_char_confidence=min_char_confidence,
        )
        if not jpeg:
            logger.warning("empty jpeg for %s", frame.stream_name)
            return fresh
        try:
            self._save(frame, jpeg, accepted, fresh, timing, jpeg_quality)
        except Exception:
            logger.exception("result store failed for %s", frame.stream_name)
        return fresh

    def observe(
        self,
        frame: Frame,
        reads: list[PlateRead],
        *,
        min_confidence: float,
        jpeg_quality: int,
        last_texts: dict[str, set[str]],
        timing: FrameTiming | None = None,
        min_char_confidence: float | None = None,
    ) -> list[PlateRead]:
        """Encode the frame, store it, and return newly accepted plates."""
        try:
            jpeg = encode_jpeg(frame, jpeg_quality)
        except Exception:
            logger.exception("jpeg encode failed for %s", frame.stream_name)
            jpeg = None
        if not jpeg:
            return classify_reads(
                frame.stream_name,
                reads,
                min_confidence,
                last_texts,
                min_char_confidence=min_char_confidence,
            )[1]
        return self.record(
            frame,
            reads,
            min_confidence=min_confidence,
            min_char_confidence=min_char_confidence,
            jpeg=jpeg,
            last_texts=last_texts,
            timing=timing,
            jpeg_quality=jpeg_quality,
        )

    def _save(
        self,
        frame: Frame,
        jpeg: bytes,
        accepted: list[PlateRead],
        fresh: list[PlateRead],
        timing: FrameTiming | None,
        jpeg_quality: int,
    ) -> None:
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")
        prepared: list[tuple[PlateRead, bytes, int, int]] = []
        for read in fresh:
            plate = pre_ocr_jpeg(read, jpeg_quality, frame.stream_name)
            if plate is None:
                plate = crop_plate_jpeg(frame, read.box, jpeg_quality)
            if plate is None:
                prepared.append((read, jpeg, frame.width, frame.height))
            else:
                prepared.append((read, plate[0], plate[1], plate[2]))
        with self._lock:
            self._frame_id += 1
            frame_id = self._frame_id
            fps = self._measure_fps(time.monotonic())
            self._latest = LatestView(
                stream=frame.stream_name,
                frame_id=frame_id,
                width=frame.width,
                height=frame.height,
                monotonic_ns=frame.monotonic_ns,
                time=stamp,
                jpeg=jpeg,
                plates=tuple(accepted),
                timing=timing,
                fps=fps,
            )
            for read, plate_jpeg, plate_width, plate_height in prepared:
                self._upsert_slot(
                    read,
                    plate_jpeg,
                    plate_width,
                    plate_height,
                    frame_id=frame_id,
                    stream=frame.stream_name,
                    stamp=stamp,
                    timing=timing,
                )
        logger.debug(
            "published frame %s id=%s plates=%d new=%d slots=%d",
            frame.stream_name,
            frame_id,
            len(accepted),
            len(fresh),
            len(self._slots),
        )

    def _upsert_slot(
        self,
        read: PlateRead,
        plate_jpeg: bytes,
        plate_width: int,
        plate_height: int,
        *,
        frame_id: int,
        stream: str,
        stamp: str,
        timing: FrameTiming | None,
    ) -> None:
        characters = tuple(zip(read.text, read.char_confidences, strict=False))
        quality = float(getattr(read, "laplacian_var", 0.0) or 0.0)
        index = self._match_slot(read.text)
        if index is None:
            self._event_id += 1
            view = HistoryView(
                id=self._event_id,
                stream=stream,
                frame_id=frame_id,
                width=plate_width,
                height=plate_height,
                time=stamp,
                text=read.text,
                confidence=read.confidence,
                box=read.box,
                jpeg=plate_jpeg,
                timing=timing,
                characters=characters,
            )
            slot = _PlateSlot(view=view, best_quality=quality)
            slot.aliases.append(read.text)
            slot.best_by_text[read.text] = float(read.confidence)
            slot.quality_by_text[read.text] = quality
            self._slots.insert(0, slot)
            del self._slots[self._limit :]
            return

        slot = self._slots.pop(index)
        current = slot.view
        slot.best_by_text[read.text] = max(
            float(slot.best_by_text.get(read.text, 0.0)),
            float(read.confidence),
        )
        slot.quality_by_text[read.text] = max(
            float(slot.quality_by_text.get(read.text, 0.0)),
            quality,
        )
        text, confidence = merge_text(
            current.text,
            current.confidence,
            read.text,
            read.confidence,
            aliases=tuple(slot.aliases),
            best_by_text=slot.best_by_text,
            quality_by_text=slot.quality_by_text,
            current_quality=slot.best_quality,
            new_quality=quality,
        )
        text_changed = text != current.text
        use_new_glyph = text == read.text
        better_frame = use_new_glyph and (
            float(read.confidence) > float(current.confidence) + 1e-6
            or (
                abs(float(read.confidence) - float(current.confidence)) <= 0.02
                and quality > slot.best_quality + 1e-6
            )
            or text_changed
        )
        if better_frame:
            slot.best_quality = max(slot.best_quality, quality)
        slot.view = replace(
            current,
            stream=stream,
            frame_id=frame_id,
            time=stamp,
            text=text,
            confidence=confidence,
            box=read.box if use_new_glyph else current.box,
            timing=timing if better_frame or text_changed else current.timing,
            characters=characters if use_new_glyph else current.characters,
            jpeg=plate_jpeg if better_frame else current.jpeg,
            width=plate_width if better_frame else current.width,
            height=plate_height if better_frame else current.height,
        )
        slot.aliases.append(read.text)
        self._slots.insert(0, slot)

    def _match_slot(self, text: str) -> int | None:
        for index, slot in enumerate(self._slots):
            if same_identity(slot.view.text, text):
                return index
            if any(same_identity(alias, text) for alias in slot.aliases):
                return index
        return None


def publish_reads(
    store: ResultStore | None,
    frame: Frame,
    reads: list[PlateRead],
    *,
    min_confidence: float,
    jpeg_quality: int,
    last_texts: dict[str, set[str]],
    timing: FrameTiming | None = None,
    min_char_confidence: float | None = None,
) -> list[PlateRead]:
    """Store the frame when a result store is attached, and return new plates."""
    if store is None:
        return classify_reads(
            frame.stream_name,
            reads,
            min_confidence,
            last_texts,
            min_char_confidence=min_char_confidence,
        )[1]
    return store.observe(
        frame,
        reads,
        min_confidence=min_confidence,
        min_char_confidence=min_char_confidence,
        jpeg_quality=jpeg_quality,
        last_texts=last_texts,
        timing=timing,
    )


def classify_reads(
    stream: str,
    reads: list[PlateRead],
    min_confidence: float,
    last_texts: dict[str, set[str]],
    *,
    min_char_confidence: float | None = None,
) -> tuple[list[PlateRead], list[PlateRead]]:
    """Accepted plates for this frame.

    ``last_texts`` is retained for callers; sticky history dedupe lives in
    ``ResultStore``.
    """
    del last_texts
    accepted: list[PlateRead] = []
    for read in reads:
        if accept_plate(
            read.text,
            read.confidence,
            min_confidence,
            min_char_confidence=min_char_confidence,
            char_confidences=read.char_confidences,
        ):
            accepted.append(read)
            continue
        min_char = min(read.char_confidences) if read.char_confidences else read.confidence
        logger.info(
            "ocr reject %s text=%s min_char=%.2f floor=%.2f char_floor=%s",
            stream,
            read.text,
            min_char,
            min_confidence,
            f"{min_char_confidence:.2f}" if min_char_confidence is not None else "none",
        )
    return accepted, list(accepted)


def encode_jpeg(frame: Frame, quality: int) -> bytes | None:
    """Encode a BGR frame. Returns None when the payload cannot be encoded."""
    image = _bgr_image(frame, warn=True)
    if image is None:
        return None
    return _jpeg_bytes(image, quality, frame.stream_name)


def pre_ocr_jpeg(
    read: PlateRead,
    quality: int,
    stream: str,
) -> tuple[bytes, int, int] | None:
    """JPEG of the enhancer output that OCR resized. Falls back when absent."""
    import numpy as np

    if not read.pre_ocr_bgr or read.pre_ocr_width < 2 or read.pre_ocr_height < 2:
        return None
    expected = read.pre_ocr_width * read.pre_ocr_height * 3
    if len(read.pre_ocr_bgr) < expected:
        return None
    image = np.frombuffer(read.pre_ocr_bgr, dtype=np.uint8, count=expected)
    image = image.reshape(read.pre_ocr_height, read.pre_ocr_width, 3)
    payload = _jpeg_bytes(image, quality, stream)
    if payload is None:
        return None
    return payload, read.pre_ocr_width, read.pre_ocr_height


def crop_plate_jpeg(
    frame: Frame,
    box: PlateBox,
    quality: int,
) -> tuple[bytes, int, int] | None:
    """JPEG of the plate box only. Returns the bytes and the crop size."""
    image = _bgr_image(frame, warn=False)
    if image is None:
        return None
    x1 = min(max(box.x, 0), frame.width - 1)
    y1 = min(max(box.y, 0), frame.height - 1)
    x2 = min(max(box.x + box.width, x1 + 1), frame.width)
    y2 = min(max(box.y + box.height, y1 + 1), frame.height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = image[y1:y2, x1:x2]
    payload = _jpeg_bytes(crop, quality, frame.stream_name)
    if payload is None:
        return None
    return payload, x2 - x1, y2 - y1


def _bgr_image(frame: Frame, *, warn: bool):
    import numpy as np

    if frame.width <= 0 or frame.height <= 0:
        if warn:
            logger.warning("frame %s has no size", frame.stream_name)
        return None
    data = frame.data
    if not isinstance(data, bytes | bytearray | memoryview):
        if warn:
            logger.warning("frame %s payload is %s", frame.stream_name, type(data).__name__)
        return None
    expected = frame.width * frame.height * 3
    buffer = memoryview(data)
    if len(buffer) < expected:
        if warn:
            logger.warning(
                "frame %s payload is %d bytes, expected %d",
                frame.stream_name,
                len(buffer),
                expected,
            )
        return None
    flat = np.frombuffer(buffer, dtype=np.uint8, count=expected)
    return flat.reshape(frame.height, frame.width, 3).copy()


def _jpeg_bytes(image: object, quality: int, stream: str) -> bytes | None:
    import cv2

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        logger.warning("jpeg encode failed for %s", stream)
        return None
    payload = encoded.tobytes()
    if not payload:
        logger.warning("jpeg encode was empty for %s", stream)
        return None
    return payload


def _timing_json(timing: FrameTiming | None) -> dict[str, object]:
    if timing is None:
        return {}
    runtime = timing.runtime if timing.runtime in {"cpu", "npu"} else "cpu"
    return {
        "detector_ms": round(timing.detector_ms, 1),
        "reader_ms": round(timing.reader_ms, 1),
        "runtime": runtime,
    }


def _plate_json(read: PlateRead) -> dict[str, object]:
    return {
        "text": read.text,
        "confidence": round(read.confidence, 4),
        "characters": [
            {"char": character, "confidence": round(score, 4)}
            for character, score in zip(read.text, read.char_confidences, strict=False)
        ],
        "box": _box_json(read.box),
    }


def _box_json(box: PlateBox) -> dict[str, object]:
    return {
        "x": box.x,
        "y": box.y,
        "width": box.width,
        "height": box.height,
        "confidence": round(box.confidence, 4),
    }
