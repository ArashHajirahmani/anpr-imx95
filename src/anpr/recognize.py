"""ONNX Runtime plate detector and OCR.

These weights are not compiled for Neutron. The board runs them on CPU.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from anpr.plates import (
    OCR_HEIGHT,
    OCR_WIDTH,
    PLATE_ALPHABET,
    center_to_corners,
    decode_characters,
    nms,
)
from anpr.types import Frame, PlateBox

logger = logging.getLogger("anpr.recognize")
DETECTOR_SIZE = 640


@dataclass(frozen=True)
class PlateRead:
    text: str
    confidence: float
    box: PlateBox


class PlateReader:
    def __init__(
        self,
        detector_path: Path,
        ocr_path: Path,
        alphabet: str = PLATE_ALPHABET,
        threshold: float = 0.25,
    ) -> None:
        import onnxruntime as ort

        if not detector_path.is_file():
            raise FileNotFoundError(detector_path)
        if not ocr_path.is_file():
            raise FileNotFoundError(ocr_path)
        self._alphabet = alphabet or PLATE_ALPHABET
        self._threshold = threshold
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self._detector = ort.InferenceSession(
            str(detector_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._ocr = ort.InferenceSession(
            str(ocr_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._detector_input = self._detector.get_inputs()[0].name
        self._ocr_input = self._ocr.get_inputs()[0].name
        logger.info(
            "plate reader loaded on onnxruntime CPU detector=%s ocr=%s",
            detector_path,
            ocr_path,
        )

    def read(self, frame: Frame) -> list[PlateRead]:
        import numpy as np

        if frame.width <= 0 or frame.height <= 0:
            return []
        image = np.frombuffer(frame.data, dtype=np.uint8)
        expected = frame.width * frame.height * 3
        if image.size < expected:
            return []
        image = image[:expected].reshape(frame.height, frame.width, 3)
        blob, scale, left, top = _letterbox(image, DETECTOR_SIZE)
        raw = self._detector.run(None, {self._detector_input: blob})[0]
        boxes = _boxes_from_yolo(raw, self._threshold, scale, left, top, frame.width, frame.height)
        reads: list[PlateRead] = []
        for x1, y1, x2, y2, confidence in boxes:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            text, ocr_confidence = self._read_crop(crop)
            reads.append(
                PlateRead(
                    text=text,
                    confidence=ocr_confidence if text else confidence,
                    box=PlateBox(x1, y1, x2 - x1, y2 - y1, confidence),
                )
            )
        return reads

    def _read_crop(self, bgr: object) -> tuple[str, float]:
        import cv2
        import numpy as np

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (OCR_WIDTH, OCR_HEIGHT), interpolation=cv2.INTER_LINEAR)
        batch = np.ascontiguousarray(resized[None, ...])
        logits = self._ocr.run(None, {self._ocr_input: batch})[0][0]
        return decode_characters(logits.tolist(), self._alphabet)


def _letterbox(image: object, size: int) -> tuple[object, float, int, int]:
    import numpy as np

    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    import cv2

    resized = cv2.resize(image, (resized_width, resized_height))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(blob), scale, left, top


def _boxes_from_yolo(
    raw: object,
    threshold: float,
    scale: float,
    left: int,
    top: int,
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int, float]]:
    prediction = raw[0].T
    selected: list[tuple[float, float, float, float, float]] = []
    for cx, cy, width, height, confidence in prediction:
        if float(confidence) < threshold:
            continue
        x1, y1, x2, y2 = center_to_corners(float(cx), float(cy), float(width), float(height))
        x1 = (x1 - left) / scale
        y1 = (y1 - top) / scale
        x2 = (x2 - left) / scale
        y2 = (y2 - top) / scale
        x1 = min(max(x1, 0), image_width - 1)
        y1 = min(max(y1, 0), image_height - 1)
        x2 = min(max(x2, 0), image_width)
        y2 = min(max(y2, 0), image_height)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        selected.append((x1, y1, x2, y2, float(confidence)))
    kept = nms(selected)
    return [(int(x1), int(y1), int(x2), int(y2), confidence) for x1, y1, x2, y2, confidence in kept]
