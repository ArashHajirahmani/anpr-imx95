"""ONNX Runtime plate detector and OCR.

`cpu` uses the float models. `npu` uses the INT8 models and the Neutron provider.
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from anpr.plates import (
    PLATE_ALPHABET,
    center_to_corners,
    decode_characters,
    nms,
    resize_rgb_for_ocr,
)
from anpr.quality import (
    EnhancerSettings,
    QualitySettings,
    crop_quality,
    enhance_crop,
    log_quality_reject,
)
from anpr.types import Frame, PlateBox

logger = logging.getLogger("anpr.recognize")
DETECTOR_SIZE = 640


@dataclass(frozen=True)
class PlateRead:
    text: str
    confidence: float
    box: PlateBox
    char_confidences: tuple[float, ...] = ()
    # BGR uint8 bytes of the crop after enhance_crop (identity when enhancer is off).
    # History JPEGs use this so the UI shows the exact image that entered OCR resize.
    pre_ocr_bgr: bytes | None = None
    pre_ocr_width: int = 0
    pre_ocr_height: int = 0


@dataclass(frozen=True)
class InferenceStats:
    runtime: str
    det_ms: float
    ocr_ms: float
    cpu_percent: float
    rss_mb: float


class PlateReader:
    def __init__(
        self,
        detector_path: Path,
        ocr_path: Path,
        alphabet: str = PLATE_ALPHABET,
        threshold: float = 0.25,
        runtime: str = "cpu",
        quality: QualitySettings | None = None,
        enhancer: EnhancerSettings | None = None,
    ) -> None:
        import onnxruntime as ort

        if runtime not in {"cpu", "npu"}:
            raise ValueError(f"runtime must be cpu or npu, got {runtime}")
        if not detector_path.is_file():
            raise FileNotFoundError(detector_path)
        if not ocr_path.is_file():
            raise FileNotFoundError(ocr_path)
        self._alphabet = alphabet or PLATE_ALPHABET
        self._threshold = threshold
        self._runtime = runtime
        self._quality = quality or QualitySettings()
        self._enhancer = enhancer or EnhancerSettings()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        providers: list[str | tuple[str, dict[str, bool]]]
        if runtime == "npu":
            providers = [
                ("NeutronExecutionProvider", {"neutron_op_only": False}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        self._detector = ort.InferenceSession(
            str(detector_path),
            sess_options=options,
            providers=providers,
        )
        self._ocr = ort.InferenceSession(
            str(ocr_path),
            sess_options=options,
            providers=providers,
        )
        self._detector_input = self._detector.get_inputs()[0].name
        self._ocr_input = self._ocr.get_inputs()[0].name
        logger.info(
            "plate reader runtime=%s providers=%s detector=%s ocr=%s",
            runtime,
            self._detector.get_providers(),
            detector_path,
            ocr_path,
        )

    def read(self, frame: Frame) -> tuple[list[PlateRead], InferenceStats]:
        import numpy as np

        empty = InferenceStats(self._runtime, 0.0, 0.0, 0.0, _rss_mb())
        if frame.width <= 0 or frame.height <= 0:
            return [], empty
        image = np.frombuffer(frame.data, dtype=np.uint8)
        expected = frame.width * frame.height * 3
        if image.size < expected:
            return [], empty
        image = image[:expected].reshape(frame.height, frame.width, 3)
        cpu_before = _cpu_seconds()
        started = time.perf_counter()
        blob, scale, left, top = _letterbox(image, DETECTOR_SIZE)
        raw = self._detector.run(None, {self._detector_input: blob})[0]
        detected = time.perf_counter()
        boxes = _boxes_from_yolo(raw, self._threshold, scale, left, top, frame.width, frame.height)
        reads: list[PlateRead] = []
        for x1, y1, x2, y2, confidence in boxes:
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            ok, metrics, reason = crop_quality(crop, self._quality)
            if not ok:
                log_quality_reject(frame.stream_name, reason, metrics)
                continue
            prepared = enhance_crop(crop, self._enhancer)
            text, ocr_confidence, char_confidences = self._ocr_crop(prepared)
            if not text:
                continue
            pre = np.ascontiguousarray(prepared)
            reads.append(
                PlateRead(
                    text=text,
                    confidence=ocr_confidence,
                    box=PlateBox(x1, y1, x2 - x1, y2 - y1, confidence),
                    char_confidences=tuple(char_confidences),
                    pre_ocr_bgr=pre.tobytes(),
                    pre_ocr_width=int(pre.shape[1]),
                    pre_ocr_height=int(pre.shape[0]),
                )
            )
        finished = time.perf_counter()
        wall = max(finished - started, 1e-6)
        stats = InferenceStats(
            runtime=self._runtime,
            det_ms=(detected - started) * 1000,
            ocr_ms=(finished - detected) * 1000,
            cpu_percent=(_cpu_seconds() - cpu_before) / wall * 100,
            rss_mb=_rss_mb(),
        )
        return reads, stats

    def _ocr_crop(self, prepared_bgr: object) -> tuple[str, float, list[float]]:
        import cv2
        import numpy as np

        rgb = cv2.cvtColor(prepared_bgr, cv2.COLOR_BGR2RGB)
        resized = resize_rgb_for_ocr(rgb)
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
    prediction = raw[0]
    if prediction.shape[0] < prediction.shape[-1]:
        prediction = prediction.T
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


def _cpu_seconds() -> float:
    try:
        parts = Path("/proc/self/stat").read_text(encoding="utf-8").split()
        ticks = os.sysconf("SC_CLK_TCK")
        return (int(parts[13]) + int(parts[14])) / ticks
    except (OSError, IndexError, ValueError):
        return 0.0


def _rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (OSError, IndexError, ValueError):
        return 0.0
    return 0.0
