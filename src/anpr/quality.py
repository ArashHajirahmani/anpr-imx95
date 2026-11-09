"""Plate-crop image quality checks and optional enhancement."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("anpr.quality")


@dataclass(frozen=True)
class QualitySettings:
    enabled: bool = True
    min_laplacian_var: float = 40.0
    min_brightness: float = 40.0
    max_brightness: float = 220.0
    min_contrast: float = 25.0


@dataclass(frozen=True)
class EnhancerSettings:
    enabled: bool = False
    clahe: bool = True
    gamma: float = 1.1
    sharpen: bool = True


@dataclass(frozen=True)
class QualityMetrics:
    laplacian_var: float
    brightness: float
    contrast: float


def crop_quality(bgr: object, settings: QualitySettings) -> tuple[bool, QualityMetrics, str]:
    """Return whether the BGR crop is sharp and well lit enough for OCR."""
    import cv2
    import numpy as np

    image = np.asarray(bgr)
    if image.size == 0 or image.ndim != 3:
        empty = QualityMetrics(0.0, 0.0, 0.0)
        return False, empty, "empty"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    metrics = QualityMetrics(
        laplacian_var=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        brightness=float(gray.mean()),
        contrast=float(gray.std()),
    )
    if not settings.enabled:
        return True, metrics, ""
    if metrics.laplacian_var < settings.min_laplacian_var:
        return False, metrics, "blurry"
    if metrics.brightness < settings.min_brightness:
        return False, metrics, "dark"
    if metrics.brightness > settings.max_brightness:
        return False, metrics, "bright"
    if metrics.contrast < settings.min_contrast:
        return False, metrics, "low_contrast"
    return True, metrics, ""


def enhance_crop(bgr: object, settings: EnhancerSettings) -> object:
    """Optional CLAHE / gamma / mild sharpening. Returns a BGR uint8 crop."""
    import cv2
    import numpy as np

    image = np.asarray(bgr)
    if not settings.enabled or image.size == 0:
        return image
    out = image
    if settings.clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        light, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        light = clahe.apply(light)
        out = cv2.cvtColor(cv2.merge((light, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    if settings.gamma != 1.0:
        table = np.array(
            [((index / 255.0) ** (1.0 / settings.gamma)) * 255 for index in range(256)],
            dtype=np.uint8,
        )
        out = cv2.LUT(out, table)
    if settings.sharpen:
        blurred = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out, 1.5, blurred, -0.5, 0)
    return np.ascontiguousarray(out.astype(np.uint8))


def log_quality_reject(stream: str, reason: str, metrics: QualityMetrics) -> None:
    logger.info(
        "crop quality reject %s reason=%s lap=%.1f bright=%.1f contrast=%.1f",
        stream,
        reason,
        metrics.laplacian_var,
        metrics.brightness,
        metrics.contrast,
    )
