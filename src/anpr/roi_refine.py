"""Refine a YOLO plate box before enhance/OCR.

Left/right border snap searches ±20% around each YOLO side, locks onto the
outermost strong plate-frame peak, then pads L/R by pad_ratio of width and
T/B by pad_ratio of height (no top/bottom edge finding). Optional perspective
warp uprights a scored quadrilateral in that region.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("anpr.roi_refine")


@dataclass(frozen=True)
class ExpandSettings:
    enabled: bool = True
    # Start scanning this fraction inside the YOLO box (left/right only).
    inner_ratio: float = 0.20
    # Max outward growth past each YOLO side, as a fraction of ROI width.
    outer_ratio: float = 0.20
    # Mean Canny (0..1) on a column that counts as a plate border.
    edge_threshold: float = 0.08
    # After L/R snap: pad each side by this fraction (width for L/R, height for T/B).
    pad_ratio: float = 0.05
    # Kept for older configs; treated as outer_ratio when outer_ratio unset in JSON.
    max_ratio: float = 0.20


@dataclass(frozen=True)
class PerspectiveSettings:
    enabled: bool = False
    # Max search pad and max warped size vs the original ROI (width/height).
    max_ratio: float = 0.35
    min_score: float = 0.45
    min_aspect: float = 1.8
    max_aspect: float = 6.0
    canny_low: int = 40
    canny_high: int = 120


@dataclass(frozen=True)
class RoiRefineSettings:
    expand: ExpandSettings
    perspective: PerspectiveSettings

    @staticmethod
    def disabled() -> RoiRefineSettings:
        return RoiRefineSettings(
            expand=ExpandSettings(enabled=False),
            perspective=PerspectiveSettings(enabled=False),
        )


@dataclass(frozen=True)
class RefinedRoi:
    """Axis-aligned box on the frame plus the BGR crop fed to enhance/OCR."""

    x1: int
    y1: int
    x2: int
    y2: int
    crop_bgr: object
    method: str


def refine_plate_roi(
    image: object,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    settings: RoiRefineSettings,
) -> RefinedRoi:
    """Snap borders and/or perspective-warp a detector box. Falls back to YOLO."""
    import numpy as np

    frame = np.asarray(image)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, width, height)
    orig = (x1, y1, x2, y2)
    method = "yolo"

    if settings.expand.enabled:
        snapped = _snap_borders(frame, x1, y1, x2, y2, settings.expand)
        x1, y1, x2, y2 = _clip_box(*snapped, width, height)
        if (x1, y1, x2, y2) != orig:
            method = "expand"

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        fallback = frame[orig[1] : orig[3], orig[0] : orig[2]]
        return RefinedRoi(orig[0], orig[1], orig[2], orig[3], fallback, "yolo")

    if settings.perspective.enabled:
        warped = _perspective_crop(frame, orig, (x1, y1, x2, y2), settings.perspective)
        if warped is not None:
            label = "perspective" if method == "yolo" else "expand+perspective"
            return RefinedRoi(x1, y1, x2, y2, warped, label)

    return RefinedRoi(x1, y1, x2, y2, np.ascontiguousarray(crop), method)


def _snap_borders(
    frame: object,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    settings: ExpandSettings,
) -> tuple[int, int, int, int]:
    """Snap left/right to plate-frame peaks; pad L/R by width and T/B by height."""
    import cv2
    import numpy as np

    image = np.asarray(frame)
    height, width = image.shape[:2]
    roi_w = max(1, x2 - x1)
    inner = settings.inner_ratio
    outer = settings.outer_ratio if settings.outer_ratio > 0 else settings.max_ratio
    search_x = int(round(roi_w * outer))
    inset_x = int(round(roi_w * inner))

    left_inner = x1 + inset_x
    right_inner = x2 - inset_x
    left_outer = x1 - search_x
    right_outer = x2 + search_x

    # Keep YOLO vertical span for edge scoring; do not snap top/bottom.
    sx1 = max(0, left_outer)
    sy1 = max(0, y1)
    sx2 = min(width, right_outer)
    sy2 = min(height, y2)
    new_x1, new_x2 = x1, x2
    new_y1, new_y2 = y1, y2

    if sx2 - sx1 >= 4 and sy2 - sy1 >= 4:
        gray = cv2.cvtColor(image[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)
        col_score = edges.mean(axis=0) / 255.0
        new_x1 = _outer_strong_peak(
            scores=col_score,
            origin=sx1,
            lo=min(left_outer, left_inner),
            hi=max(left_outer, left_inner),
            threshold=settings.edge_threshold,
            fallback=x1,
            prefer="min",
        )
        # Never shrink the left edge: inward snaps clip the leading glyph (e.g. C).
        if new_x1 > x1:
            new_x1 = x1
        new_x2 = _outer_strong_peak(
            scores=col_score,
            origin=sx1,
            lo=min(right_inner, right_outer),
            hi=max(right_inner, right_outer),
            threshold=settings.edge_threshold,
            fallback=x2,
            prefer="max",
        )
        min_w = max(8, int(round(roi_w * 0.5)))
        if new_x2 - new_x1 < min_w:
            new_x1, new_x2 = x1, x2

    pad = max(0.0, float(settings.pad_ratio))
    if pad > 0.0:
        snap_w = max(1, new_x2 - new_x1)
        snap_h = max(1, new_y2 - new_y1)
        pad_x = int(round(snap_w * pad))
        pad_y = int(round(snap_h * pad))
        new_x1 -= pad_x
        new_x2 += pad_x
        new_y1 -= pad_y
        new_y2 += pad_y
    return new_x1, new_y1, new_x2, new_y2


def _outer_strong_peak(
    *,
    scores: object,
    origin: int,
    lo: int,
    hi: int,
    threshold: float,
    fallback: int,
    prefer: str,
) -> int:
    """Among strong local-max peaks in [lo, hi], pick the outermost.

    Glyph strokes can be the single strongest peak; requiring ~half of the best
    peak and then taking the outer one prefers the plate frame over a letter.
    """
    import numpy as np

    profile = np.asarray(scores, dtype=np.float64)
    n = len(profile)
    if n == 0 or hi < lo:
        return fallback

    peaks: list[tuple[int, float]] = []
    for x in range(lo, hi + 1):
        index = x - origin
        if index < 0 or index >= n:
            continue
        value = float(profile[index])
        if value < threshold:
            continue
        left = float(profile[index - 1]) if index > 0 else value
        right = float(profile[index + 1]) if index + 1 < n else value
        if value >= left and value >= right:
            peaks.append((x, value))
    if not peaks:
        return fallback
    best = max(value for _, value in peaks)
    strong = [x for x, value in peaks if value >= 0.5 * best]
    if not strong:
        return fallback
    return min(strong) if prefer == "min" else max(strong)


def _perspective_crop(
    frame: object,
    original: tuple[int, int, int, int],
    expanded: tuple[int, int, int, int],
    settings: PerspectiveSettings,
) -> object | None:
    import numpy as np

    image = np.asarray(frame)
    height, width = image.shape[:2]
    ox1, oy1, ox2, oy2 = original
    orig_w = max(1, ox2 - ox1)
    orig_h = max(1, oy2 - oy1)
    pad_x = int(round(orig_w * settings.max_ratio))
    pad_y = int(round(orig_h * settings.max_ratio))
    sx1 = max(0, ox1 - pad_x)
    sy1 = max(0, oy1 - pad_y)
    sx2 = min(width, ox2 + pad_x)
    sy2 = min(height, oy2 + pad_y)
    ex1, ey1, ex2, ey2 = expanded
    sx1 = min(sx1, ex1)
    sy1 = min(sy1, ey1)
    sx2 = max(sx2, ex2)
    sy2 = max(sy2, ey2)
    sx1, sy1, sx2, sy2 = _clip_box(sx1, sy1, sx2, sy2, width, height)
    search = image[sy1:sy2, sx1:sx2]
    if search.size == 0:
        return None
    quad = _best_quad(search, settings)
    if quad is None:
        return None
    ranked = quad.astype(np.float32) + np.array([sx1, sy1], dtype=np.float32)
    score = _score_quad(ranked, original, settings)
    if score < settings.min_score:
        return None
    return _warp_quad(image, ranked, original, settings)


def _best_quad(search_bgr: object, settings: PerspectiveSettings) -> object | None:
    import cv2
    import numpy as np

    image = np.asarray(search_bgr)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, settings.canny_low, settings.canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    search_area = float(image.shape[0] * image.shape[1])
    best = None
    best_local = -1.0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 0.05 * search_area or area > 0.98 * search_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        points = approx.reshape(4, 2).astype(np.float32)
        ordered = _order_corners(points)
        width_a = float(np.linalg.norm(ordered[1] - ordered[0]))
        width_b = float(np.linalg.norm(ordered[2] - ordered[3]))
        height_a = float(np.linalg.norm(ordered[3] - ordered[0]))
        height_b = float(np.linalg.norm(ordered[2] - ordered[1]))
        plate_w = max(width_a, width_b, 1.0)
        plate_h = max(height_a, height_b, 1.0)
        aspect = plate_w / plate_h
        if not settings.min_aspect <= aspect <= settings.max_aspect:
            continue
        local = (area / search_area) * min(aspect / 3.0, 1.0)
        if local > best_local:
            best_local = local
            best = ordered
    return best


def _score_quad(
    quad: object,
    original: tuple[int, int, int, int],
    settings: PerspectiveSettings,
) -> float:
    import numpy as np

    points = np.asarray(quad, dtype=np.float32)
    ox1, oy1, ox2, oy2 = original
    ocx = 0.5 * (ox1 + ox2)
    ocy = 0.5 * (oy1 + oy2)
    ow = max(1.0, float(ox2 - ox1))
    oh = max(1.0, float(oy2 - oy1))
    cx = float(points[:, 0].mean())
    cy = float(points[:, 1].mean())
    dist = ((cx - ocx) ** 2 + (cy - ocy) ** 2) ** 0.5
    center_score = max(0.0, 1.0 - dist / max(ow, oh))
    width_a = float(np.linalg.norm(points[1] - points[0]))
    height_a = float(np.linalg.norm(points[3] - points[0]))
    aspect = width_a / max(height_a, 1.0)
    if aspect < settings.min_aspect or aspect > settings.max_aspect:
        return 0.0
    aspect_score = 1.0 - min(abs(aspect - 3.0) / 3.0, 1.0)
    area = width_a * height_a
    orig_area = ow * oh
    area_ratio = area / orig_area
    if area_ratio < 0.35 or area_ratio > (1.0 + settings.max_ratio) ** 2:
        return 0.0
    area_score = 1.0 - min(abs(area_ratio - 1.0), 1.0)
    return 0.45 * center_score + 0.3 * aspect_score + 0.25 * area_score


def _warp_quad(
    frame: object,
    quad: object,
    original: tuple[int, int, int, int],
    settings: PerspectiveSettings,
) -> object | None:
    import cv2
    import numpy as np

    points = np.asarray(quad, dtype=np.float32)
    ox1, oy1, ox2, oy2 = original
    orig_w = max(1, ox2 - ox1)
    orig_h = max(1, oy2 - oy1)
    width_a = float(np.linalg.norm(points[1] - points[0]))
    width_b = float(np.linalg.norm(points[2] - points[3]))
    height_a = float(np.linalg.norm(points[3] - points[0]))
    height_b = float(np.linalg.norm(points[2] - points[1]))
    out_w = int(round(max(width_a, width_b)))
    out_h = int(round(max(height_a, height_b)))
    max_w = int(round(orig_w * (1.0 + settings.max_ratio)))
    max_h = int(round(orig_h * (1.0 + settings.max_ratio)))
    if out_w < 8 or out_h < 8:
        return None
    if out_w > max_w or out_h > max_h:
        scale = min(max_w / out_w, max_h / out_h)
        out_w = max(8, int(round(out_w * scale)))
        out_h = max(8, int(round(out_h * scale)))
    destination = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(points, destination)
    warped = cv2.warpPerspective(np.asarray(frame), matrix, (out_w, out_h))
    return np.ascontiguousarray(warped)


def _order_corners(points: object) -> object:
    import numpy as np

    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _clip_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = min(max(int(x1), 0), max(width - 1, 0))
    y1 = min(max(int(y1), 0), max(height - 1, 0))
    x2 = min(max(int(x2), x1 + 1), width)
    y2 = min(max(int(y2), y1 + 1), height)
    return x1, y1, x2, y2
