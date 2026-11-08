"""Refine a YOLO plate box before enhance/OCR.

Expand left/right when ink touches the crop edge. Optionally find a scored
quadrilateral and warp it upright. Both steps are capped relative to the
original detector ROI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("anpr.roi_refine")


@dataclass(frozen=True)
class ExpandSettings:
    enabled: bool = True
    # Max total width growth as a fraction of the original ROI width.
    max_ratio: float = 0.25
    # Border strip width as a fraction of ROI width used to detect edge ink.
    edge_fraction: float = 0.08
    # Mean Canny response (0..1) on a border strip that triggers expand.
    ink_threshold: float = 0.12
    # Target pad on a triggered side, as a fraction of original ROI width.
    pad_fraction: float = 0.12


@dataclass(frozen=True)
class PerspectiveSettings:
    enabled: bool = True
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
    """Expand and/or perspective-warp a detector box. Falls back to the YOLO crop."""
    import numpy as np

    frame = np.asarray(image)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, width, height)
    orig = (x1, y1, x2, y2)
    method = "yolo"

    if settings.expand.enabled:
        x1, y1, x2, y2 = _expand_box(frame, x1, y1, x2, y2, settings.expand)
        x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, width, height)
        x1, y1, x2, y2 = _cap_box_to_ratio(orig, (x1, y1, x2, y2), settings.expand.max_ratio)
        x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, width, height)
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


def _expand_box(
    frame: object,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    settings: ExpandSettings,
) -> tuple[int, int, int, int]:
    import cv2
    import numpy as np

    crop = np.asarray(frame)[y1:y2, x1:x2]
    if crop.size == 0:
        return x1, y1, x2, y2
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    orig_w = max(1, x2 - x1)
    max_extra = max(1, int(round(orig_w * settings.max_ratio)))
    target = max(1, int(round(orig_w * settings.pad_fraction)))
    left_need = _edge_ink(gray, "left", settings.edge_fraction) >= settings.ink_threshold
    right_need = _edge_ink(gray, "right", settings.edge_fraction) >= settings.ink_threshold
    left_pad = 0
    right_pad = 0
    if left_need:
        left_pad = min(target, max_extra)
    if right_need:
        right_pad = min(target, max_extra - left_pad)
    if left_pad == 0 and right_need:
        right_pad = min(target, max_extra)
    if not left_need and not right_need:
        return x1, y1, x2, y2
    return x1 - left_pad, y1, x2 + right_pad, y2


def _edge_ink(gray: object, side: str, edge_fraction: float) -> float:
    import cv2
    import numpy as np

    image = np.asarray(gray)
    height, width = image.shape[:2]
    strip_w = max(1, int(round(width * edge_fraction)))
    if side == "left":
        strip = image[:, :strip_w]
    else:
        strip = image[:, width - strip_w :]
    blurred = cv2.GaussianBlur(strip, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return float(edges.mean()) / 255.0


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
    # Search around the original YOLO box, not unbounded.
    sx1 = max(0, ox1 - pad_x)
    sy1 = max(0, oy1 - pad_y)
    sx2 = min(width, ox2 + pad_x)
    sy2 = min(height, oy2 + pad_y)
    # Also keep the expanded box inside the search window.
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
    # Map quad into full-frame coordinates, then score vs original YOLO.
    ranked = quad.astype(np.float32) + np.array([sx1, sy1], dtype=np.float32)
    score = _score_quad(ranked, original, settings)
    if score < settings.min_score:
        return None
    warped = _warp_quad(image, ranked, original, settings)
    return warped


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
        # Prefer larger plate-like quads that are not the full search window.
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
    # Distance from YOLO center in units of original size.
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


def _cap_box_to_ratio(
    original: tuple[int, int, int, int],
    box: tuple[int, int, int, int],
    max_ratio: float,
) -> tuple[int, int, int, int]:
    ox1, oy1, ox2, oy2 = original
    x1, y1, x2, y2 = box
    orig_w = max(1, ox2 - ox1)
    orig_h = max(1, oy2 - oy1)
    max_w = int(round(orig_w * (1.0 + max_ratio)))
    max_h = int(round(orig_h * (1.0 + max_ratio)))
    width = x2 - x1
    height = y2 - y1
    if width > max_w:
        trim = width - max_w
        left_trim = trim // 2
        right_trim = trim - left_trim
        # Prefer keeping the side that moved less from the original.
        left_growth = max(0, ox1 - x1)
        right_growth = max(0, x2 - ox2)
        if left_growth >= right_growth:
            x1 += min(left_trim, left_growth)
            rem = left_trim - min(left_trim, left_growth) + right_trim
            x2 -= rem
        else:
            x2 -= min(right_trim, right_growth)
            rem = right_trim - min(right_trim, right_growth) + left_trim
            x1 += rem
    if height > max_h:
        cy = (y1 + y2) // 2
        y1 = cy - max_h // 2
        y2 = y1 + max_h
    return x1, y1, x2, y2


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
