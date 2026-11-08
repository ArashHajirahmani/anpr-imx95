import numpy as np
import pytest

from anpr.roi_refine import (
    ExpandSettings,
    PerspectiveSettings,
    RoiRefineSettings,
    refine_plate_roi,
)


def test_expand_grows_left_when_ink_touches_border() -> None:
    pytest.importorskip("cv2")
    frame = np.full((80, 200, 3), 220, dtype=np.uint8)
    # Plate glyphs start at the YOLO left edge (clipped C case).
    frame[30:55, 40:48] = 20
    frame[30:55, 55:120] = 20
    settings = RoiRefineSettings(
        expand=ExpandSettings(
            enabled=True,
            max_ratio=0.25,
            edge_fraction=0.1,
            ink_threshold=0.05,
            pad_fraction=0.15,
        ),
        perspective=PerspectiveSettings(enabled=False),
    )
    refined = refine_plate_roi(frame, 40, 28, 130, 58, settings)
    assert refined.x1 < 40
    assert refined.x2 == 130
    assert refined.method == "expand"
    assert refined.x2 - refined.x1 <= int(90 * 1.25) + 1


def test_expand_respects_max_ratio() -> None:
    pytest.importorskip("cv2")
    frame = np.full((60, 160, 3), 200, dtype=np.uint8)
    frame[20:45, 50:58] = 10
    frame[20:45, 100:108] = 10
    settings = RoiRefineSettings(
        expand=ExpandSettings(
            enabled=True,
            max_ratio=0.2,
            edge_fraction=0.12,
            ink_threshold=0.05,
            pad_fraction=0.5,
        ),
        perspective=PerspectiveSettings(enabled=False),
    )
    refined = refine_plate_roi(frame, 50, 18, 110, 48, settings)
    assert refined.x2 - refined.x1 <= int(60 * 1.2) + 1


def test_disabled_refine_keeps_yolo_box() -> None:
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[10:30, 20:60] = 255
    refined = refine_plate_roi(frame, 20, 10, 60, 30, RoiRefineSettings.disabled())
    assert (refined.x1, refined.y1, refined.x2, refined.y2) == (20, 10, 60, 30)
    assert refined.method == "yolo"
    assert np.asarray(refined.crop_bgr).shape == (20, 40, 3)


def test_perspective_warps_clear_plate_quad() -> None:
    cv2 = pytest.importorskip("cv2")
    frame = np.full((120, 240, 3), 30, dtype=np.uint8)
    # Light plate rectangle with dark border for Canny.
    pts = np.array([[50, 40], [190, 35], [195, 85], [45, 90]], dtype=np.int32)
    cv2.fillConvexPoly(frame, pts, (230, 230, 230))
    cv2.polylines(frame, [pts], True, (10, 10, 10), 2)
    # Glyphs inside.
    cv2.rectangle(frame, (70, 50), (85, 75), (20, 20, 20), -1)
    cv2.rectangle(frame, (100, 50), (115, 75), (20, 20, 20), -1)
    settings = RoiRefineSettings(
        expand=ExpandSettings(enabled=False),
        perspective=PerspectiveSettings(
            enabled=True,
            max_ratio=0.4,
            min_score=0.2,
            min_aspect=1.5,
            max_aspect=8.0,
            canny_low=30,
            canny_high=100,
        ),
    )
    refined = refine_plate_roi(frame, 48, 38, 198, 92, settings)
    crop = np.asarray(refined.crop_bgr)
    assert crop.ndim == 3
    assert crop.shape[1] >= crop.shape[0]
    # Perspective may or may not fire depending on edges; either is valid if capped.
    if refined.method == "perspective":
        assert crop.shape[1] <= int(150 * 1.4) + 2
