import numpy as np
import pytest

from anpr.roi_refine import (
    ExpandSettings,
    PerspectiveSettings,
    RoiRefineSettings,
    refine_plate_roi,
)


def _settings(*, pad_ratio: float = 0.05, perspective: bool = False) -> RoiRefineSettings:
    return RoiRefineSettings(
        expand=ExpandSettings(
            enabled=True,
            inner_ratio=0.20,
            outer_ratio=0.20,
            edge_threshold=0.05,
            pad_ratio=pad_ratio,
        ),
        perspective=PerspectiveSettings(enabled=perspective),
    )


def test_border_snap_expands_left_to_outer_plate_edge() -> None:
    pytest.importorskip("cv2")
    frame = np.full((100, 220, 3), 40, dtype=np.uint8)
    frame[35:70, 30:160] = 220
    frame[35:70, 30:34] = 10
    frame[35:70, 156:160] = 10
    refined = refine_plate_roi(frame, 50, 35, 150, 70, _settings(pad_ratio=0.0))
    assert refined.method == "expand"
    assert refined.x1 < 50
    assert refined.y1 == 35
    assert refined.y2 == 70


def test_border_snap_recenters_shifted_yolo_box() -> None:
    pytest.importorskip("cv2")
    frame = np.full((120, 320, 3), (90, 70, 180), dtype=np.uint8)
    frame[45:85, 80:220] = 220
    frame[45:85, 80:86] = 5
    frame[45:85, 214:220] = 5
    refined = refine_plate_roi(frame, 100, 44, 245, 86, _settings(pad_ratio=0.0))
    assert refined.method == "expand"
    assert refined.x1 < 100
    assert refined.x2 < 245
    assert refined.y1 == 44
    assert refined.y2 == 86


def test_pad_ratio_uses_width_and_height() -> None:
    pytest.importorskip("cv2")
    frame = np.full((100, 240, 3), 80, dtype=np.uint8)
    frame[40:70, 50:170] = 220
    frame[40:70, 50:54] = 5
    frame[40:70, 166:170] = 5
    tight = refine_plate_roi(frame, 55, 40, 175, 70, _settings(pad_ratio=0.0))
    padded = refine_plate_roi(frame, 55, 40, 175, 70, _settings(pad_ratio=0.05))
    assert padded.x1 < tight.x1
    assert padded.x2 > tight.x2
    # Top/bottom are YOLO ± pad only (no vertical edge snap).
    assert padded.y1 < 40
    assert padded.y2 > 70
    assert padded.y1 == 40 - int(round(30 * 0.05))
    assert padded.y2 == 70 + int(round(30 * 0.05))


def test_disabled_refine_keeps_yolo_box() -> None:
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    frame[10:30, 20:60] = 255
    refined = refine_plate_roi(frame, 20, 10, 60, 30, RoiRefineSettings.disabled())
    assert (refined.x1, refined.y1, refined.x2, refined.y2) == (20, 10, 60, 30)
    assert refined.method == "yolo"


def test_perspective_warps_clear_plate_quad() -> None:
    cv2 = pytest.importorskip("cv2")
    frame = np.full((120, 240, 3), 30, dtype=np.uint8)
    pts = np.array([[50, 40], [190, 35], [195, 85], [45, 90]], dtype=np.int32)
    cv2.fillConvexPoly(frame, pts, (230, 230, 230))
    cv2.polylines(frame, [pts], True, (10, 10, 10), 2)
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
