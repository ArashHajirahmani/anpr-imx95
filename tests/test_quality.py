import numpy as np

from anpr.plates import OCR_HEIGHT, OCR_WIDTH
from anpr.quality import (
    EnhancerSettings,
    QualitySettings,
    crop_quality,
    enhance_crop,
)


def test_quality_rejects_flat_dark_crop() -> None:
    crop = np.full((40, 120, 3), 10, dtype=np.uint8)
    ok, metrics, reason = crop_quality(crop, QualitySettings())
    assert ok is False
    assert reason in {"blurry", "dark", "low_contrast"}
    assert metrics.brightness < 40


def test_quality_accepts_contrasty_crop() -> None:
    crop = np.zeros((40, 120, 3), dtype=np.uint8)
    crop[:, :60] = 30
    crop[:, 60:] = 220
    # Add high-frequency edges so Laplacian variance clears the floor.
    crop[::2, ::2] = 255
    ok, _metrics, reason = crop_quality(crop, QualitySettings(min_laplacian_var=10))
    assert ok is True
    assert reason == ""


def test_quality_can_be_disabled() -> None:
    crop = np.full((20, 40, 3), 5, dtype=np.uint8)
    ok, _metrics, reason = crop_quality(crop, QualitySettings(enabled=False))
    assert ok is True
    assert reason == ""


def test_enhancer_keeps_bgr_uint8_shape() -> None:
    crop = np.random.default_rng(0).integers(0, 255, size=(50, 100, 3), dtype=np.uint8)
    out = enhance_crop(
        crop,
        EnhancerSettings(enabled=True, clahe=True, gamma=1.1, sharpen=True),
    )
    assert out.dtype == np.uint8
    assert out.shape == crop.shape


def test_ocr_preprocess_contract_after_enhance() -> None:
    import cv2

    crop = np.random.default_rng(1).integers(40, 200, size=(48, 160, 3), dtype=np.uint8)
    enhanced = enhance_crop(crop, EnhancerSettings(enabled=True))
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (OCR_WIDTH, OCR_HEIGHT), interpolation=cv2.INTER_LINEAR)
    batch = np.ascontiguousarray(resized[None, ...])
    assert batch.dtype == np.uint8
    assert batch.shape == (1, OCR_HEIGHT, OCR_WIDTH, 3)
