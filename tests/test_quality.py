import numpy as np

from anpr.plates import OCR_HEIGHT, OCR_WIDTH
from anpr.quality import (
    EnhancerSettings,
    QualitySettings,
    crop_quality,
    crop_sharpness,
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

    from anpr.plates import resize_rgb_for_ocr

    crop = np.random.default_rng(1).integers(40, 200, size=(48, 160, 3), dtype=np.uint8)
    enhanced = enhance_crop(crop, EnhancerSettings(enabled=True))
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    resized = resize_rgb_for_ocr(rgb)
    batch = np.ascontiguousarray(resized[None, ...])
    assert batch.dtype == np.uint8
    assert batch.shape == (1, OCR_HEIGHT, OCR_WIDTH, 3)


def _text_like(width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), 230, dtype=np.uint8)
    step = max(2, width // 16)
    for x in range(step, width - step, step * 2):
        image[height // 5 : height * 4 // 5, x : x + step // 2 + 1] = 20
    return image


def test_sharpness_separates_blurred_from_sharp() -> None:
    import cv2

    sharp = _text_like(256, 128)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 6)
    assert crop_sharpness(sharp) > 150
    assert crop_sharpness(blurred) < 50


def test_sharpness_does_not_depend_on_plate_size() -> None:
    import cv2

    base = _text_like(512, 256)
    sizes = [(512, 256), (256, 128), (160, 80)]
    values = [
        crop_sharpness(cv2.resize(base, size, interpolation=cv2.INTER_AREA)) for size in sizes
    ]
    assert max(values) / min(values) < 1.5


def test_sharpness_of_an_empty_crop_is_zero() -> None:
    assert crop_sharpness(np.zeros((0, 0, 3), dtype=np.uint8)) == 0.0
