import numpy as np
import pytest

from anpr.plates import (
    OCR_HEIGHT,
    OCR_WIDTH,
    accept_plate,
    decode_characters,
    nms,
    ocr_floor,
    ocr_resize_method,
    resize_rgb_for_ocr,
    yolo_candidates,
)


def test_decode_strips_padding() -> None:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    pad = alphabet.index("_")
    digit = alphabet.index("A")
    steps = [[0.0] * len(alphabet) for _ in range(3)]
    steps[0][digit] = 5
    steps[1][pad] = 5
    steps[2][pad] = 4
    text, confidence, char_confidences, margins, blank_uncertainty = decode_characters(
        steps
    )
    assert text == "A"
    assert confidence > 0.5
    assert len(char_confidences) == 1
    assert char_confidences[0] > 0.5
    assert len(margins) == 1
    assert blank_uncertainty >= 0.0


def test_decode_collapses_repeated_non_blanks() -> None:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    pad = alphabet.index("_")
    c_idx = alphabet.index("C")
    a_idx = alphabet.index("A")
    steps = [[0.0] * len(alphabet) for _ in range(5)]
    steps[0][c_idx] = 2.0  # weaker C (logits → softmax path)
    steps[1][c_idx] = 5.0  # stronger C — same run
    steps[2][pad] = 5.0
    steps[3][a_idx] = 4.0
    steps[4][a_idx] = 3.0
    text, confidence, char_confidences, margins, _blank = decode_characters(steps)
    assert text == "CA"
    assert len(char_confidences) == 2
    assert char_confidences[0] > char_confidences[1] or char_confidences[0] > 0.5
    assert confidence == min(char_confidences)
    assert len(margins) == 2


def test_decode_uses_probability_steps_without_softmax() -> None:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    pad = alphabet.index("_")
    c_idx = alphabet.index("C")
    a_idx = alphabet.index("A")
    steps = [[0.0] * len(alphabet) for _ in range(3)]
    steps[0][c_idx] = 0.91
    steps[0][pad] = 0.09
    steps[1] = [0.0] * len(alphabet)
    steps[1][c_idx] = 0.55
    steps[1][a_idx] = 0.45
    steps[2][a_idx] = 0.88
    steps[2][pad] = 0.12
    text, confidence, char_confidences, margins, _blank = decode_characters(steps)
    assert text == "CA"
    assert char_confidences[0] == 0.91  # max of the C run
    assert char_confidences[1] == 0.88
    assert confidence == 0.88
    assert len(margins) == 2


def test_accept_plate_rejects_weak_character() -> None:
    assert (
        accept_plate(
            "AB",
            0.5,
            0.35,
            min_char_confidence=0.35,
            char_confidences=(0.9, 0.02),
        )
        is False
    )
    assert accept_plate(
        "AB",
        0.5,
        0.35,
        min_char_confidence=0.35,
        char_confidences=(0.9, 0.4),
    )


def test_nms_keeps_the_higher_confidence_box() -> None:
    boxes = [
        (0.0, 0.0, 10.0, 10.0, 0.4),
        (1.0, 1.0, 11.0, 11.0, 0.9),
        (40.0, 40.0, 50.0, 50.0, 0.6),
    ]
    kept = nms(boxes, iou_threshold=0.45)
    assert len(kept) == 2
    assert kept[0][4] == 0.9
    assert yolo_candidates([(1, 1, 2, 2, 0.2), (1, 1, 2, 2, 0.8)], 0.5) == [(1, 1, 2, 2, 0.8)]


def test_confidence_floor_matches_high_trust() -> None:
    assert accept_plate("777W455", 0.70)
    assert accept_plate("777W455", 0.69) is False
    assert accept_plate("", 0.9) is False
    assert ocr_floor(0.70, "cpu") == 0.70
    assert ocr_floor(0.70, "npu") == 0.695
    assert accept_plate("192XC", 0.695, ocr_floor(0.70, "npu"))


def test_ocr_resize_method_picks_area_for_downscale() -> None:
    assert ocr_resize_method(256, 128) == "area"
    assert ocr_resize_method(40, 20) == "lanczos4"
    assert ocr_resize_method(OCR_WIDTH, OCR_HEIGHT) == "linear"
    assert ocr_resize_method(200, 40) == "lanczos4"


def test_resize_rgb_for_ocr_shape_and_downscale_kernel() -> None:
    pytest.importorskip("cv2")
    large = np.zeros((128, 256, 3), dtype=np.uint8)
    large[:, :] = (10, 20, 30)
    out = resize_rgb_for_ocr(large)
    assert out.shape == (OCR_HEIGHT, OCR_WIDTH, 3)
    small = np.zeros((32, 64, 3), dtype=np.uint8)
    assert resize_rgb_for_ocr(small).shape == (OCR_HEIGHT, OCR_WIDTH, 3)
