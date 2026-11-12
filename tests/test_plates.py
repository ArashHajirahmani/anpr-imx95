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
    text, confidence, char_confidences, margins, blank_uncertainty = decode_characters(steps)
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


def test_join_detector_outputs_merges_split_head() -> None:
    import numpy as np

    from anpr.recognize import join_detector_outputs

    boxes = np.arange(8, dtype=np.float32).reshape(1, 4, 2)
    scores = np.array([[[0.9, 0.1]]], dtype=np.float32)
    joined = join_detector_outputs([scores, boxes])
    assert joined.shape == (1, 5, 2)
    assert np.array_equal(joined[:, :4], boxes)
    assert np.array_equal(joined[:, 4:], scores)
    single = np.zeros((1, 5, 2), dtype=np.float32)
    assert join_detector_outputs([single]) is single


def test_uint8_letterbox_matches_quantized_float_path() -> None:
    import numpy as np

    pytest.importorskip("cv2")
    from anpr.recognize import _letterbox, _letterbox_u8

    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(90, 160, 3), dtype=np.uint8)
    blob, scale, left, top = _letterbox(image, 64)
    fast, fast_scale, fast_left, fast_top = _letterbox_u8(image, 64)
    assert (scale, left, top) == (fast_scale, fast_left, fast_top)
    quantized = np.clip(np.round(blob / (1 / 255)) + -128, -128, 127).astype(np.int8)
    assert fast.dtype == np.uint8
    assert np.array_equal(np.bitwise_xor(fast, 128).view(np.int8), quantized)


def test_decode_dfl_head_matches_a_hand_computed_box() -> None:
    from anpr.recognize import DFL_BINS, decode_dfl_head

    size, strides = 32, (8, 16)  # 4x4 + 2x2 anchors = 20
    count = 4 * 4 + 2 * 2
    logits = np.zeros((1, 64, count), dtype=np.float32)
    scores = np.zeros((1, 1, count), dtype=np.float32)
    # Anchor 5 is on the stride-8 grid at cell (x=1, y=1) -> center (1.5, 1.5) in stride units.
    # Peak the bins so the distances are l=1, t=2, r=3, b=4.
    for side, peak in enumerate((1, 2, 3, 4)):
        logits[0, side * DFL_BINS + peak, 5] = 50.0
    scores[0, 0, 5] = 0.9
    scores[0, 0, 6] = 0.1  # below threshold: must stay zero
    out = decode_dfl_head(logits, scores, 0.5, size=size, strides=strides)
    assert out.shape == (1, 5, count)
    # x1y1 = (1.5-1, 1.5-2) = (0.5, -0.5); x2y2 = (1.5+3, 1.5+4) = (4.5, 5.5); stride 8.
    assert np.allclose(out[0, :, 5], [2.5 * 8, 2.5 * 8, 4.0 * 8, 6.0 * 8, 0.9], atol=1e-3)
    assert not out[0, :, 6].any()
    assert np.count_nonzero(out[0, 4]) == 1


def test_decode_dfl_head_with_no_candidates_is_all_zero() -> None:
    from anpr.recognize import decode_dfl_head

    logits = np.zeros((1, 64, 20), dtype=np.float32)
    scores = np.zeros((1, 1, 20), dtype=np.float32)
    assert not decode_dfl_head(logits, scores, 0.5, size=32, strides=(8, 16)).any()


def test_join_detector_outputs_decodes_dfl_pair() -> None:
    from anpr.recognize import join_detector_outputs

    logits = np.zeros((1, 64, 8400), dtype=np.float32)
    scores = np.zeros((1, 1, 8400), dtype=np.float32)
    assert join_detector_outputs([scores, logits], 0.25).shape == (1, 5, 8400)
