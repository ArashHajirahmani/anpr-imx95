"""Plate-model decoding that does not import ONNX Runtime."""

import math

PLATE_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
PLATE_PAD = "_"
OCR_WIDTH = 128
OCR_HEIGHT = 64
# Minimum per-character softmax after CTC collapse. Chance over 37 symbols is ~0.027.
# 0.70 is the high-trust floor from board OCR logs: every kept glyph must clear it.
MIN_OCR_CONFIDENCE = 0.70
# The INT8 OCR score for the same crop sits just under the float score.
INT8_OCR_MARGIN = 0.005


def ocr_resize_method(
    src_width: int,
    src_height: int,
    dst_width: int = OCR_WIDTH,
    dst_height: int = OCR_HEIGHT,
) -> str:
    """Pick OpenCV resize mode for mapping a plate crop onto the OCR canvas.

    Downscales with area averaging so thin strokes survive reduction. Upscales
    (and mixed one-up/one-down cases) with Lanczos instead of bilinear.
    """
    if src_width > dst_width and src_height > dst_height:
        return "area"
    if src_width < dst_width or src_height < dst_height:
        return "lanczos4"
    return "linear"


def resize_rgb_for_ocr(
    rgb: object,
    width: int = OCR_WIDTH,
    height: int = OCR_HEIGHT,
) -> object:
    """Resize an RGB uint8 crop to the OCR input size with a scale-aware kernel."""
    import cv2
    import numpy as np

    image = np.asarray(rgb)
    if image.size == 0:
        return image
    src_height, src_width = image.shape[:2]
    if src_width == width and src_height == height:
        return np.ascontiguousarray(image)
    method = ocr_resize_method(src_width, src_height, width, height)
    if method == "area":
        interpolation = cv2.INTER_AREA
    elif method == "lanczos4":
        interpolation = cv2.INTER_LANCZOS4
    else:
        interpolation = cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def decode_characters(
    steps: list[list[float]],
    alphabet: str = PLATE_ALPHABET,
    pad: str = PLATE_PAD,
) -> tuple[str, float, list[float], list[float], float]:
    """CTC-greedy decode: drop blanks, collapse repeats, report min char conf.

    Returns text, min confidence, per-char confidences, per-char top1-top2
    margins, and the worst blank-competition uncertainty anywhere in the steps.
    """
    if not steps:
        return "", 0.0, [], [], 0.0
    chars: list[str] = []
    confidences: list[float] = []
    margins: list[float] = []
    blank_uncertainty = 0.0
    for step in steps:
        index, confidence, margin, probs = _step_choice(step)
        character = alphabet[index] if index < len(alphabet) else pad
        if character == pad:
            second = _second_best_non_blank(probs, alphabet, pad)
            if second is not None:
                blank_uncertainty = max(blank_uncertainty, second)
            continue
        if chars and character == chars[-1]:
            if confidence > confidences[-1]:
                confidences[-1] = confidence
                margins[-1] = margin
            continue
        chars.append(character)
        confidences.append(confidence)
        margins.append(margin)
    if not chars:
        return "", 0.0, [], [], blank_uncertainty
    return "".join(chars), min(confidences), confidences, margins, blank_uncertainty


def ocr_floor(minimum: float, runtime: str) -> float:
    """Return the accept threshold for this runtime."""
    if runtime == "npu":
        return max(0.0, minimum - INT8_OCR_MARGIN)
    return minimum


def accept_plate(
    text: str,
    confidence: float,
    minimum: float = MIN_OCR_CONFIDENCE,
    *,
    min_char_confidence: float | None = None,
    char_confidences: list[float] | tuple[float, ...] | None = None,
) -> bool:
    """Keep a read when min character confidence clears the floor(s)."""
    if not text:
        return False
    if min_char_confidence is not None and char_confidences is not None:
        if any(score < min_char_confidence for score in char_confidences):
            return False
    return confidence >= minimum


def yolo_candidates(
    rows: list[tuple[float, float, float, float, float]],
    threshold: float,
) -> list[tuple[float, float, float, float, float]]:
    """Keep rows of center-x, center-y, width, height, confidence above the threshold."""
    return [row for row in rows if row[4] >= threshold]


def nms(
    boxes: list[tuple[float, float, float, float, float]],
    iou_threshold: float = 0.45,
) -> list[tuple[float, float, float, float, float]]:
    """Greedy non-maximum suppression. Boxes are x1, y1, x2, y2, confidence."""
    ordered = sorted(boxes, key=lambda box: box[4], reverse=True)
    kept: list[tuple[float, float, float, float, float]] = []
    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        ordered = [box for box in ordered if _iou(best, box) < iou_threshold]
    return kept


def center_to_corners(
    cx: float,
    cy: float,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2


def _step_choice(values: list[float]) -> tuple[int, float, float, list[float]]:
    """Argmax with confidence and top1-top2 margin."""
    if not values:
        return 0, 0.0, 0.0, []
    if _looks_like_probabilities(values):
        probs = [float(value) for value in values]
    else:
        probs = _softmax_probs(values)
    index = max(range(len(probs)), key=probs.__getitem__)
    ordered = sorted(probs, reverse=True)
    top1 = ordered[0] if ordered else 0.0
    top2 = ordered[1] if len(ordered) > 1 else 0.0
    return index, float(probs[index]), max(0.0, top1 - top2), probs


def _second_best_non_blank(
    probs: list[float],
    alphabet: str,
    pad: str,
) -> float | None:
    """Probability of the best non-blank class when blank won the step."""
    pad_index = alphabet.find(pad)
    best = None
    for index, value in enumerate(probs):
        if index == pad_index:
            continue
        if index >= len(alphabet):
            continue
        if best is None or value > best:
            best = float(value)
    return best


def _looks_like_probabilities(values: list[float]) -> bool:
    """CCT exports NHWC probs that already sum to ~1 per timestep."""
    if any(value < -1e-4 or value > 1.0 + 1e-3 for value in values):
        return False
    total = sum(values)
    return 0.75 <= total <= 1.25


def _softmax_probs(values: list[float]) -> list[float]:
    peak = max(values)
    scores = [math.exp(value - peak) for value in values]
    total = sum(scores) or 1.0
    return [score / total for score in scores]


def _softmax_choice(values: list[float]) -> tuple[int, float]:
    probs = _softmax_probs(values)
    index = max(range(len(probs)), key=probs.__getitem__)
    return index, probs[index]


def _iou(
    left: tuple[float, float, float, float, float],
    right: tuple[float, float, float, float, float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection == 0:
        return 0.0
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    if union <= 0:
        return 0.0
    return intersection / union
