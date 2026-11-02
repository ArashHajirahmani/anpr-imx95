"""Plate-model decoding that does not import ONNX Runtime."""

import math

PLATE_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
PLATE_PAD = "_"
OCR_WIDTH = 128
OCR_HEIGHT = 64
# Mean per-character probability. Chance over this 37-symbol alphabet is about 0.027.
# Reads around 0.07 are the ones that match the plate; lower scores are noise.
MIN_OCR_CONFIDENCE = 0.07
# The INT8 OCR score for the same crop sits just under the float score.
INT8_OCR_MARGIN = 0.005


def decode_characters(
    steps: list[list[float]],
    alphabet: str = PLATE_ALPHABET,
    pad: str = PLATE_PAD,
) -> tuple[str, float]:
    """Turn per-character logits into text and a mean confidence."""
    if not steps:
        return "", 0.0
    chars: list[str] = []
    confidences: list[float] = []
    for step in steps:
        index, confidence = _softmax_choice(step)
        character = alphabet[index] if index < len(alphabet) else pad
        if character == pad:
            continue
        chars.append(character)
        confidences.append(confidence)
    if not chars:
        return "", 0.0
    return "".join(chars), sum(confidences) / len(confidences)


def ocr_floor(minimum: float, runtime: str) -> float:
    """Return the accept threshold for this runtime."""
    if runtime == "npu":
        return max(0.0, minimum - INT8_OCR_MARGIN)
    return minimum


def accept_plate(text: str, confidence: float, minimum: float = MIN_OCR_CONFIDENCE) -> bool:
    """Keep a read only when it has text at or above the accurate confidence floor."""
    return bool(text) and confidence >= minimum


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


def _softmax_choice(values: list[float]) -> tuple[int, float]:
    peak = max(values)
    scores = [math.exp(value - peak) for value in values]
    total = sum(scores)
    index = max(range(len(values)), key=values.__getitem__)
    return index, scores[index] / total


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
