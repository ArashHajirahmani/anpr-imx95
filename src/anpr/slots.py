"""Slot scoring helpers: per-slot probabilities, candidate texts, and text likelihoods.

The OCR model has one classification head per character slot (10 for CCT), not a CTC
alignment: slot i is character i of the plate and `_` pads the tail, so a repeated
glyph such as the two 7s of "077" sits in two adjacent slots and must be kept.

Pure stdlib so the fusion logic runs on a bare dev host. ``steps`` is the OCR output
for one frame, one probability vector per slot.
"""

from __future__ import annotations

import math

from anpr.plates import PLATE_PAD, _looks_like_probabilities, _softmax_probs

NEG_INF = float("-inf")
PROB_FLOOR = 1e-6


def step_probabilities(steps: list[list[float]]) -> list[list[float]]:
    """Normalise each slot to probabilities (the CCT export already emits them)."""
    out: list[list[float]] = []
    for values in steps:
        if _looks_like_probabilities(values):
            out.append([float(v) for v in values])
        else:
            out.append(_softmax_probs(values))
    return out


def _log(prob: float) -> float:
    return math.log(max(prob, PROB_FLOOR))


def slot_log_likelihood(
    probs: list[list[float]],
    text: str,
    alphabet: str,
    pad: str = PLATE_PAD,
) -> float:
    """log P(text | slots): characters in slots 0..n-1, pad in every slot after."""
    blank = alphabet.find(pad)
    if not probs or not text or len(text) > len(probs):
        return NEG_INF
    total = 0.0
    for slot, step in enumerate(probs):
        if slot < len(text):
            index = alphabet.find(text[slot])
            if index < 0 or index == blank:
                return NEG_INF
        else:
            index = blank
        total += _log(step[index])
    return total


def beam_candidates(
    probs: list[list[float]],
    alphabet: str,
    beam: int = 8,
    prune: float = 0.001,
    pad: str = PLATE_PAD,
) -> list[tuple[str, float]]:
    """Beam search over slots. Returns up to ``beam`` (text, log P) pairs, best first."""
    blank = alphabet.find(pad)
    if not probs or blank < 0:
        return []
    # tail[i] = log P(pad in every slot from i on): the cost of ending the text at slot i.
    tail = [0.0] * (len(probs) + 1)
    for slot in range(len(probs) - 1, -1, -1):
        tail[slot] = tail[slot + 1] + _log(probs[slot][blank])
    beams: list[tuple[str, float]] = [("", 0.0)]
    finished: dict[str, float] = {}
    for slot, step in enumerate(probs):
        for text, score in beams:
            if text:
                finished[text] = score + tail[slot]
        grown: list[tuple[str, float]] = []
        for text, score in beams:
            for cls, p in enumerate(step):
                if cls >= len(alphabet) or cls == blank or p < prune:
                    continue
                grown.append((text + alphabet[cls], score + _log(p)))
        grown.sort(key=lambda item: item[1], reverse=True)
        beams = grown[:beam]
    for text, score in beams:
        finished[text] = score
    out = sorted(finished.items(), key=lambda item: item[1], reverse=True)
    return out[:beam]


def mean_step_entropy(probs: list[list[float]]) -> float:
    """Mean per-step entropy (nats). Low for a crisp read, high for a guess."""
    if not probs:
        return 0.0
    total = 0.0
    for step in probs:
        total -= sum(p * math.log(p) for p in step if p > 1e-9)
    return total / len(probs)
