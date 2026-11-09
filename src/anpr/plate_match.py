"""Identity matching for sticky plate history.

Partial OCR forms of the same plate (CAKW568 vs AKW568, AKW568 vs AMW568)
must update one slot instead of opening a new history entry. Among spellings
of one identity, prefer longer text, then higher confidence, then crop quality.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

NEAR_DUP_EDITS = 2
MATCH_ALIASES = 10
CONF_FLIP_MARGIN = 0.05


def same_identity(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left == right:
        return True
    if compatible_longer(left, right) is not None:
        return True
    if is_one_char_shorter(left, right) or is_one_char_shorter(right, left):
        return True
    if len(left) == len(right):
        return _hamming(left, right) <= NEAR_DUP_EDITS
    return False


def merge_text(
    current: str,
    current_confidence: float,
    new: str,
    new_confidence: float,
    *,
    aliases: Sequence[str] = (),
    best_by_text: Mapping[str, float] | None = None,
    quality_by_text: Mapping[str, float] | None = None,
    current_quality: float = 0.0,
    new_quality: float = 0.0,
) -> tuple[str, float]:
    """Pick the better plate text/confidence for one identity."""
    if not current:
        return new, new_confidence
    if not new:
        return current, current_confidence

    scores: dict[str, tuple[float, float, int]] = {}
    counts = Counter(aliases)
    counts[current] += 0
    counts[new] += 1

    conf_map = dict(best_by_text or {})
    conf_map[current] = max(float(conf_map.get(current, 0.0)), float(current_confidence))
    conf_map[new] = max(float(conf_map.get(new, 0.0)), float(new_confidence))

    qual_map = dict(quality_by_text or {})
    qual_map[current] = max(float(qual_map.get(current, 0.0)), float(current_quality))
    qual_map[new] = max(float(qual_map.get(new, 0.0)), float(new_quality))

    for text in set(counts) | set(conf_map) | {current, new}:
        if not same_identity(current, text) and not same_identity(new, text):
            continue
        scores[text] = (
            float(conf_map.get(text, 0.0)),
            float(qual_map.get(text, 0.0)),
            int(counts.get(text, 0)),
        )

    if not scores:
        return new, new_confidence

    winner = max(scores, key=lambda text: _spelling_key(text, scores[text]))
    # Text follows the best spelling; confidence is the best score seen for this
    # identity so a later high-conf short read still raises the slot confidence.
    best_confidence = max(score[0] for score in scores.values())
    return winner, float(best_confidence)


def clearly_better(
    current_confidence: float,
    new_confidence: float,
    *,
    current_quality: float = 0.0,
    new_quality: float = 0.0,
) -> bool:
    """True when the new read should be allowed to challenge sticky spelling."""
    if float(new_confidence) >= float(current_confidence) + CONF_FLIP_MARGIN:
        return True
    if float(new_confidence) + 1e-6 < float(current_confidence):
        return False
    return float(new_quality) > float(current_quality) + 1e-6


def compatible_longer(left: str, right: str) -> str | None:
    if left == right:
        return left
    if not left or not right:
        return None
    if left.endswith(right) or right.endswith(left):
        return left if len(left) >= len(right) else right
    if left.startswith(right) or right.startswith(left):
        return left if len(left) >= len(right) else right
    return None


def is_one_char_shorter(longer: str, shorter: str) -> bool:
    if not longer or not shorter or len(longer) != len(shorter) + 1:
        return False
    for index in range(len(longer)):
        if longer[:index] + longer[index + 1 :] == shorter:
            return True
    return False


def _spelling_key(text: str, score: tuple[float, float, int]) -> tuple:
    conf, quality, count = score
    # Longer plate text wins first (CAKW568 over AKW568), then confidence,
    # then crop quality, then how often that spelling was seen.
    return (len(text), conf, quality, count)


def _hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))
