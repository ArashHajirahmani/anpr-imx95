"""Temporal plate voting ladder.

OCR candidates accumulate in a short window. High-trust reads still need a
second agreeing sample before release so single-frame OCR flips (K↔M, missing
digits) do not publish. Longer prefix/suffix-compatible strings win over
shorter ones, and a sticky last release (kept per stream) blocks same-length
near-duplicates until they outvote it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from anpr.recognize import PlateRead
from anpr.types import PlateBox

HIGH_TRUST_MIN = 0.70
MARGIN_TRUST = 0.10
VOTE_WINDOW = 9
VOTE_NEED = 2
VOTE_NEED_WEAK = 3
VOTE_NEED_SWITCH = 3
IOU_MATCH = 0.15
NEAR_DUP_EDITS = 2


@dataclass(frozen=True)
class VoteSample:
    read: PlateRead
    weight: float


@dataclass
class _Track:
    samples: deque[VoteSample]
    box: PlateBox
    sticky: str = ""


class PlateVoteTracker:
    """Per-stream tracks that fuse OCR reads across time."""

    def __init__(self) -> None:
        self._tracks: dict[str, list[_Track]] = {}
        self._stream_sticky: dict[str, str] = {}

    def consider(self, stream: str, read: PlateRead) -> PlateRead | None:
        track = self._match(stream, read.box)
        if not track.sticky:
            track.sticky = self._stream_sticky.get(stream, "")
        weight = _sample_weight(read)
        track.samples.append(VoteSample(read=read, weight=weight))
        track.box = read.box

        if _shorter_than_known(track, read.text):
            longer = _known_longer(track, read.text)
            for sample in reversed(track.samples):
                if longer and sample.read.text == longer:
                    return self._commit(stream, track, sample.read)
            # Hold the longer sticky; do not publish a truncated majority.
            return None

        switching = bool(track.sticky) and read.text != track.sticky
        if switching:
            upgrade = _compatible_longer(track.sticky, read.text)
            if upgrade == read.text and len(read.text) > len(track.sticky):
                # Longer prefix/suffix of sticky (AKW568→CAKW568): easy upgrade.
                return self._commit(
                    stream,
                    track,
                    _majority_release(track, need=VOTE_NEED),
                )
            if _near_duplicate(track.sticky, read.text) or upgrade is None:
                return self._commit(
                    stream,
                    track,
                    _majority_release(track, need=VOTE_NEED_SWITCH),
                )

        if (
            _high_trust(read)
            and not _has_near_duplicate_competitor(track, read)
            and _agreeing_count(track, read.text) >= VOTE_NEED
        ):
            return self._commit(stream, track, read)

        return self._commit(stream, track, _majority_release(track))

    def _commit(
        self,
        stream: str,
        track: _Track,
        read: PlateRead | None,
    ) -> PlateRead | None:
        if read is None:
            return None
        track.sticky = read.text
        self._stream_sticky[stream] = read.text
        return read

    def _match(self, stream: str, box: PlateBox) -> _Track:
        tracks = self._tracks.setdefault(stream, [])
        best: _Track | None = None
        best_iou = 0.0
        for track in tracks:
            iou = _box_iou(track.box, box)
            if iou > best_iou:
                best_iou = iou
                best = track
        if best is not None and best_iou >= IOU_MATCH:
            return best
        track = _Track(samples=deque(maxlen=VOTE_WINDOW), box=box)
        sticky = self._stream_sticky.get(stream, "")
        if sticky:
            track.sticky = sticky
        tracks.append(track)
        if len(tracks) > 16:
            del tracks[:-16]
        return track


def _high_trust(read: PlateRead) -> bool:
    if read.confidence < HIGH_TRUST_MIN:
        return False
    if read.char_confidences and min(read.char_confidences) < HIGH_TRUST_MIN:
        return False
    if read.char_margins and any(margin < MARGIN_TRUST for margin in read.char_margins):
        return False
    if read.blank_uncertainty >= 0.25:
        return False
    return True


def _agreeing_count(track: _Track, text: str) -> int:
    count = 0
    for sample in track.samples:
        preferred = _compatible_longer(sample.read.text, text)
        if preferred == text or sample.read.text == text:
            count += 1
    return count


def _shorter_than_known(track: _Track, text: str) -> bool:
    """True when sticky or window already has a longer compatible plate."""
    return _known_longer(track, text) is not None


def _known_longer(track: _Track, text: str) -> str | None:
    texts = [sample.read.text for sample in track.samples]
    if track.sticky:
        texts.append(track.sticky)
    best: str | None = None
    for known in texts:
        preferred = _compatible_longer(known, text)
        if preferred is not None and len(preferred) > len(text):
            if best is None or len(preferred) > len(best):
                best = preferred
            continue
        # Single-character drop of a longer sticky/window read (AKW568→AKW58).
        if _is_one_char_shorter(known, text):
            if best is None or len(known) > len(best):
                best = known
    return best


def _is_one_char_shorter(longer: str, shorter: str) -> bool:
    if not longer or not shorter or len(longer) != len(shorter) + 1:
        return False
    for index in range(len(longer)):
        if longer[:index] + longer[index + 1 :] == shorter:
            return True
    return False


def _has_near_duplicate_competitor(track: _Track, read: PlateRead) -> bool:
    texts = [sample.read.text for sample in track.samples]
    if track.sticky:
        texts.append(track.sticky)
    for text in texts:
        if text == read.text:
            continue
        if _near_duplicate(text, read.text):
            return True
    return False


def _near_duplicate(left: str, right: str) -> bool:
    if left == right or not left or not right:
        return False
    if _compatible_longer(left, right) is not None:
        return False
    if len(left) != len(right):
        return False
    return sum(a != b for a, b in zip(left, right, strict=True)) <= NEAR_DUP_EDITS


def _majority_release(
    track: _Track,
    need: int | None = None,
) -> PlateRead | None:
    if not track.samples:
        return None
    if need is None:
        need = VOTE_NEED_WEAK if _weak_quality(track.samples[-1].read) else VOTE_NEED
    if len(track.samples) < need:
        return None

    groups: dict[str, list[VoteSample]] = {}
    for sample in track.samples:
        placed = False
        for key in list(groups):
            preferred = _compatible_longer(key, sample.read.text)
            if preferred is None:
                continue
            bucket = groups.pop(key)
            bucket.append(sample)
            groups.setdefault(preferred, []).extend(bucket)
            placed = True
            break
        if not placed:
            groups.setdefault(sample.read.text, []).append(sample)

    best_text = ""
    best_score = -1.0
    best_read: PlateRead | None = None
    best_count = 0
    for text, bucket in groups.items():
        score = sum(item.weight for item in bucket)
        count = len(bucket)
        read = max(
            bucket,
            key=lambda item: (len(item.read.text), item.read.confidence),
        ).read
        if count < need:
            continue
        better_len = len(text) > len(best_text)
        better_conf = (
            len(text) == len(best_text)
            and read.confidence > (best_read.confidence if best_read else -1.0)
        )
        if score > best_score or (score == best_score and (better_len or better_conf)):
            best_text = text
            best_score = score
            best_read = read
            best_count = count
    if best_read is None or best_count < need:
        return None
    preferred = _compatible_longer(best_read.text, best_text)
    if best_read.text != best_text and preferred == best_text:
        return PlateRead(
            text=best_text,
            confidence=best_read.confidence,
            box=best_read.box,
            char_confidences=best_read.char_confidences,
            char_margins=best_read.char_margins,
            blank_uncertainty=best_read.blank_uncertainty,
            laplacian_var=best_read.laplacian_var,
            brightness=best_read.brightness,
            contrast=best_read.contrast,
            pre_ocr_bgr=best_read.pre_ocr_bgr,
            pre_ocr_width=best_read.pre_ocr_width,
            pre_ocr_height=best_read.pre_ocr_height,
        )
    return best_read


def _compatible_longer(left: str, right: str) -> str | None:
    """Prefer the longer string when one is a prefix or suffix of the other.

    Middle-character insertions (AXKW568 vs AKW568) are not compatible — those
    must compete through the near-duplicate / switch vote path.
    """
    if left == right:
        return left
    if not left or not right:
        return None
    if left.endswith(right) or right.endswith(left):
        return left if len(left) >= len(right) else right
    if left.startswith(right) or right.startswith(left):
        return left if len(left) >= len(right) else right
    return None


def _prefer_longer_suffix(left: str, right: str) -> str | None:
    """Backward-compatible alias used by older tests/callers."""
    return _compatible_longer(left, right)


def _sample_weight(read: PlateRead) -> float:
    return max(0.05, float(read.confidence)) * _quality_weight(read)


def _quality_weight(read: PlateRead) -> float:
    score = 1.0
    if read.laplacian_var > 0.0:
        score *= min(max(read.laplacian_var / 40.0, 0.4), 1.3)
    if read.contrast > 0.0:
        score *= min(max(read.contrast / 25.0, 0.4), 1.3)
    return max(0.3, min(score, 1.5))


def _weak_quality(read: PlateRead) -> bool:
    if read.laplacian_var > 0.0 and read.laplacian_var < 40.0:
        return True
    if read.contrast > 0.0 and read.contrast < 25.0:
        return True
    return False


def _box_iou(left: PlateBox, right: PlateBox) -> float:
    lx2 = left.x + left.width
    ly2 = left.y + left.height
    rx2 = right.x + right.width
    ry2 = right.y + right.height
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(lx2, rx2)
    y2 = min(ly2, ry2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    union = left.width * left.height + right.width * right.height - intersection
    if union <= 0:
        return 0.0
    return intersection / union
