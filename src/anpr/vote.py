"""Temporal plate voting: windowed, confidence-weighted exact-text consensus.

Every read in a short per-track window votes for its exact text with weight
``confidence**2`` (down-weighted when a character margin or the blank mass says
the read was a guess). Longer strings do not win by length: a hallucinated
leading character (JAKW568, PAKW568) or a dropped one (AKW58) is just another
spelling that must collect votes. A published (sticky) spelling is only
replaced when a challenger has enough reads and clearly more vote weight, so
single-frame flips (K to M, extra prefix) never reach the UI.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from anpr.recognize import PlateRead
from anpr.types import PlateBox

MARGIN_TRUST = 0.10
BLANK_TRUST = 0.25
UNSURE_FACTOR = 0.25
VOTE_WINDOW = 9
VOTE_NEED = 2
VOTE_NEED_SWITCH = 3
VOTE_MIN_WEIGHT = 1.0
SWITCH_RATIO = 1.5
IOU_MATCH = 0.15


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
        track.samples.append(VoteSample(read=read, weight=_sample_weight(read)))
        track.box = read.box

        weights, counts = _tally(track)
        leader = _leader(weights, counts, track.sticky)
        if leader is None or leader != read.text:
            # Hold: the current frame does not belong to the winning spelling.
            return None
        track.sticky = leader
        self._stream_sticky[stream] = leader
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
        track = _Track(
            samples=deque(maxlen=VOTE_WINDOW),
            box=box,
            sticky=self._stream_sticky.get(stream, ""),
        )
        tracks.append(track)
        if len(tracks) > 16:
            del tracks[:-16]
        return track


def _tally(track: _Track) -> tuple[dict[str, float], dict[str, int]]:
    weights: dict[str, float] = {}
    counts: dict[str, int] = {}
    for sample in track.samples:
        text = sample.read.text
        weights[text] = weights.get(text, 0.0) + sample.weight
        counts[text] = counts.get(text, 0) + 1
    return weights, counts


def _leader(weights: dict[str, float], counts: dict[str, int], sticky: str) -> str | None:
    """Spelling that currently owns the track, or None when nothing is proven."""
    challenger = max(weights, key=lambda text: (weights[text], counts[text]), default="")
    if sticky:
        if challenger == sticky or not challenger:
            return sticky
        # A sticky spelling stays until a challenger clearly outvotes it.
        if (
            counts[challenger] >= VOTE_NEED_SWITCH
            and weights[challenger] >= SWITCH_RATIO * weights.get(sticky, 0.0)
            and weights[challenger] >= VOTE_MIN_WEIGHT
        ):
            return challenger
        return sticky
    if counts[challenger] >= VOTE_NEED and weights[challenger] >= VOTE_MIN_WEIGHT:
        return challenger
    return None


def _sample_weight(read: PlateRead) -> float:
    weight = max(0.05, float(read.confidence)) ** 2
    if read.char_margins and min(read.char_margins) < MARGIN_TRUST:
        weight *= UNSURE_FACTOR
    if read.blank_uncertainty >= BLANK_TRUST:
        weight *= UNSURE_FACTOR
    return weight


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
