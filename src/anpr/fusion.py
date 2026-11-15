"""Evidence fusion for one plate track.

Instead of voting on decoded strings, every frame's per-slot probabilities score a
small pool of candidate texts (the slot likelihood). Candidate scores add up
over a sliding window of frames, so each new frame sharpens the posterior and a
weak or truncated read costs a candidate little instead of casting a lost vote.
A fixed "unknown" hypothesis keeps the posterior honest when no candidate explains
the frames. Pure stdlib.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from anpr.plates import PLATE_ALPHABET
from anpr.slots import NEG_INF, beam_candidates, slot_log_likelihood


@dataclass(frozen=True)
class FusionSettings:
    window: int = 16
    beam: int = 5
    max_pool: int = 10
    # Scales summed log-likelihood. Below 1 because consecutive frames are correlated.
    evidence_weight: float = 0.3
    # Per-frame log-likelihood of the "unknown" hypothesis; also the floor for any text.
    null_logp: float = -8.0
    release_posterior: float = 0.90
    # Frame weight is 1 / (1 + quality_slope * mean step entropy): uncertain crops count less.
    quality_slope: float = 4.0
    # Tiered voting. A frame whose quality (detector score^det_weight_power times the
    # entropy factor) reaches high_quality is kept in its own window, so a flood of weak
    # frames cannot push it out. With high_min_frames of them the text is decided from
    # that tier alone. 0 disables tiers.
    high_quality: float = 0.0
    high_min_frames: int = 3
    det_weight_power: float = 1.0
    # High-tier frames expire after this many frames were added to the track.
    high_ttl_frames: int = 120


@dataclass
class _Frame:
    probs: list[list[float]]
    weight: float
    quality: float = 0.0
    seq: int = 0
    lp: dict[str, float] = field(default_factory=dict)


class FusionState:
    def __init__(self, settings: FusionSettings, alphabet: str = PLATE_ALPHABET) -> None:
        self._s = settings
        self._alphabet = alphabet
        self._frames: deque[_Frame] = deque(maxlen=settings.window)
        self._high: deque[_Frame] = deque(maxlen=settings.window)
        self._pool: set[str] = set()
        self._seq = 0

    def __len__(self) -> int:
        return len(self._frames)

    def add(
        self,
        probs: list[list[float]],
        weight: float = 1.0,
        extra: tuple[str, ...] = (),
        quality: float | None = None,
    ) -> None:
        self._seq += 1
        weight = max(0.0, weight)
        frame = _Frame(
            probs=probs,
            weight=weight,
            quality=weight if quality is None else quality,
            seq=self._seq,
        )
        self._frames.append(frame)
        if self._s.high_quality > 0 and frame.quality >= self._s.high_quality:
            self._high.append(frame)
        for text, _ in beam_candidates(probs, self._alphabet, self._s.beam):
            self._pool.add(text)
        self._pool.update(text for text in extra if text)
        if len(self._pool) > self._s.max_pool:
            self._prune()

    def _prune(self) -> None:
        """Keep the best texts of the whole window and of the high tier."""
        limit = self._s.max_pool
        keep: set[str] = set()
        tiers = [self._scores(self._frames)]
        if self._high:
            tiers.append(self._scores(self._high))
        share = max(1, limit // len(tiers))
        for scores in tiers:
            ranked = sorted(self._pool, key=lambda t: scores[t], reverse=True)
            keep.update(ranked[:share])
        for text in sorted(self._pool, key=tiers[0].__getitem__, reverse=True):
            if len(keep) >= limit:
                break
            keep.add(text)
        self._pool = keep

    def _frame_lp(self, frame: _Frame, text: str) -> float:
        value = frame.lp.get(text)
        if value is None:
            value = slot_log_likelihood(frame.probs, text, self._alphabet)
            frame.lp[text] = value
        return max(value, self._s.null_logp) if value != NEG_INF else self._s.null_logp

    def _scores(self, frames: object) -> dict[str, float]:
        scale = self._s.evidence_weight
        return {
            text: scale * sum(f.weight * self._frame_lp(f, text) for f in frames)
            for text in self._pool
        }

    def _live_high(self) -> list[_Frame]:
        oldest = self._seq - self._s.high_ttl_frames
        return [f for f in self._high if f.seq > oldest]

    def posterior(self, frames: object = None) -> dict[str, float]:
        """Posterior over the pool; the empty-string key is the unknown hypothesis."""
        frames = list(self._frames) if frames is None else list(frames)
        if not frames:
            return {}
        scores = self._scores(frames)
        scores[""] = self._s.evidence_weight * sum(f.weight * self._s.null_logp for f in frames)
        high = max(scores.values())
        exp = {text: math.exp(value - high) for text, value in scores.items()}
        total = sum(exp.values())
        return {text: value / total for text, value in exp.items()}

    def leader(self) -> tuple[str, float]:
        """Best text and its posterior, from the high tier when it has enough frames."""
        high = self._live_high()
        use_high = self._s.high_quality > 0 and len(high) >= self._s.high_min_frames
        post = self.posterior(high if use_high else None)
        text = max((t for t in post if t), key=post.__getitem__, default="")
        return (text, post[text]) if text else ("", 0.0)
