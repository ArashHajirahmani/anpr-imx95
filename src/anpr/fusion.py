"""Evidence fusion for one plate track.

Instead of voting on decoded strings, every frame's CTC step probabilities score a
small pool of candidate texts (the CTC forward likelihood). Candidate scores add up
over a sliding window of frames, so each new frame sharpens the posterior and a
weak or truncated read costs a candidate little instead of casting a lost vote.
A fixed "unknown" hypothesis keeps the posterior honest when no candidate explains
the frames. Pure stdlib.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from anpr.ctc import NEG_INF, beam_candidates, ctc_log_likelihood
from anpr.plates import PLATE_ALPHABET


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


@dataclass
class _Frame:
    probs: list[list[float]]
    weight: float
    lp: dict[str, float] = field(default_factory=dict)


class FusionState:
    def __init__(self, settings: FusionSettings, alphabet: str = PLATE_ALPHABET) -> None:
        self._s = settings
        self._alphabet = alphabet
        self._frames: deque[_Frame] = deque(maxlen=settings.window)
        self._pool: set[str] = set()

    def __len__(self) -> int:
        return len(self._frames)

    def add(
        self, probs: list[list[float]], weight: float = 1.0, extra: tuple[str, ...] = ()
    ) -> None:
        frame = _Frame(probs=probs, weight=max(0.0, weight))
        self._frames.append(frame)
        for text, _ in beam_candidates(probs, self._alphabet, self._s.beam):
            self._pool.add(text)
        self._pool.update(text for text in extra if text)
        if len(self._pool) > self._s.max_pool:
            scores = self._scores()
            keep = sorted(self._pool, key=lambda t: scores[t], reverse=True)[: self._s.max_pool]
            self._pool = set(keep)

    def _frame_lp(self, frame: _Frame, text: str) -> float:
        value = frame.lp.get(text)
        if value is None:
            value = ctc_log_likelihood(frame.probs, text, self._alphabet)
            frame.lp[text] = value
        return max(value, self._s.null_logp) if value != NEG_INF else self._s.null_logp

    def _scores(self) -> dict[str, float]:
        scale = self._s.evidence_weight
        return {
            text: scale * sum(f.weight * self._frame_lp(f, text) for f in self._frames)
            for text in self._pool
        }

    def posterior(self) -> dict[str, float]:
        """Posterior over the pool; the empty-string key is the unknown hypothesis."""
        if not self._frames:
            return {}
        scores = self._scores()
        scores[""] = self._s.evidence_weight * sum(
            f.weight * self._s.null_logp for f in self._frames
        )
        high = max(scores.values())
        exp = {text: math.exp(value - high) for text, value in scores.items()}
        total = sum(exp.values())
        return {text: value / total for text, value in exp.items()}

    def leader(self) -> tuple[str, float]:
        post = self.posterior()
        text = max((t for t in post if t), key=post.__getitem__, default="")
        return (text, post[text]) if text else ("", 0.0)
