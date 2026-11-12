"""CTC scoring helpers: per-step probabilities, candidate texts, and text likelihoods.

Pure stdlib so the fusion logic runs on a bare dev host. ``steps`` is the OCR output
for one frame, one probability vector per time step.
"""

from __future__ import annotations

import math

from anpr.plates import PLATE_PAD, _looks_like_probabilities, _softmax_probs

NEG_INF = float("-inf")
PROB_FLOOR = 1e-6


def step_probabilities(steps: list[list[float]]) -> list[list[float]]:
    """Normalise each step to probabilities (the CCT export already emits them)."""
    out: list[list[float]] = []
    for values in steps:
        if _looks_like_probabilities(values):
            out.append([float(v) for v in values])
        else:
            out.append(_softmax_probs(values))
    return out


def _logsumexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    high = max(a, b)
    return high + math.log(math.exp(a - high) + math.exp(b - high))


def ctc_log_likelihood(
    probs: list[list[float]],
    text: str,
    alphabet: str,
    pad: str = PLATE_PAD,
) -> float:
    """log P(text | probs) summed over every CTC alignment (forward algorithm)."""
    blank = alphabet.find(pad)
    labels: list[int] = []
    for char in text:
        index = alphabet.find(char)
        if index < 0 or index == blank:
            return NEG_INF
        labels.append(index)
    if not probs or not labels:
        return NEG_INF
    ext = [blank]
    for label in labels:
        ext.extend((label, blank))
    size = len(ext)

    def lp(step: list[float], cls: int) -> float:
        return math.log(max(step[cls], PROB_FLOOR))

    alpha = [NEG_INF] * size
    first = probs[0]
    alpha[0] = lp(first, ext[0])
    if size > 1:
        alpha[1] = lp(first, ext[1])
    for step in probs[1:]:
        nxt = [NEG_INF] * size
        for s in range(size):
            total = alpha[s]
            if s >= 1:
                total = _logsumexp(total, alpha[s - 1])
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                total = _logsumexp(total, alpha[s - 2])
            if total != NEG_INF:
                nxt[s] = total + lp(step, ext[s])
        alpha = nxt
    return _logsumexp(alpha[-1], alpha[-2]) if size > 1 else alpha[-1]


def beam_candidates(
    probs: list[list[float]],
    alphabet: str,
    beam: int = 8,
    prune: float = 0.001,
    pad: str = PLATE_PAD,
) -> list[tuple[str, float]]:
    """CTC prefix beam search. Returns up to ``beam`` (text, log P) pairs, best first."""
    blank = alphabet.find(pad)
    # prefix -> (log p ending in blank, log p ending in non-blank)
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}
    for step in probs:
        nxt: dict[tuple[int, ...], tuple[float, float]] = {}

        def add(prefix: tuple[int, ...], pb: float, pnb: float) -> None:
            ob, onb = nxt.get(prefix, (NEG_INF, NEG_INF))
            nxt[prefix] = (_logsumexp(ob, pb), _logsumexp(onb, pnb))

        for prefix, (pb, pnb) in beams.items():
            total = _logsumexp(pb, pnb)
            for cls, p in enumerate(step):
                if cls >= len(alphabet) or p < prune:
                    continue
                lpc = math.log(max(p, PROB_FLOOR))
                if cls == blank:
                    add(prefix, total + lpc, NEG_INF)
                    continue
                last = prefix[-1] if prefix else -1
                extended = prefix + (cls,)
                if cls == last:
                    add(extended, NEG_INF, pb + lpc)
                    add(prefix, NEG_INF, pnb + lpc)
                else:
                    add(extended, NEG_INF, total + lpc)
        ranked = sorted(nxt.items(), key=lambda kv: _logsumexp(*kv[1]), reverse=True)
        beams = dict(ranked[:beam])
    out = [
        ("".join(alphabet[i] for i in prefix), _logsumexp(pb, pnb))
        for prefix, (pb, pnb) in beams.items()
        if prefix
    ]
    out.sort(key=lambda item: item[1], reverse=True)
    return out


def mean_step_entropy(probs: list[list[float]]) -> float:
    """Mean per-step entropy (nats). Low for a crisp read, high for a guess."""
    if not probs:
        return 0.0
    total = 0.0
    for step in probs:
        total -= sum(p * math.log(p) for p in step if p > 1e-9)
    return total / len(probs)
