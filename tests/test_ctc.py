import math

from anpr.ctc import beam_candidates, ctc_log_likelihood, mean_step_entropy, step_probabilities

ALPHABET = "ABC_"


def _steps(*rows: str) -> list[list[float]]:
    """One-hot-ish steps: each char gets 0.9, the rest share 0.1."""
    out = []
    for char in rows:
        step = [0.1 / 3] * 4
        step[ALPHABET.index(char)] = 0.9
        out.append(step)
    return out


def test_likelihood_prefers_the_decoded_text() -> None:
    probs = _steps("A", "_", "B", "B", "_", "C")
    good = ctc_log_likelihood(probs, "ABC", ALPHABET)
    assert good > ctc_log_likelihood(probs, "AC", ALPHABET)
    assert good > ctc_log_likelihood(probs, "ABBC", ALPHABET)
    assert good > ctc_log_likelihood(probs, "CBA", ALPHABET)


def test_likelihood_handles_repeated_letters_and_bad_input() -> None:
    probs = _steps("A", "_", "A")
    assert ctc_log_likelihood(probs, "AA", ALPHABET) > ctc_log_likelihood(probs, "A", ALPHABET)
    assert ctc_log_likelihood(probs, "Z", ALPHABET) == -math.inf
    assert ctc_log_likelihood([], "A", ALPHABET) == -math.inf


def test_beam_returns_best_text_first_with_alternatives() -> None:
    probs = _steps("A", "_", "B")
    probs[2] = [0.0, 0.5, 0.45, 0.05]  # B vs C is nearly a tie
    beams = beam_candidates(probs, ALPHABET, beam=4)
    assert beams[0][0] == "AB"
    assert "AC" in [text for text, _ in beams]


def test_entropy_is_low_for_crisp_and_high_for_flat_steps() -> None:
    crisp = _steps("A", "B")
    flat = [[0.25] * 4, [0.25] * 4]
    assert mean_step_entropy(crisp) < mean_step_entropy(flat)
    assert mean_step_entropy([]) == 0.0


def test_step_probabilities_softmaxes_raw_logits() -> None:
    out = step_probabilities([[2.0, 0.0, 0.0, 0.0]])
    assert math.isclose(sum(out[0]), 1.0)
