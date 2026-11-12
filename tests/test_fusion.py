from anpr.fusion import FusionSettings, FusionState

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"


def _frame(text: str, sure: float = 0.999, alt: dict[int, tuple[str, float]] | None = None):
    """CTC steps for ``text`` with a blank between letters. ``alt`` gives a runner-up char."""
    steps = []
    for index, char in enumerate(text):
        for blank in (False, True):
            step = [0.0] * len(ALPHABET)
            if blank:
                step[ALPHABET.index("_")] = 1.0
            else:
                step[ALPHABET.index(char)] = sure
                rest = 1.0 - sure
                if alt and index in alt:
                    other, share = alt[index]
                    step[ALPHABET.index(other)] = rest * share
                    step[ALPHABET.index("_")] = rest * (1 - share)
                else:
                    step[ALPHABET.index("_")] = rest
            steps.append(step)
    return steps


def test_clear_frames_release_without_waiting_for_a_window() -> None:
    state = FusionState(FusionSettings())
    for _ in range(2):
        state.add(_frame("AKW568"))
    text, posterior = state.leader()
    assert text == "AKW568"
    assert posterior >= 0.9


def test_runner_up_evidence_beats_a_wrong_majority() -> None:
    # Most frames lean to AX but always keep CA as a close runner-up, one frame prefers CA.
    state = FusionState(FusionSettings(evidence_weight=1.0))
    for _ in range(3):
        state.add(_frame("AXKW568", sure=0.55, alt={0: ("C", 0.9), 1: ("A", 0.9)}))
    state.add(_frame("CAKW568", sure=0.9))
    state.add(_frame("CAKW568", sure=0.9))
    text, _ = state.leader()
    assert text == "CAKW568"


def test_truncated_frame_does_not_flip_the_plate() -> None:
    state = FusionState(FusionSettings())
    for _ in range(4):
        state.add(_frame("CAKW568"))
    state.add(_frame("AKW58"), 0.3)
    assert state.leader()[0] == "CAKW568"


def test_unexplained_frames_leave_the_posterior_on_unknown() -> None:
    state = FusionState(FusionSettings())
    flat = [[1.0 / len(ALPHABET)] * len(ALPHABET) for _ in range(8)]
    state.add(flat)
    assert state.leader()[1] < 0.9


def test_window_lets_a_new_plate_replace_the_old() -> None:
    state = FusionState(FusionSettings(window=6))
    for _ in range(6):
        state.add(_frame("AAA111"))
    for _ in range(6):
        state.add(_frame("BBB222"))
    assert state.leader()[0] == "BBB222"
