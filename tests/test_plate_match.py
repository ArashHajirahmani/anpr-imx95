from anpr.plate_match import clearly_better, merge_text, same_identity


def test_partial_forms_share_identity() -> None:
    assert same_identity("AKW568", "CAKW568")
    assert same_identity("AKW568", "AKW58")
    assert same_identity("AKW568", "AMW568")
    assert same_identity("AXKW568", "CAKW568")
    assert not same_identity("AKW568", "FP1575")


def test_merge_prefers_longer_then_higher_confidence() -> None:
    text, confidence = merge_text("AKW568", 0.9, "CAKW568", 0.8)
    assert text == "CAKW568"
    assert confidence == 0.9
    text, confidence = merge_text("CAKW568", 0.8, "AKW568", 0.99)
    assert text == "CAKW568"
    assert confidence == 0.99


def test_merge_prefers_higher_conf_over_garbage_spelling() -> None:
    text, confidence = merge_text(
        "AXKW568",
        0.77,
        "CAKW568",
        0.94,
        aliases=("AXKW568", "AXKW568", "AKW568"),
        best_by_text={"AXKW568": 0.77, "AKW568": 0.9, "CAKW568": 0.94},
    )
    assert text == "CAKW568"
    assert confidence == 0.94
    # Lower-conf garbage must not overwrite the better spelling.
    text, confidence = merge_text(
        "CAKW568",
        0.94,
        "AXKW568",
        0.77,
        aliases=("CAKW568", "AXKW568"),
        best_by_text={"CAKW568": 0.94, "AXKW568": 0.77},
    )
    assert text == "CAKW568"
    assert confidence == 0.94


def test_clearly_better_requires_margin_or_quality() -> None:
    assert clearly_better(0.77, 0.94)
    assert not clearly_better(0.94, 0.77)
    assert clearly_better(0.90, 0.91, current_quality=10.0, new_quality=80.0)
    assert not clearly_better(0.90, 0.91, current_quality=80.0, new_quality=10.0)
