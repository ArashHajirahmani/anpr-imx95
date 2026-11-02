from anpr.plates import accept_plate, decode_characters, nms, yolo_candidates


def test_decode_strips_padding() -> None:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_"
    pad = alphabet.index("_")
    digit = alphabet.index("A")
    steps = [[0.0] * len(alphabet) for _ in range(3)]
    steps[0][digit] = 5
    steps[1][pad] = 5
    steps[2][pad] = 4
    text, confidence = decode_characters(steps)
    assert text == "A"
    assert confidence > 0.5


def test_nms_keeps_the_higher_confidence_box() -> None:
    boxes = [
        (0.0, 0.0, 10.0, 10.0, 0.4),
        (1.0, 1.0, 11.0, 11.0, 0.9),
        (40.0, 40.0, 50.0, 50.0, 0.6),
    ]
    kept = nms(boxes, iou_threshold=0.45)
    assert len(kept) == 2
    assert kept[0][4] == 0.9
    assert yolo_candidates([(1, 1, 2, 2, 0.2), (1, 1, 2, 2, 0.8)], 0.5) == [(1, 1, 2, 2, 0.8)]


def test_confidence_floor_matches_accurate_reads() -> None:
    assert accept_plate("777W455", 0.07)
    assert accept_plate("777W455", 0.069) is False
    assert accept_plate("", 0.9) is False
