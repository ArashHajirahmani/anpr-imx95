from pathlib import Path

import pytest

from export_int8 import (
    MIN_CALIBRATION_IMAGES,
    CalibrationError,
    build_export_command,
    render_data_yaml,
    require_calibration_images,
)


def test_rejects_fewer_than_one_hundred_frames(tmp_path: Path) -> None:
    for index in range(MIN_CALIBRATION_IMAGES - 1):
        (tmp_path / f"frame_{index:03d}.jpg").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    with pytest.raises(CalibrationError, match="99 calibration frames"):
        require_calibration_images(tmp_path)


def test_accepts_one_hundred_frames(tmp_path: Path) -> None:
    for index in range(MIN_CALIBRATION_IMAGES):
        suffix = ".png" if index % 2 == 0 else ".JPEG"
        (tmp_path / f"frame_{index:03d}{suffix}").write_bytes(b"")
    images = require_calibration_images(tmp_path)
    assert len(images) == MIN_CALIBRATION_IMAGES


def test_missing_directory_counts_as_empty(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="0 calibration frames"):
        require_calibration_images(tmp_path / "images")


def test_export_command_passes_calibration_yaml(tmp_path: Path) -> None:
    weights = tmp_path / "plate.pt"
    data_yaml = tmp_path / "calibration.yaml"
    command = build_export_command(weights, data_yaml)
    assert command[:3] == ["yolo", "export", f"model={weights}"]
    assert "format=tflite" in command
    assert "int8=True" in command
    assert f"data={data_yaml}" in command


def test_data_yaml_points_at_the_image_directory(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    text = render_data_yaml(image_dir)
    assert f"path: {image_dir.resolve().parent}" in text
    assert "train: images" in text
    assert "val: images" in text
