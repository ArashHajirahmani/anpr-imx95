"""Export an INT8 TFLite model with a required calibration image set.

The script refuses to call Ultralytics until the calibration directory holds
at least 100 real frames. A COCO sample is not accepted by this check; the
caller has to point `--images` at the gate or parking frames.
"""

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path

MIN_CALIBRATION_IMAGES = 100
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
logger = logging.getLogger("anpr.export")


class CalibrationError(Exception):
    """The calibration directory does not contain enough frames."""


def default_images_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "datasets" / "calibration" / "images"


def list_calibration_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def require_calibration_images(
    directory: Path,
    minimum: int = MIN_CALIBRATION_IMAGES,
) -> list[Path]:
    images = list_calibration_images(directory)
    if len(images) < minimum:
        raise CalibrationError(
            f"{directory} has {len(images)} calibration frames; {minimum} are required"
        )
    return images


def render_data_yaml(image_dir: Path) -> str:
    root = image_dir.resolve().parent
    folder = image_dir.name
    return f"path: {root}\ntrain: {folder}\nval: {folder}\nnames:\n  0: plate\n"


def build_export_command(weights: Path, data_yaml: Path) -> list[str]:
    return [
        "yolo",
        "export",
        f"model={weights}",
        "format=tflite",
        "int8=True",
        f"data={data_yaml}",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export INT8 TFLite with a calibration set")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--images", type=Path, default=default_images_dir())
    parser.add_argument("--minimum", type=int, default=MIN_CALIBRATION_IMAGES)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        images = require_calibration_images(args.images, args.minimum)
    except CalibrationError as exc:
        logger.error("%s", exc)
        return 1
    if not args.weights.is_file():
        logger.error("weights file not found: %s", args.weights)
        return 1
    logger.info("calibration frames: %d", len(images))
    with tempfile.TemporaryDirectory() as tmp:
        data_yaml = Path(tmp) / "calibration.yaml"
        data_yaml.write_text(render_data_yaml(args.images), encoding="utf-8")
        command = build_export_command(args.weights, data_yaml)
        logger.info("running %s", " ".join(command))
        completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
