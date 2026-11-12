#!/usr/bin/env python3
"""Convert a float ONNX detector to an imx95 Neutron TFLite model.

Requires host packages from https://eiq.nxp.com/repository/ matching the board
Neutron driver (LF 6.18 / wrynose uses eiq-neutron-sdk 3.2.1):

  pip install --extra-index-url https://eiq.nxp.com/repository/ \
    'eiq-neutron-sdk==3.2.1' eiq-onnx2tflite

Flow: ONNX → TFLite FP32 → profile (≥100 calibration frames) → PTQ INT8
→ neutron-compiler --target imx95. DIV/SOFTMAX stay float so the CPU
fallback path can invoke them. The head is split into separate `boxes` and `scores`
outputs first so INT8 keeps each tensor's own scale (see onnx_split_head.py).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from onnx_split_head import SplitError, split_detection_head

MIN_CALIBRATION_IMAGES = 100
DETECTOR_SIZE = 640


def _count_images(directory: Path) -> list[Path]:
    paths = sorted(
        [
            *directory.glob("*.jpg"),
            *directory.glob("*.jpeg"),
            *directory.glob("*.png"),
            *directory.glob("*.JPG"),
            *directory.glob("*.JPEG"),
            *directory.glob("*.PNG"),
        ]
    )
    return paths


def _letterbox_nchw(image_bgr: np.ndarray, size: int = DETECTOR_SIZE) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    scale = min(size / height, size / width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image_bgr, (resized_width, resized_height))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, ...])


def _serialize_calibration(images: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(images):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read calibration image {path}")
        _letterbox_nchw(image).tofile(out_dir / f"{index:04d}.bin")


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="float ONNX detector")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("datasets/calibration/images"),
        help="calibration JPEG/PNG directory (≥100 frames)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output Neutron TFLite path (e.g. models/plate_detector.neutron.tflite)",
    )
    parser.add_argument("--target", default="imx95", help="neutron-compiler target")
    parser.add_argument(
        "--keep-work",
        type=Path,
        default=None,
        help="optional directory to keep intermediate TFLite/profile files",
    )
    args = parser.parse_args()

    if not args.onnx.is_file():
        print(f"missing onnx model: {args.onnx}", file=sys.stderr)
        return 1
    images = _count_images(args.images)
    if len(images) < MIN_CALIBRATION_IMAGES:
        print(
            f"need ≥{MIN_CALIBRATION_IMAGES} calibration images in {args.images}, "
            f"found {len(images)}",
            file=sys.stderr,
        )
        return 1

    work = (
        Path(tempfile.mkdtemp(prefix="anpr-neutron-")) if args.keep_work is None else args.keep_work
    )
    work.mkdir(parents=True, exist_ok=True)
    split_onnx = work / "model_split.onnx"
    fp32 = work / "model_fp32.tflite"
    profile = work / "model.profile.csv"
    ptq = work / "model_ptq.tflite"
    calib = work / "calib_nchw"
    neutron = args.output

    try:
        _serialize_calibration(images, calib)
        # Boxes (pixels) and scores (0..1) must not share one INT8 scale.
        split_detection_head(args.onnx, split_onnx)
        _run(["onnx2tflite", str(split_onnx), "-o", str(fp32)])
        _run(
            [
                "tflite_profiler",
                "--input",
                str(fp32),
                "--output",
                str(profile),
                "--dataset",
                str(calib),
            ]
        )
        _run(
            [
                "tflite_quantizer",
                "--input",
                str(fp32),
                "--profile",
                str(profile),
                "--output",
                str(ptq),
                "--quantize-operator-types-except",
                "DIV,SOFTMAX",
            ]
        )
        _run(
            [
                "neutron_compiler",
                "--input",
                str(ptq),
                "--output",
                str(neutron),
                "--target",
                args.target,
            ]
        )
    except SplitError as exc:
        print(f"cannot split detector outputs: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"conversion failed: {exc}", file=sys.stderr)
        return exc.returncode or 1
    finally:
        if args.keep_work is None:
            shutil.rmtree(work, ignore_errors=True)

    print(f"wrote {neutron}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
