"""Quantize the plate ONNX models to INT8 with the bench calibration frames.

The board runs these files with ONNX Runtime's NeutronExecutionProvider.
`tools/host/export_int8.py` is the Ultralytics path; these weights are already ONNX.
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from anpr.recognize import _letterbox
from export_int8 import CalibrationError, require_calibration_images

logger = logging.getLogger("anpr.quantize")
DETECTOR_SIZE = 640
OCR_WIDTH = 128
OCR_HEIGHT = 64
# Box coordinates and the class score share this tensor. Quantizing it with one
# scale follows the coordinates and rounds every score to zero.
DETECTOR_OUTPUT_NODE = "/model.22/Concat_3"


class _TensorReader(CalibrationDataReader):
    def __init__(self, tensors: list[np.ndarray], input_name: str) -> None:
        self._tensors = tensors
        self._input_name = input_name
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._tensors):
            return None
        tensor = self._tensors[self._index]
        self._index += 1
        return {self._input_name: tensor}

    def rewind(self) -> None:
        self._index = 0


def _fix_batch(model_path: Path, output_path: Path) -> None:
    model = onnx.load(model_path)
    for value in list(model.graph.input) + list(model.graph.output):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape") or not tensor_type.shape.dim:
            continue
        dim = tensor_type.shape.dim[0]
        if dim.HasField("dim_param"):
            dim.ClearField("dim_param")
            dim.dim_value = 1
    onnx.save(model, output_path)


def _detector_tensors(images: list[Path], input_name: str) -> list[np.ndarray]:
    tensors = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        blob, _scale, _left, _top = _letterbox(image, DETECTOR_SIZE)
        tensors.append(blob)
    if len(tensors) < len(images):
        raise CalibrationError("some calibration frames could not be read")
    logger.info("detector calibration tensors: %d for input %s", len(tensors), input_name)
    return tensors


def _ocr_tensors(images: list[Path], detector: Path) -> list[np.ndarray]:
    session = ort.InferenceSession(str(detector), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    crops: list[np.ndarray] = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        blob, scale, left, top = _letterbox(image, DETECTOR_SIZE)
        raw = session.run(None, {input_name: blob})[0]
        prediction = raw[0].T
        order = np.argsort(-prediction[:, 4])
        chosen = None
        for index in order[:5]:
            if float(prediction[index, 4]) < 0.25:
                break
            cx, cy, width, height, _confidence = prediction[index]
            x1 = int(max(0, (float(cx) - float(width) / 2 - left) / scale))
            y1 = int(max(0, (float(cy) - float(height) / 2 - top) / scale))
            x2 = int(min(image.shape[1], (float(cx) + float(width) / 2 - left) / scale))
            y2 = int(min(image.shape[0], (float(cy) + float(height) / 2 - top) / scale))
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                chosen = image[y1:y2, x1:x2]
                break
        if chosen is None:
            chosen = image
        rgb = cv2.cvtColor(chosen, cv2.COLOR_BGR2RGB)
        try:
            from anpr.plates import resize_rgb_for_ocr

            resized = resize_rgb_for_ocr(rgb)
        except ImportError:
            resized = cv2.resize(rgb, (OCR_WIDTH, OCR_HEIGHT), interpolation=cv2.INTER_AREA)
        crops.append(np.ascontiguousarray(resized[None, ...]))
    logger.info("ocr calibration tensors: %d", len(crops))
    return crops


def _quantize(
    model_path: Path,
    output_path: Path,
    tensors: list[np.ndarray],
    input_name: str,
    nodes_to_exclude: list[str] | None = None,
) -> None:
    fixed = output_path.with_suffix(".fixed.onnx")
    _fix_batch(model_path, fixed)
    quantize_static(
        str(fixed),
        str(output_path),
        _TensorReader(tensors, input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude,
    )
    fixed.unlink(missing_ok=True)
    logger.info("wrote %s", output_path)


def _confidence_channel(raw: np.ndarray) -> np.ndarray:
    prediction = raw[0]
    if prediction.shape[0] < prediction.shape[-1]:
        return prediction[4]
    return prediction[:, 4]


def _check_detector_confidence(
    float_model: Path,
    int8_model: Path,
    images: list[Path],
) -> None:
    """Fail when INT8 drops a real plate score to zero."""
    float_session = ort.InferenceSession(str(float_model), providers=["CPUExecutionProvider"])
    int8_session = ort.InferenceSession(str(int8_model), providers=["CPUExecutionProvider"])
    float_input = float_session.get_inputs()[0].name
    int8_input = int8_session.get_inputs()[0].name
    checked = 0
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            continue
        blob, _scale, _left, _top = _letterbox(image, DETECTOR_SIZE)
        float_raw = float_session.run(None, {float_input: blob})[0]
        int8_raw = int8_session.run(None, {int8_input: blob})[0]
        float_max = float(_confidence_channel(float_raw).max())
        int8_max = float(_confidence_channel(int8_raw).max())
        if float_max < 0.25:
            continue
        checked += 1
        if int8_max < 0.05:
            raise CalibrationError(
                f"{path.name}: float confidence {float_max:.3f} became {int8_max:.3f} after INT8"
            )
        if checked >= 3:
            break
    if checked == 0:
        raise CalibrationError("calibration frames have no detector score above 0.25")
    logger.info("INT8 detector kept the confidence channel on %d frames", checked)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quantize plate ONNX models to INT8")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        images = require_calibration_images(args.images)
    except CalibrationError as exc:
        logger.error("%s", exc)
        return 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detector_out = args.output_dir / "plate_detector.int8.onnx"
    ocr_out = args.output_dir / "cct_xs_v2_global.int8.onnx"
    detector_session = ort.InferenceSession(str(args.detector), providers=["CPUExecutionProvider"])
    detector_input = detector_session.get_inputs()[0].name
    _quantize(
        args.detector,
        detector_out,
        _detector_tensors(images, detector_input),
        detector_input,
        nodes_to_exclude=[DETECTOR_OUTPUT_NODE],
    )
    try:
        _check_detector_confidence(args.detector, detector_out, images)
    except CalibrationError as exc:
        logger.error("%s", exc)
        return 1
    ocr_session = ort.InferenceSession(str(args.ocr), providers=["CPUExecutionProvider"])
    ocr_input = ocr_session.get_inputs()[0].name
    _quantize(args.ocr, ocr_out, _ocr_tensors(images, args.detector), ocr_input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
