#!/usr/bin/env python3
"""Convert a fast-plate-ocr CCT ONNX model (XS or S) to an imx95 Neutron TFLite model.

Same eIQ packages as `convert_neutron_tflite.py` (eiq-neutron-sdk 3.2.1, eiq-onnx2tflite).

Flow: fix batch=1 -> fold constants (the learned query tokens would otherwise reach
`neutron_compiler` as constant-input FULLY_CONNECTED ops, which it rejects) -> ONNX to
TFLite FP32 -> profile on real plate crops -> PTQ INT8 (SOFTMAX stays float) ->
`neutron_compiler --target imx95`.

Calibration crops come from the calibration frames: the float plate detector finds each
plate, the same ROI refine and resize the service uses produce the 64x128 RGB crop, and
brightness/blur/shift variants top the set up to the minimum count. When `ai-edge-litert`
is installed the quantized (pre-compile) model is compared with the float ONNX on those
crops, so a bad quantization shows up before deployment.

  .venv/bin/python tools/host/convert_neutron_ocr.py \
    --onnx models/cct_xs_v2_global.onnx --detector models/plate_detector.onnx \
    --output models/cct_xs_v2_global.neutron.tflite
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

MIN_CALIBRATION_CROPS = 100
OCR_SHAPE = (64, 128, 3)


def fold_onnx(source: Path, target: Path) -> None:
    """Fix batch=1 and constant-fold with onnxruntime's basic graph optimizer."""
    import onnx
    import onnxruntime as ort
    from onnx.tools import update_model_dims

    model = onnx.load(str(source))
    inputs = {i.name: [1, *OCR_SHAPE] for i in model.graph.input}
    outputs = {
        o.name: [1, *[d.dim_value for d in o.type.tensor_type.shape.dim][1:]]
        for o in model.graph.output
    }
    fixed = target.with_suffix(".b1.onnx")
    onnx.save(update_model_dims.update_inputs_outputs_dims(model, inputs, outputs), str(fixed))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(target)
    ort.InferenceSession(str(fixed), options, providers=["CPUExecutionProvider"])
    strip_input_cast(target)


def strip_input_cast(path: Path) -> None:
    """Make the model input float32 holding 0..255 and drop the leading uint8 Cast.

    With a uint8 input the quantized graph starts with a CAST that the Neutron graph does
    not read correctly. A float input lets the quantizer give the model a native INT8 input.
    """
    import onnx
    from onnx import TensorProto

    model = onnx.load(str(path))
    graph = model.graph
    name = graph.input[0].name
    for node in list(graph.node):
        if node.op_type == "Cast" and node.input[0] == name:
            for other in graph.node:
                for index, tensor in enumerate(other.input):
                    if tensor == node.output[0]:
                        other.input[index] = name
            graph.node.remove(node)
    graph.input[0].type.tensor_type.elem_type = TensorProto.FLOAT
    onnx.save(model, str(path))


def frames_from(images: Path, videos: list[Path], per_video: int):
    """Calibration frames: every image in `images`, plus `per_video` evenly spaced video frames."""
    for path in sorted(
        p for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            yield image
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in np.linspace(0, max(total - 1, 0), per_video).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = capture.read()
            if ok:
                yield image
        capture.release()


def detect_crops(frames, detector: Path) -> list[np.ndarray]:
    """64x128 RGB uint8 plate crops, one per frame the detector fires on."""
    from anpr.plates import resize_rgb_for_ocr
    from anpr.recognize import PlateReader
    from anpr.roi_refine import refine_plate_roi
    from anpr.types import Frame

    ocr_stub = next(Path("models").glob("cct_*_global.onnx"), None)
    if ocr_stub is None:
        raise FileNotFoundError("models/cct_*_global.onnx is needed to build a PlateReader")
    reader = PlateReader(detector, ocr_stub)
    crops: list[np.ndarray] = []
    for image in frames:
        height, width = image.shape[:2]
        detected = reader.detect(Frame("calib", image.tobytes(), 0, width, height))
        for x1, y1, x2, y2, _ in detected.boxes[:1]:
            refined = refine_plate_roi(detected.image, x1, y1, x2, y2, reader._roi_refine)
            rgb = cv2.cvtColor(np.asarray(refined.crop_bgr), cv2.COLOR_BGR2RGB)
            crops.append(np.ascontiguousarray(resize_rgb_for_ocr(rgb), dtype=np.uint8))
    return crops


def augment(crops: list[np.ndarray], minimum: int) -> list[np.ndarray]:
    """Add brightness, contrast, blur and shift variants until there are `minimum` crops."""
    if not crops:
        return crops
    rng = np.random.default_rng(0)
    out = list(crops)
    while len(out) < minimum:
        crop = crops[len(out) % len(crops)].astype(np.float32)
        crop = crop * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
        if rng.random() < 0.5:
            crop = cv2.GaussianBlur(crop, (3, 3), rng.uniform(0.3, 1.2))
        shift = np.float32([[1, 0, rng.integers(-3, 4)], [0, 1, rng.integers(-2, 3)]])
        crop = cv2.warpAffine(
            crop, shift, (OCR_SHAPE[1], OCR_SHAPE[0]), borderMode=cv2.BORDER_REPLICATE
        )
        out.append(np.clip(crop, 0, 255).astype(np.uint8))
    return out


def write_calibration(crops: list[np.ndarray], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index, crop in enumerate(crops):
        crop.astype(np.float32).tofile(directory / f"{index:04d}.bin")


def readable(crops: list[np.ndarray], onnx_path: Path, alphabet: str) -> list[np.ndarray]:
    """Crops the float model reads as plate text (4+ characters).

    Detector hits on frames without a plate give blank reads; calibrating on them would
    spend the activation ranges on background. Falls back to all crops if none read.
    """
    import onnxruntime as ort

    from anpr.plates import decode_characters

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    good = [
        crop
        for crop in crops
        if len(decode_characters(session.run(None, {name: crop[None]})[0][0].tolist(), alphabet)[0])
        >= 4
    ]
    return good or crops


def _run(command: list[str]) -> None:
    # eIQ entry points sit next to the interpreter of the venv this script runs in.
    sibling = Path(sys.executable).parent / command[0]
    if sibling.is_file():
        command = [str(sibling), *command[1:]]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def verify(quantized: Path, onnx_path: Path, crops: list[np.ndarray], alphabet: str) -> None:
    """Compare the pre-compile INT8 model with the float ONNX on the real crops."""
    try:
        import onnxruntime as ort
        from ai_edge_litert.interpreter import Interpreter

        from anpr.plates import decode_characters
    except ImportError:
        print("skip verify: pip install ai-edge-litert to compare against float")
        return
    interpreter = Interpreter(model_path=str(quantized))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    plate = next(o for o in interpreter.get_output_details() if len(o["shape"]) == 3)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    same = 0
    for crop in crops:
        batch = crop[None, ...]
        reference = session.run(None, {name: batch})[0][0]
        scale_in, zero_in = inp["quantization"]
        if inp["dtype"] == np.int8 and scale_in:
            feed = np.clip(np.round(batch / scale_in) + zero_in, -128, 127).astype(np.int8)
        else:
            feed = batch.astype(inp["dtype"])
        interpreter.set_tensor(inp["index"], feed)
        interpreter.invoke()
        got = interpreter.get_tensor(plate["index"])[0].astype(np.float32)
        scale, zero = plate["quantization"]
        if scale:
            got = (got - zero) * scale
        same += (
            decode_characters(reference.tolist(), alphabet)[0]
            == decode_characters(got.tolist(), alphabet)[0]
        )
    print(f"verify: INT8 TFLite reads the float text on {same}/{len(crops)} readable crops")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="float OCR ONNX (XS or S)")
    parser.add_argument("--detector", type=Path, required=True, help="float plate detector ONNX")
    parser.add_argument("--images", type=Path, default=Path("datasets/calibration/images"))
    parser.add_argument(
        "--video",
        type=Path,
        action="append",
        default=[],
        help="video of the real scene; sampled frames add plate crops (repeatable)",
    )
    parser.add_argument("--video-frames", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="imx95")
    parser.add_argument("--percentile", type=float, default=99.0, help="activation clipping")
    parser.add_argument(
        "--keep-float",
        default="SOFTMAX,ADD",
        help="operator types left float. INT8 ADD gives wrong results on the NPU (residuals)",
    )
    parser.add_argument("--alphabet", default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_")
    parser.add_argument("--keep-work", type=Path, default=None)
    args = parser.parse_args()

    if not args.onnx.is_file() or not args.detector.is_file():
        print("missing --onnx or --detector file", file=sys.stderr)
        return 1
    frames = [p for p in args.images.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if len(frames) < 100:
        print(
            f"need >=100 calibration frames in {args.images}, found {len(frames)}", file=sys.stderr
        )
        return 1
    work = Path(tempfile.mkdtemp(prefix="anpr-ocr-")) if args.keep_work is None else args.keep_work
    work.mkdir(parents=True, exist_ok=True)
    folded, fp32, profile = work / "ocr_fold.onnx", work / "ocr_fp32.tflite", work / "ocr.csv"
    ptq = work / "ocr_ptq.tflite"
    try:
        crops = detect_crops(frames_from(args.images, args.video, args.video_frames), args.detector)
        print(f"{len(crops)} real plate crops from {len(frames)} frames")
        if not crops:
            print("the detector found no plates in the calibration frames", file=sys.stderr)
            return 1
        good = readable(crops, args.onnx, args.alphabet)
        print(f"{len(good)} of them read as plate text; calibrating on those")
        everything = augment(good, MIN_CALIBRATION_CROPS)
        write_calibration(everything, work / "calib")
        fold_onnx(args.onnx, folded)
        _run(["onnx2tflite", str(folded), "-o", str(fp32)])
        _run(
            [
                "tflite_profiler",
                "--input",
                str(fp32),
                "--output",
                str(profile),
                "--dataset",
                str(work / "calib"),
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
                "--quantize-inputs",
                "true",
                "--quantize-operator-types-except",
                args.keep_float,
                "--quantization-calibration-method",
                "Percentile",
                "--quantization-percentile-val",
                str(args.percentile),
            ]
        )
        verify(ptq, args.onnx, good, args.alphabet)
        _run(
            [
                "neutron_compiler",
                "--input",
                str(ptq),
                "--output",
                str(args.output),
                "--target",
                args.target,
            ]
        )
    except subprocess.CalledProcessError as exc:
        print(f"conversion failed: {exc}", file=sys.stderr)
        return exc.returncode or 1
    finally:
        if args.keep_work is None:
            shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {args.output}")
    print(f"check the NPU share: python tools/host/neutron_graph_report.py {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
