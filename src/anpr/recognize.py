"""Plate detector and OCR backends.

`cpu` uses float ONNX models on ORT CPU.
`npu` uses a Neutron-compiled TFLite detector (`*.tflite`) via
`libneutron_delegate.so`, and INT8 ONNX OCR on ORT CPU.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from anpr.ctc import step_probabilities
from anpr.infer import probe_delegates
from anpr.plates import (
    PLATE_ALPHABET,
    center_to_corners,
    decode_characters,
    nms,
    resize_rgb_for_ocr,
)
from anpr.quality import (
    EnhancerSettings,
    QualitySettings,
    crop_quality,
    enhance_crop,
    log_quality_reject,
)
from anpr.roi_refine import RoiRefineSettings, refine_plate_roi
from anpr.types import Frame, PlateBox

logger = logging.getLogger("anpr.recognize")
DETECTOR_SIZE = 640
DEFAULT_NEUTRON_DELEGATE = Path("/usr/lib/libneutron_delegate.so")


@dataclass(frozen=True)
class PlateRead:
    text: str
    confidence: float
    box: PlateBox
    char_confidences: tuple[float, ...] = ()
    char_margins: tuple[float, ...] = ()
    blank_uncertainty: float = 0.0
    laplacian_var: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    # BGR uint8 bytes of the crop after enhance_crop (identity when enhancer is off).
    # History JPEGs use this so the UI shows the exact image that entered OCR resize.
    pre_ocr_bgr: bytes | None = None
    pre_ocr_width: int = 0
    pre_ocr_height: int = 0
    # CTC step probabilities of the OCR pass (steps x classes) for evidence fusion.
    step_probs: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class DetectedFrame:
    """Output of the detector stage; the OCR stage turns it into plate reads."""

    frame: Frame
    image: object
    boxes: tuple[tuple[int, int, int, int, float], ...]
    runtime: str
    det_ms: float
    cpu_s: float


@dataclass(frozen=True)
class OcrOutcome:
    reads: list["PlateRead"]
    ocr_ms: float
    cpu_s: float


@dataclass(frozen=True)
class InferenceStats:
    runtime: str
    det_ms: float
    ocr_ms: float
    cpu_percent: float
    rss_mb: float


class PlateReader:
    def __init__(
        self,
        detector_path: Path,
        ocr_path: Path,
        alphabet: str = PLATE_ALPHABET,
        threshold: float = 0.25,
        runtime: str = "cpu",
        quality: QualitySettings | None = None,
        enhancer: EnhancerSettings | None = None,
        roi_refine: RoiRefineSettings | None = None,
        delegate_path: Path | None = None,
        ocr_threads: int = 2,
    ) -> None:
        import onnxruntime as ort

        if runtime not in {"cpu", "npu"}:
            raise ValueError(f"runtime must be cpu or npu, got {runtime}")
        if not detector_path.is_file():
            raise FileNotFoundError(detector_path)
        if not ocr_path.is_file():
            raise FileNotFoundError(ocr_path)
        self._alphabet = alphabet or PLATE_ALPHABET
        self._threshold = threshold
        self._runtime = runtime
        self._quality = quality or QualitySettings()
        self._enhancer = enhancer or EnhancerSettings()
        self._roi_refine = roi_refine or RoiRefineSettings.disabled()
        self._detector_backend = "onnx"
        self._tflite = None
        self._detector = None
        self._detector_input = ""
        self._detector_scale = 1.0
        self._detector_zero_point = 0
        self._detector_outputs: list[tuple[int, float, int]] = []
        self._detector_u8 = False

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        ocr_providers: list[str] = ["CPUExecutionProvider"]
        if runtime == "npu" and detector_path.suffix.lower() == ".tflite":
            self._detector_backend = "tflite"
            self._load_tflite_detector(detector_path, delegate_path)
        else:
            providers: list[str | tuple[str, dict[str, bool]]]
            if runtime == "npu":
                # Plain INT8 ONNX without NeutronGraph stays on CPU; keep EP listed
                # for older models that already embed NeutronGraph custom ops.
                providers = [
                    ("NeutronExecutionProvider", {"neutron_op_only": True}),
                    "CPUExecutionProvider",
                ]
            else:
                providers = ["CPUExecutionProvider"]
            self._detector = ort.InferenceSession(
                str(detector_path),
                sess_options=options,
                providers=providers,
            )
            self._detector_input = self._detector.get_inputs()[0].name
        # InferenceSession.run is thread-safe: staged OCR workers share this session,
        # each run using `ocr_threads` intra-op threads (1 when several workers run).
        ocr_options = ort.SessionOptions()
        ocr_options.intra_op_num_threads = max(1, int(ocr_threads))
        self._ocr = ort.InferenceSession(
            str(ocr_path),
            sess_options=ocr_options,
            providers=ocr_providers,
        )
        self._ocr_input = self._ocr.get_inputs()[0].name
        logger.info(
            "plate reader runtime=%s detector_backend=%s detector=%s ocr=%s",
            runtime,
            self._detector_backend,
            detector_path,
            ocr_path,
        )

    def _load_tflite_detector(self, detector_path: Path, delegate_path: Path | None) -> None:
        import numpy as np
        from tflite_runtime.interpreter import Interpreter, load_delegate

        path = delegate_path
        if path is None or not path.is_file():
            probe = probe_delegates()
            path = probe.selected or DEFAULT_NEUTRON_DELEGATE
        if not path.is_file():
            raise FileNotFoundError(f"Neutron TFLite delegate not found: {path}")
        delegate = load_delegate(str(path))
        interpreter = Interpreter(
            model_path=str(detector_path),
            experimental_delegates=[delegate],
        )
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        outputs = interpreter.get_output_details()
        self._tflite = interpreter
        self._detector_input_index = int(inp["index"])
        self._detector_input_dtype = inp["dtype"]
        scale, zero_point = inp.get("quantization", (0.0, 0))
        self._detector_scale = float(scale) if scale else 1.0
        self._detector_zero_point = int(zero_point)
        # int8 input with scale 1/255 and zero point -128 is exactly `pixel - 128`, so the
        # detector can be fed from uint8 pixels without a float32 round trip.
        self._detector_u8 = bool(
            self._detector_input_dtype == np.int8
            and abs(self._detector_scale - 1 / 255) < 1e-6
            and self._detector_zero_point == -128
        )
        # Boxes (pixels) and scores (0..1) are separate outputs so each keeps its own
        # INT8 scale; a single [1, 5, N] tensor would quantize every score to 0.
        self._detector_outputs = []
        for out in outputs:
            out_scale, out_zp = out.get("quantization", (0.0, 0))
            self._detector_outputs.append(
                (int(out["index"]), float(out_scale) if out_scale else 1.0, int(out_zp))
            )
        logger.info(
            "tflite neutron detector delegate=%s input_dtype=%s outputs=%d",
            path,
            getattr(self._detector_input_dtype, "__name__", self._detector_input_dtype),
            len(self._detector_outputs),
        )

    def _detect(self, blob: object) -> object:
        import numpy as np

        if self._detector_backend == "tflite":
            assert self._tflite is not None
            feed = blob
            if blob.dtype == np.uint8:
                feed = np.bitwise_xor(blob, 128).view(np.int8)  # pixel - 128
            elif self._detector_input_dtype != np.float32:
                quant = np.round(blob / self._detector_scale) + self._detector_zero_point
                feed = np.clip(quant, -128, 127).astype(self._detector_input_dtype)
            self._tflite.set_tensor(self._detector_input_index, feed)
            self._tflite.invoke()
            tensors = []
            for index, scale, zero_point in self._detector_outputs:
                tensor = self._tflite.get_tensor(index)
                if tensor.dtype != np.float32:
                    tensor = (tensor.astype(np.float32) - float(zero_point)) * scale
                tensors.append(tensor)
            return join_detector_outputs(tensors, self._threshold)
        assert self._detector is not None
        return self._detector.run(None, {self._detector_input: blob})[0]

    @property
    def runtime(self) -> str:
        return self._runtime

    def read(self, frame: Frame) -> tuple[list[PlateRead], InferenceStats]:
        """Detect and recognize on the calling thread (single-threaded path)."""
        detected = self.detect(frame)
        outcome = self.recognize(detected)
        wall = max((detected.det_ms + outcome.ocr_ms) / 1000, 1e-6)
        stats = InferenceStats(
            runtime=self._runtime,
            det_ms=detected.det_ms,
            ocr_ms=outcome.ocr_ms,
            cpu_percent=(detected.cpu_s + outcome.cpu_s) / wall * 100,
            rss_mb=_rss_mb(),
        )
        return outcome.reads, stats

    def detect(self, frame: Frame) -> DetectedFrame:
        """Run only the detector. Call from the one thread that owns the interpreter."""
        import numpy as np

        empty = DetectedFrame(frame, None, (), self._runtime, 0.0, 0.0)
        if frame.width <= 0 or frame.height <= 0:
            return empty
        image = np.frombuffer(frame.data, dtype=np.uint8)
        expected = frame.width * frame.height * 3
        if image.size < expected:
            return empty
        image = image[:expected].reshape(frame.height, frame.width, 3)
        cpu_before = time.thread_time()
        started = time.perf_counter()
        letterbox = _letterbox_u8 if self._detector_u8 else _letterbox
        blob, scale, left, top = letterbox(image, DETECTOR_SIZE)
        raw = self._detect(blob)
        boxes = _boxes_from_yolo(raw, self._threshold, scale, left, top, frame.width, frame.height)
        return DetectedFrame(
            frame=frame,
            image=image,
            boxes=tuple(boxes),
            runtime=self._runtime,
            det_ms=(time.perf_counter() - started) * 1000,
            cpu_s=time.thread_time() - cpu_before,
        )

    def recognize(self, detected: DetectedFrame) -> OcrOutcome:
        """ROI refine, enhance, quality gate, and OCR for each detected box.

        Safe to call from several worker threads at once.
        """
        import numpy as np

        frame = detected.frame
        image = detected.image
        cpu_before = time.thread_time()
        started = time.perf_counter()
        reads: list[PlateRead] = []
        for x1, y1, x2, y2, confidence in detected.boxes:
            refined = refine_plate_roi(image, x1, y1, x2, y2, self._roi_refine)
            crop = np.asarray(refined.crop_bgr)
            if crop.size == 0:
                continue
            prepared = enhance_crop(crop, self._enhancer)
            ok, metrics, reason = crop_quality(prepared, self._quality)
            if not ok and reason == "empty":
                log_quality_reject(frame.stream_name, reason, metrics)
                continue
            if not ok:
                # Soft gate: keep OCR, attach metrics for vote weighting.
                log_quality_reject(frame.stream_name, reason, metrics)
            (
                text,
                ocr_confidence,
                char_confidences,
                char_margins,
                blank_uncertainty,
                step_probs,
            ) = self._ocr_crop(prepared)
            if not text:
                continue
            pre = np.ascontiguousarray(prepared)
            if refined.method != "yolo":
                logger.debug(
                    "roi refine %s method=%s box=%d,%d %dx%d",
                    frame.stream_name,
                    refined.method,
                    refined.x1,
                    refined.y1,
                    refined.x2 - refined.x1,
                    refined.y2 - refined.y1,
                )
            reads.append(
                PlateRead(
                    text=text,
                    confidence=ocr_confidence,
                    box=PlateBox(
                        refined.x1,
                        refined.y1,
                        refined.x2 - refined.x1,
                        refined.y2 - refined.y1,
                        confidence,
                    ),
                    char_confidences=tuple(char_confidences),
                    char_margins=tuple(char_margins),
                    blank_uncertainty=float(blank_uncertainty),
                    laplacian_var=metrics.laplacian_var,
                    brightness=metrics.brightness,
                    contrast=metrics.contrast,
                    pre_ocr_bgr=pre.tobytes(),
                    pre_ocr_width=int(pre.shape[1]),
                    pre_ocr_height=int(pre.shape[0]),
                    step_probs=step_probs,
                )
            )
        return OcrOutcome(
            reads=reads,
            ocr_ms=(time.perf_counter() - started) * 1000,
            cpu_s=time.thread_time() - cpu_before,
        )

    def _ocr_crop(
        self, prepared_bgr: object
    ) -> tuple[str, float, list[float], list[float], float, tuple[tuple[float, ...], ...]]:
        import cv2
        import numpy as np

        rgb = cv2.cvtColor(prepared_bgr, cv2.COLOR_BGR2RGB)
        resized = resize_rgb_for_ocr(rgb)
        batch = np.ascontiguousarray(resized[None, ...])
        logits = self._ocr.run(None, {self._ocr_input: batch})[0][0]
        steps = logits.tolist()
        decoded = decode_characters(steps, self._alphabet)
        probs = tuple(tuple(round(p, 5) for p in row) for row in step_probabilities(steps))
        return (*decoded, probs)


DFL_BINS = 16
DETECTOR_STRIDES = (8, 16, 32)


def _anchor_grid(size: int, strides: tuple[int, ...]) -> tuple[object, object]:
    """Anchor centers ``(x + 0.5, y + 0.5)`` in stride units and the stride of each anchor."""
    import numpy as np

    points = []
    steps = []
    for stride in strides:
        cells = size // stride
        ys, xs = np.mgrid[0:cells, 0:cells]
        points.append(np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], axis=1))
        steps.append(np.full(cells * cells, float(stride)))
    return np.concatenate(points).astype(np.float32), np.concatenate(steps).astype(np.float32)


_ANCHOR_CACHE: dict[tuple[int, tuple[int, ...]], tuple[object, object]] = {}


def decode_dfl_head(
    box_dfl: object,
    scores: object,
    threshold: float,
    size: int = DETECTOR_SIZE,
    strides: tuple[int, ...] = DETECTOR_STRIDES,
) -> object:
    """Decode raw DFL box logits into a YOLO ``[1, 5, N]`` tensor (cx, cy, w, h, score).

    Only anchors with ``score > threshold`` are decoded; the rest stay zero. Matches the
    ONNX head: softmax over the 16 bins of each side, expected bin = distance in stride
    units, ``x1y1 = anchor - lt``, ``x2y2 = anchor + rb``, then scaled by the stride.
    """
    import numpy as np

    key = (size, strides)
    if key not in _ANCHOR_CACHE:
        _ANCHOR_CACHE[key] = _anchor_grid(size, strides)
    anchors, anchor_strides = _ANCHOR_CACHE[key]
    score_row = np.asarray(scores).reshape(-1)
    count = score_row.shape[0]
    out = np.zeros((1, 5, count), dtype=np.float32)
    selected = np.flatnonzero(score_row > threshold)
    if selected.size == 0:
        return out
    logits = np.asarray(box_dfl)[0][:, selected].T.reshape(-1, 4, DFL_BINS)
    logits = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=-1, keepdims=True)
    distance = weights @ np.arange(DFL_BINS, dtype=np.float32)  # [K, 4]: l, t, r, b
    centers = anchors[selected]
    stride = anchor_strides[selected, None]
    left_top = centers - distance[:, :2]
    right_bottom = centers + distance[:, 2:]
    out[0, 0:2, selected] = (left_top + right_bottom) / 2 * stride
    out[0, 2:4, selected] = (right_bottom - left_top) * stride
    out[0, 4, selected] = score_row[selected]
    return out


def join_detector_outputs(tensors: list[object], threshold: float = 0.0) -> object:
    """Return a YOLO ``[1, 5, N]`` tensor from the detector's output tensors.

    Accepted layouts: one joined ``[1, 5, N]``; split pixel boxes ``[1, 4, N]`` + scores
    ``[1, 1, N]``; or raw DFL logits ``[1, 64, N]`` + scores (decoded on the host).
    """
    import numpy as np

    if len(tensors) == 1:
        return tensors[0]
    if len(tensors) != 2:
        raise ValueError(f"expected 1 or 2 detector outputs, got {len(tensors)}")
    by_channels = {int(tensor.shape[1]): tensor for tensor in tensors}
    if set(by_channels) == {64, 1}:
        return decode_dfl_head(by_channels[64], by_channels[1], threshold)
    if set(by_channels) != {4, 1}:
        raise ValueError(
            f"expected 4/64-channel boxes and 1-channel scores, got {sorted(by_channels)}"
        )
    return np.concatenate([by_channels[4], by_channels[1]], axis=1)


def _letterbox_canvas(image: object, size: int) -> tuple[object, float, int, int]:
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - resized_height) // 2
    left = (size - resized_width) // 2
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, scale, left, top


def _letterbox(image: object, size: int) -> tuple[object, float, int, int]:
    import numpy as np

    canvas, scale, left, top = _letterbox_canvas(image, size)
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    blob = np.transpose(rgb, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(blob), scale, left, top


def _letterbox_u8(image: object, size: int) -> tuple[object, float, int, int]:
    """Same letterbox as `_letterbox`, as uint8 NCHW RGB (no float32 conversion)."""
    import numpy as np

    canvas, scale, left, top = _letterbox_canvas(image, size)
    blob = np.transpose(canvas[:, :, ::-1], (2, 0, 1))[None, ...]
    return np.ascontiguousarray(blob), scale, left, top


def _boxes_from_yolo(
    raw: object,
    threshold: float,
    scale: float,
    left: int,
    top: int,
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int, float]]:
    prediction = raw[0]
    if prediction.shape[0] < prediction.shape[-1]:
        prediction = prediction.T
    selected: list[tuple[float, float, float, float, float]] = []
    # Vectorized pre-filter: identical to skipping rows with confidence <= threshold.
    for cx, cy, width, height, confidence in prediction[prediction[:, 4] > threshold]:
        x1, y1, x2, y2 = center_to_corners(float(cx), float(cy), float(width), float(height))
        x1 = (x1 - left) / scale
        y1 = (y1 - top) / scale
        x2 = (x2 - left) / scale
        y2 = (y2 - top) / scale
        x1 = min(max(x1, 0), image_width - 1)
        y1 = min(max(y1, 0), image_height - 1)
        x2 = min(max(x2, 0), image_width)
        y2 = min(max(y2, 0), image_height)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        selected.append((x1, y1, x2, y2, float(confidence)))
    kept = nms(selected)
    return [(int(x1), int(y1), int(x2), int(y2), confidence) for x1, y1, x2, y2, confidence in kept]


def _rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (OSError, IndexError, ValueError):
        return 0.0
    return 0.0
