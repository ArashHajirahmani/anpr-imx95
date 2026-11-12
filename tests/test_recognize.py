import threading
from pathlib import Path

import numpy as np

from anpr.recognize import PlateReader
from anpr.sysmon import BusyMeter


class _FakeInterpreter:
    def __init__(self) -> None:
        self.fed: np.ndarray | None = None

    def set_tensor(self, index: int, value: np.ndarray) -> None:
        self.fed = value

    def invoke(self) -> None:
        return None

    def get_tensor(self, index: int) -> np.ndarray:
        return np.full((1, 10, 37), -128 + 64, dtype=np.int8)


def _tflite_ocr_reader() -> PlateReader:
    reader = PlateReader.__new__(PlateReader)
    reader._ocr_backend = "tflite"
    reader._ocr_tflite = _FakeInterpreter()
    reader._ocr_lock = threading.Lock()
    reader.npu_meter = BusyMeter()
    reader._ocr_input_index = 0
    reader._ocr_input_dtype = np.uint8
    reader._ocr_input_scale = 0.0
    reader._ocr_input_zero_point = 0
    reader._ocr_output_index = 1
    reader._ocr_output_scale = 1 / 256
    reader._ocr_output_zero_point = -128
    return reader


def test_tflite_ocr_dequantizes_probabilities() -> None:
    reader = _tflite_ocr_reader()
    probs = reader._ocr_logits(np.zeros((1, 64, 128, 3), dtype=np.uint8))
    assert probs.shape == (10, 37)
    assert probs.dtype == np.float32
    assert abs(float(probs[0, 0]) - 0.25) < 1e-6
    assert reader._ocr_tflite.fed.dtype == np.uint8


def test_describe_reports_ocr_on_npu() -> None:
    reader = _tflite_ocr_reader()
    reader._detector_backend = "tflite"
    reader._detector_path = Path("det.neutron.tflite")
    reader._ocr_path = Path("ocr.neutron.tflite")
    reader._ocr_threads = 1
    info = reader.describe()
    assert info["ocr"]["device"] == "npu"
    assert info["ocr"]["backend"] == "tflite"
    assert info["detector"]["device"] == "npu"


def test_describe_reports_ocr_on_cpu_for_onnx() -> None:
    reader = _tflite_ocr_reader()
    reader._ocr_backend = "onnx"
    reader._detector_backend = "tflite"
    reader._detector_path = Path("det.neutron.tflite")
    reader._ocr_path = Path("ocr.int8.onnx")
    reader._ocr_threads = 1
    assert reader.describe()["ocr"]["device"] == "cpu"


def test_int8_ocr_input_is_quantized_from_pixels() -> None:
    reader = _tflite_ocr_reader()
    reader._ocr_input_dtype = np.int8
    pixels = np.array([[0, 100, 255]], dtype=np.uint8)
    reader._ocr_input_scale = 1.0
    reader._ocr_input_zero_point = -128
    assert reader._ocr_feed(pixels).tolist() == [[-128, -28, 127]]
    reader._ocr_input_scale = 0.9
    reader._ocr_input_zero_point = -120
    assert reader._ocr_feed(pixels).tolist() == [[-120, -9, 127]]
