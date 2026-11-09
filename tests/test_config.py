from pathlib import Path

import pytest

from anpr.cli import main
from anpr.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "config.example.json"


def test_example_config_loads() -> None:
    config = load_config(EXAMPLE)
    assert [stream.name for stream in config.streams] == ["videoCapture", "local_test"]
    assert config.enabled_streams()[0].name == "videoCapture"
    assert config.enabled_streams()[0].uri == "rtsp://10.0.0.117:8554/videoCapture"
    assert config.detector.path.endswith("plate_detector.onnx")
    assert config.detector.conf_threshold == 0.70
    assert config.recognizer.enabled is True
    assert config.recognizer.path.endswith("cct_s_v2_global.onnx")
    assert config.recognizer.min_confidence == 0.70
    assert config.recognizer.min_char_confidence == 0.70
    assert config.recognizer.quality.enabled is True
    assert config.recognizer.quality.min_laplacian_var == 40
    assert config.recognizer.enhancer.enabled is True
    assert config.recognizer.enhancer.gamma == 1.1
    assert config.recognizer.roi_refine.expand.enabled is True
    assert config.recognizer.roi_refine.expand.inner_ratio == 0.20
    assert config.recognizer.roi_refine.expand.outer_ratio == 0.20
    assert config.recognizer.roi_refine.expand.pad_ratio == 0.05
    assert config.recognizer.roi_refine.perspective.enabled is False
    assert config.recognizer.roi_refine.perspective.max_ratio == 0.35
    assert config.service.runtime == "cpu"
    assert config.service.inference_fps_limit == 2
    assert config.service.api_socket == "/run/anpr/api.sock"
    assert config.service.jpeg_quality == 80
    assert config.service.history_limit == 50
    assert config.web.listen_host == "0.0.0.0"
    assert config.web.listen_port == 8080
    assert config.web.poll_seconds == 0.5
    assert config.detector.npu_path.endswith("plate_detector.int8.onnx")
    assert config.recognizer.npu_path.endswith("cct_s_v2_global.int8.onnx")


def test_check_config_command() -> None:
    assert main(["check-config", "--config", str(EXAMPLE)]) == 0


def test_rejects_threshold_above_one(tmp_path: Path) -> None:
    path = _write(tmp_path, conf_threshold=1.5)
    with pytest.raises(ConfigError, match="between 0 and 1"):
        load_config(path)


def test_rejects_unknown_stream_type(tmp_path: Path) -> None:
    path = _write(tmp_path, stream_type="file")
    with pytest.raises(ConfigError, match="type must be one of"):
        load_config(path)


def test_rejects_relative_api_socket(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8").replace(
        '"api_socket": "/run/anpr/api.sock"',
        '"api_socket": "anpr.sock"',
    )
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="absolute"):
        load_config(path)


def test_rejects_web_port_above_65535(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8").replace(
        '"listen_port": 8080', '"listen_port": 70000'
    )
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="listen_port"):
        load_config(path)


def test_rejects_duplicate_stream_names(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8").replace('"local_test"', '"videoCapture"', 1)
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="unique"):
        load_config(path)


def _write(tmp_path: Path, *, conf_threshold: float = 0.5, stream_type: str = "rtsp") -> Path:
    text = EXAMPLE.read_text(encoding="utf-8")
    text = text.replace('"conf_threshold": 0.70', f'"conf_threshold": {conf_threshold}')
    text = text.replace('"type": "rtsp"', f'"type": "{stream_type}"', 1)
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    return path
