from pathlib import Path

import pytest

from anpr.cli import main
from anpr.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "config.example.json"


def test_example_config_loads() -> None:
    config = load_config(EXAMPLE)
    assert [stream.name for stream in config.streams] == ["gate_entrance", "local_test"]
    assert config.enabled_streams()[0].name == "gate_entrance"
    assert config.detector.delegate_path == ""
    assert config.detector.conf_threshold == 0.5
    assert config.recognizer.enabled is False
    assert config.service.inference_fps_limit == 15


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


def test_rejects_duplicate_stream_names(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8").replace('"local_test"', '"gate_entrance"', 1)
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="unique"):
        load_config(path)


def test_run_wires_and_exits() -> None:
    assert main(["run", "--config", str(EXAMPLE)]) == 0


def _write(tmp_path: Path, *, conf_threshold: float = 0.5, stream_type: str = "rtp") -> Path:
    text = EXAMPLE.read_text(encoding="utf-8")
    text = text.replace('"conf_threshold": 0.5', f'"conf_threshold": {conf_threshold}')
    text = text.replace('"type": "rtp"', f'"type": "{stream_type}"', 1)
    path = tmp_path / "config.json"
    path.write_text(text, encoding="utf-8")
    return path
