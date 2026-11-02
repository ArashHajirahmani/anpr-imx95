"""Load the ANPR JSON config with the stdlib."""

import json
from dataclasses import dataclass
from pathlib import Path

STREAM_TYPES = frozenset({"rtsp", "rtp", "v4l2"})
RUNTIMES = frozenset({"cpu", "npu"})


class ConfigError(ValueError):
    """The config file is missing a field or has the wrong type."""


@dataclass(frozen=True)
class StreamConfig:
    name: str
    type: str
    uri: str
    enabled: bool
    pipeline: str


@dataclass(frozen=True)
class DetectorConfig:
    path: str
    npu_path: str
    delegate_path: str
    conf_threshold: float
    input_width: int
    input_height: int


@dataclass(frozen=True)
class RecognizerConfig:
    enabled: bool
    path: str
    npu_path: str
    delegate_path: str
    charset: str
    min_confidence: float


@dataclass(frozen=True)
class ServiceConfig:
    runtime: str
    inference_fps_limit: int
    api_socket: str
    jpeg_quality: int
    history_limit: int


@dataclass(frozen=True)
class WebConfig:
    listen_host: str
    listen_port: int
    poll_seconds: float


@dataclass(frozen=True)
class AppConfig:
    streams: tuple[StreamConfig, ...]
    detector: DetectorConfig
    recognizer: RecognizerConfig
    service: ServiceConfig
    web: WebConfig

    def enabled_streams(self) -> tuple[StreamConfig, ...]:
        return tuple(stream for stream in self.streams if stream.enabled)


def load_config(path: Path) -> AppConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"{path}: config must be a JSON object")
    streams = _streams(payload)
    names = [stream.name for stream in streams]
    if len(names) != len(set(names)):
        raise ConfigError("stream names must be unique")
    return AppConfig(
        streams=streams,
        detector=_detector(payload),
        recognizer=_recognizer(payload),
        service=_service(payload),
        web=_web(payload),
    )


def _streams(payload: dict[str, object]) -> tuple[StreamConfig, ...]:
    raw = payload.get("streams")
    if not isinstance(raw, list):
        raise ConfigError("streams must be a list")
    return tuple(_stream(item, index) for index, item in enumerate(raw))


def _stream(item: object, index: int) -> StreamConfig:
    label = f"streams[{index}]"
    body = _object(item, label)
    stream_type = _string(body.get("type"), f"{label}.type")
    if stream_type not in STREAM_TYPES:
        raise ConfigError(f"{label}.type must be one of {sorted(STREAM_TYPES)}")
    pipeline = body.get("pipeline", "")
    return StreamConfig(
        name=_string(body.get("name"), f"{label}.name"),
        type=stream_type,
        uri=_string(body.get("uri"), f"{label}.uri"),
        enabled=_bool(body.get("enabled"), f"{label}.enabled"),
        pipeline=_string(pipeline, f"{label}.pipeline"),
    )


def _detector(payload: dict[str, object]) -> DetectorConfig:
    body = _object(payload.get("detector"), "detector")
    threshold = body.get("conf_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, int | float):
        raise ConfigError("detector.conf_threshold must be a number")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ConfigError("detector.conf_threshold must be between 0 and 1")
    return DetectorConfig(
        path=_string(body.get("path"), "detector.path"),
        npu_path=_string(body.get("npu_path", ""), "detector.npu_path"),
        delegate_path=_string(body.get("delegate_path"), "detector.delegate_path"),
        conf_threshold=float(threshold),
        input_width=_positive_int(body.get("input_width"), "detector.input_width"),
        input_height=_positive_int(body.get("input_height"), "detector.input_height"),
    )


def _recognizer(payload: dict[str, object]) -> RecognizerConfig:
    body = _object(payload.get("recognizer"), "recognizer")
    minimum = body.get("min_confidence", 0.07)
    if isinstance(minimum, bool) or not isinstance(minimum, int | float):
        raise ConfigError("recognizer.min_confidence must be a number")
    if not 0.0 <= float(minimum) <= 1.0:
        raise ConfigError("recognizer.min_confidence must be between 0 and 1")
    return RecognizerConfig(
        enabled=_bool(body.get("enabled"), "recognizer.enabled"),
        path=_string(body.get("path"), "recognizer.path"),
        npu_path=_string(body.get("npu_path", ""), "recognizer.npu_path"),
        delegate_path=_string(body.get("delegate_path"), "recognizer.delegate_path"),
        charset=_string(body.get("charset"), "recognizer.charset"),
        min_confidence=float(minimum),
    )


def _service(payload: dict[str, object]) -> ServiceConfig:
    body = _object(payload.get("service"), "service")
    runtime = body.get("runtime", "cpu")
    if not isinstance(runtime, str) or runtime not in RUNTIMES:
        raise ConfigError("service.runtime must be cpu or npu")
    return ServiceConfig(
        runtime=runtime,
        inference_fps_limit=_positive_int(
            body.get("inference_fps_limit"),
            "service.inference_fps_limit",
        ),
        api_socket=_socket_path(body.get("api_socket"), "service.api_socket"),
        jpeg_quality=_bounded_int(body.get("jpeg_quality"), "service.jpeg_quality", 1, 100),
        history_limit=_positive_int(body.get("history_limit"), "service.history_limit"),
    )


def _web(payload: dict[str, object]) -> WebConfig:
    body = _object(payload.get("web"), "web")
    return WebConfig(
        listen_host=_nonempty(body.get("listen_host"), "web.listen_host"),
        listen_port=_bounded_int(body.get("listen_port"), "web.listen_port", 1, 65535),
        poll_seconds=_positive_number(body.get("poll_seconds"), "web.poll_seconds"),
    )


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{label} must be a string")
    return value


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be a boolean")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{label} must be a positive integer")
    return value


def _bounded_int(value: object, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{label} must be an integer")
    if not low <= value <= high:
        raise ConfigError(f"{label} must be between {low} and {high}")
    return value


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ConfigError(f"{label} must be a positive number")
    return float(value)


def _socket_path(value: object, label: str) -> str:
    text = _string(value, label)
    if not text.startswith("/") or text.endswith("/"):
        raise ConfigError(f"{label} must be an absolute path")
    return text


def _nonempty(value: object, label: str) -> str:
    text = _string(value, label)
    if not text:
        raise ConfigError(f"{label} must not be empty")
    return text
