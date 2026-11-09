"""Load the ANPR JSON config with the stdlib."""

import json
from dataclasses import dataclass
from pathlib import Path

from anpr.quality import EnhancerSettings, QualitySettings
from anpr.roi_refine import ExpandSettings, PerspectiveSettings, RoiRefineSettings

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
    min_char_confidence: float
    quality: QualitySettings
    enhancer: EnhancerSettings
    roi_refine: RoiRefineSettings


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
    minimum = _unit_float(body.get("min_confidence", 0.70), "recognizer.min_confidence")
    min_char = _unit_float(
        body.get("min_char_confidence", minimum),
        "recognizer.min_char_confidence",
    )
    return RecognizerConfig(
        enabled=_bool(body.get("enabled"), "recognizer.enabled"),
        path=_string(body.get("path"), "recognizer.path"),
        npu_path=_string(body.get("npu_path", ""), "recognizer.npu_path"),
        delegate_path=_string(body.get("delegate_path"), "recognizer.delegate_path"),
        charset=_string(body.get("charset"), "recognizer.charset"),
        min_confidence=minimum,
        min_char_confidence=min_char,
        quality=_quality(body.get("quality")),
        enhancer=_enhancer(body.get("enhancer")),
        roi_refine=_roi_refine(body.get("roi_refine")),
    )


def _quality(raw: object) -> QualitySettings:
    if raw is None:
        return QualitySettings()
    body = _object(raw, "recognizer.quality")
    min_brightness = _non_negative(
        body.get("min_brightness", 40.0),
        "recognizer.quality.min_brightness",
    )
    max_brightness = _non_negative(
        body.get("max_brightness", 220.0),
        "recognizer.quality.max_brightness",
    )
    if max_brightness < min_brightness:
        raise ConfigError("recognizer.quality.max_brightness must be >= min_brightness")
    return QualitySettings(
        enabled=_bool(body.get("enabled", True), "recognizer.quality.enabled"),
        min_laplacian_var=_non_negative(
            body.get("min_laplacian_var", 40.0),
            "recognizer.quality.min_laplacian_var",
        ),
        min_brightness=min_brightness,
        max_brightness=max_brightness,
        min_contrast=_non_negative(
            body.get("min_contrast", 25.0),
            "recognizer.quality.min_contrast",
        ),
    )


def _enhancer(raw: object) -> EnhancerSettings:
    if raw is None:
        return EnhancerSettings()
    body = _object(raw, "recognizer.enhancer")
    gamma = body.get("gamma", 1.1)
    if isinstance(gamma, bool) or not isinstance(gamma, int | float) or float(gamma) <= 0:
        raise ConfigError("recognizer.enhancer.gamma must be a positive number")
    return EnhancerSettings(
        enabled=_bool(body.get("enabled", False), "recognizer.enhancer.enabled"),
        clahe=_bool(body.get("clahe", True), "recognizer.enhancer.clahe"),
        gamma=float(gamma),
        sharpen=_bool(body.get("sharpen", True), "recognizer.enhancer.sharpen"),
    )


def _roi_refine(raw: object) -> RoiRefineSettings:
    if raw is None:
        return RoiRefineSettings(
            expand=ExpandSettings(),
            perspective=PerspectiveSettings(),
        )
    body = _object(raw, "recognizer.roi_refine")
    return RoiRefineSettings(
        expand=_expand(body.get("expand")),
        perspective=_perspective(body.get("perspective")),
    )


def _expand(raw: object) -> ExpandSettings:
    if raw is None:
        return ExpandSettings()
    body = _object(raw, "recognizer.roi_refine.expand")
    outer_default = body.get("outer_ratio", body.get("max_ratio", 0.20))
    return ExpandSettings(
        enabled=_bool(body.get("enabled", True), "recognizer.roi_refine.expand.enabled"),
        inner_ratio=_unit_float(
            body.get("inner_ratio", 0.20),
            "recognizer.roi_refine.expand.inner_ratio",
        ),
        outer_ratio=_unit_float(outer_default, "recognizer.roi_refine.expand.outer_ratio"),
        edge_threshold=_unit_float(
            body.get("edge_threshold", body.get("ink_threshold", 0.08)),
            "recognizer.roi_refine.expand.edge_threshold",
        ),
        pad_ratio=_unit_float(
            body.get("pad_ratio", 0.05),
            "recognizer.roi_refine.expand.pad_ratio",
        ),
        max_ratio=_unit_float(outer_default, "recognizer.roi_refine.expand.max_ratio"),
    )


def _perspective(raw: object) -> PerspectiveSettings:
    if raw is None:
        return PerspectiveSettings()
    body = _object(raw, "recognizer.roi_refine.perspective")
    min_aspect = _positive_number(
        body.get("min_aspect", 1.8),
        "recognizer.roi_refine.perspective.min_aspect",
    )
    max_aspect = _positive_number(
        body.get("max_aspect", 6.0),
        "recognizer.roi_refine.perspective.max_aspect",
    )
    if max_aspect < min_aspect:
        raise ConfigError("recognizer.roi_refine.perspective.max_aspect must be >= min_aspect")
    return PerspectiveSettings(
        enabled=_bool(
            body.get("enabled", False),
            "recognizer.roi_refine.perspective.enabled",
        ),
        max_ratio=_unit_float(
            body.get("max_ratio", 0.35),
            "recognizer.roi_refine.perspective.max_ratio",
        ),
        min_score=_unit_float(
            body.get("min_score", 0.45),
            "recognizer.roi_refine.perspective.min_score",
        ),
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        canny_low=_bounded_int(
            body.get("canny_low", 40),
            "recognizer.roi_refine.perspective.canny_low",
            0,
            255,
        ),
        canny_high=_bounded_int(
            body.get("canny_high", 120),
            "recognizer.roi_refine.perspective.canny_high",
            0,
            255,
        ),
    )


def _unit_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{label} must be a number")
    if not 0.0 <= float(value) <= 1.0:
        raise ConfigError(f"{label} must be between 0 and 1")
    return float(value)


def _non_negative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or float(value) < 0:
        raise ConfigError(f"{label} must be a non-negative number")
    return float(value)


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
