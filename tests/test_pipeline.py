import threading
from pathlib import Path

from anpr.capture import pipeline_for_stream, rtp_pipeline, rtsp_pipeline
from anpr.config import StreamConfig, load_config
from anpr.pipeline import run_streams, wire
from anpr.types import Frame

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "config.example.json"
ROOT = Path(__file__).resolve().parents[1]


def test_latest_frame_slot_keeps_newest() -> None:
    wired = wire(load_config(EXAMPLE))
    slot = wired.slots["videoCapture"]
    slot.publish(Frame("videoCapture", "first", 1))
    slot.publish(Frame("videoCapture", "second", 2))
    taken = slot.take()
    assert taken is not None
    assert taken.data == "second"
    assert slot.take() is None


def test_explicit_pipeline_is_kept() -> None:
    config = load_config(EXAMPLE)
    wired = wire(config)
    description = wired.pipelines["videoCapture"]
    assert description == rtsp_pipeline("rtsp://10.0.0.117:8554/videoCapture")
    assert "rtspsrc" in description
    assert "protocols=tcp" in description
    assert "v4l2h264dec" in description
    assert "local_test" not in wired.pipelines


def test_empty_pipeline_builds_rtp_string() -> None:
    stream = StreamConfig(
        name="gate",
        type="rtp",
        uri="rtp://192.168.1.100:5004",
        enabled=True,
        pipeline="",
    )
    built = pipeline_for_stream(stream)
    assert built == rtp_pipeline(5004)
    assert "v4l2h264dec" in built
    assert "appsink name=anpr drop=true max-buffers=1 sync=false" in built


def test_run_streams_publishes_one_frame_and_closes() -> None:
    opened: list[str] = []
    closed: list[bool] = []
    stop = threading.Event()

    class _Fake:
        def pull(self, timeout_ns: int = 0) -> tuple[bytes, int, int, int]:
            stop.set()
            return b"frame", 10, 0, 0

        def close(self) -> None:
            closed.append(True)

    def opener(description: str) -> _Fake:
        opened.append(description)
        return _Fake()

    wired = run_streams(load_config(EXAMPLE), stop, opener)
    assert len(opened) == 1
    assert "videoCapture" in opened[0]
    assert closed == [True]
    frame = wired.slots["videoCapture"].take()
    assert frame is not None
    assert frame.data == b"frame"


def test_stopped_service_does_not_open_capture() -> None:
    opened: list[str] = []
    stop = threading.Event()
    stop.set()
    run_streams(load_config(EXAMPLE), stop, opened.append)
    assert opened == []


def test_publisher_script_names_the_video_capture_path() -> None:
    script = (ROOT / "tools/host/run-rtsp-server.sh").read_text(encoding="utf-8")
    assert "videoCapture" in script
    assert "youtube" not in script.lower()
