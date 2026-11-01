from pathlib import Path

from anpr.capture import pipeline_for_stream, rtp_pipeline
from anpr.config import StreamConfig, load_config
from anpr.pipeline import wire
from anpr.types import Frame

EXAMPLE = Path(__file__).resolve().parents[1] / "configs" / "config.example.json"


def test_latest_frame_slot_keeps_newest() -> None:
    wired = wire(load_config(EXAMPLE))
    slot = wired.slots["gate_entrance"]
    slot.publish(Frame("gate_entrance", "first", 1))
    slot.publish(Frame("gate_entrance", "second", 2))
    taken = slot.take()
    assert taken is not None
    assert taken.data == "second"
    assert slot.take() is None


def test_explicit_pipeline_is_kept() -> None:
    config = load_config(EXAMPLE)
    wired = wire(config)
    assert "v4l2h264dec" in wired.pipelines["gate_entrance"]
    assert "imxvideoconvert_g2d" in wired.pipelines["gate_entrance"]
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
    assert "appsink drop=true max-buffers=1" in built
