"""GStreamer pipeline strings for board capture.

`gi` is imported only when a caller opens a pipeline.
"""

from urllib.parse import urlparse

from anpr.config import StreamConfig

RTP_PIPELINE = (
    "udpsrc port={port} ! application/x-rtp,encoding-name=H264 ! rtph264depay ! "
    "h264parse ! v4l2h264dec ! imxvideoconvert_g2d ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1"
)

V4L2_PIPELINE = (
    "v4l2src device={device} ! videoconvert ! video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1"
)


def rtp_pipeline(port: int) -> str:
    if port <= 0 or port > 65535:
        raise ValueError(f"RTP port out of range: {port}")
    return RTP_PIPELINE.format(port=port)


def pipeline_for_stream(stream: StreamConfig) -> str:
    if stream.pipeline.strip():
        return stream.pipeline
    if stream.type == "rtp":
        return rtp_pipeline(_rtp_port(stream.uri))
    if stream.type == "v4l2":
        return V4L2_PIPELINE.format(device=stream.uri)
    raise ValueError(f"unsupported stream type: {stream.type}")


def open_gstreamer(pipeline: str) -> object:
    """Parse a pipeline with PyGObject. Call this only on the board."""
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst.parse_launch(pipeline)


def _rtp_port(uri: str) -> int:
    port = urlparse(uri).port
    if port is None:
        raise ValueError(f"RTP uri has no port: {uri}")
    return port
