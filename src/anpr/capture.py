"""GStreamer pipeline strings and the board capture opener.

`gi` is imported only when a caller opens a pipeline.
"""

import logging
from urllib.parse import urlparse

from anpr.config import StreamConfig

logger = logging.getLogger("anpr.capture")

APPSINK = "appsink name=anpr drop=true max-buffers=1 sync=false"

RTSP_PIPELINE = (
    "rtspsrc location={uri} latency=100 protocols=tcp ! rtph264depay ! "
    "h264parse ! v4l2h264dec ! imxvideoconvert_g2d ! video/x-raw,format=BGR ! " + APPSINK
)

RTP_PIPELINE = (
    "udpsrc port={port} ! application/x-rtp,encoding-name=H264 ! rtph264depay ! "
    "h264parse ! v4l2h264dec ! imxvideoconvert_g2d ! video/x-raw,format=BGR ! " + APPSINK
)

V4L2_PIPELINE = "v4l2src device={device} ! videoconvert ! video/x-raw,format=BGR ! " + APPSINK


def rtsp_pipeline(uri: str) -> str:
    if urlparse(uri).scheme != "rtsp" or urlparse(uri).hostname is None:
        raise ValueError(f"RTSP uri is missing a host: {uri}")
    return RTSP_PIPELINE.format(uri=uri)


def rtp_pipeline(port: int) -> str:
    if port <= 0 or port > 65535:
        raise ValueError(f"RTP port out of range: {port}")
    return RTP_PIPELINE.format(port=port)


def pipeline_for_stream(stream: StreamConfig) -> str:
    if stream.pipeline.strip():
        return stream.pipeline
    if stream.type == "rtsp":
        return rtsp_pipeline(stream.uri)
    if stream.type == "rtp":
        return rtp_pipeline(_rtp_port(stream.uri))
    if stream.type == "v4l2":
        return V4L2_PIPELINE.format(device=stream.uri)
    raise ValueError(f"unsupported stream type: {stream.type}")


class GStreamerCapture:
    """Play one pipeline and pull the latest BGR sample from the named appsink."""

    def __init__(self, description: str) -> None:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        pipeline = Gst.parse_launch(description)
        sink = pipeline.get_by_name("anpr")
        if sink is None:
            raise RuntimeError("capture pipeline is missing appsink name=anpr")
        self._gst = Gst
        self._pipeline = pipeline
        self._sink = sink
        pipeline.set_state(Gst.State.PLAYING)
        logger.info("capture playing")

    def pull(self, timeout_ns: int = 500_000_000) -> tuple[bytes, int] | None:
        self._raise_if_stopped()
        sample = self._sink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            return None
        buffer = sample.get_buffer()
        if buffer is None:
            return None
        ok, info = buffer.map(self._gst.MapFlags.READ)
        if not ok:
            return None
        try:
            payload = bytes(info.data)
        finally:
            buffer.unmap(info)
        return payload, buffer.pts

    def _raise_if_stopped(self) -> None:
        bus = self._pipeline.get_bus()
        while True:
            message = bus.pop()
            if message is None:
                return
            if message.type == self._gst.MessageType.ERROR:
                err, _debug = message.parse_error()
                raise RuntimeError(str(err))
            if message.type == self._gst.MessageType.EOS:
                raise RuntimeError("capture stream ended")

    def close(self) -> None:
        self._pipeline.set_state(self._gst.State.NULL)


def _rtp_port(uri: str) -> int:
    port = urlparse(uri).port
    if port is None:
        raise ValueError(f"RTP uri has no port: {uri}")
    return port
