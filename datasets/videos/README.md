# Bench RTSP videos

H.264 `.mp4` files for `tools/host/run-rtsp-server.sh`. The board decodes with `v4l2h264dec`, so tracks must be H.264 (not HEVC/H.265).

```bash
tools/host/run-rtsp-server.sh datasets/videos
tools/host/run-rtsp-server.sh datasets/videos/IMG_9475.mp4
```

In a foreground terminal: Space pauses, ← / → seek ±10s, `q` quits. Background/`nohup` runs have no keyboard control. Seek restarts the RTSP publisher; the board capture reconnects after a short gap.

Board URL (example from this PC): `rtsp://10.0.0.117:8554/videoCapture`

`.mp4` files here are gitignored. Keep this README tracked.
