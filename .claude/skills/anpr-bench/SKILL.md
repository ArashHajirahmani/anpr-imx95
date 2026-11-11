---
name: anpr-bench
description: Publish bench video over RTSP from this PC for the i.MX95 board and check the stream/board steps. Use when testing capture or recognition end to end with MP4 files, or when the RTSP source looks dead.
disable-model-invocation: true
---

# RTSP bench

1. Confirm `tools/host/mediamtx` exists (gitignored; see README). Videos must be H.264; phone `.MOV` files are often HEVC and need `ffmpeg` conversion first.
2. Publish: `tools/host/run-rtsp-server.sh datasets/videos` (or one file). It prints the board URL; the sample config expects the same URL on port 8554, path `videoCapture`. Run it in the background and keep it for the session.
3. On the board (addresses in `CLAUDE.local.md`; ask before changing anything): `journalctl -u anpr -u anpr-web -f` and
   `curl --unix-socket /run/anpr/api.sock http://localhost/v1/latest`.
4. If the PC address changed, update `stream` URLs in the JSON config, not code.

Publishing a stream and touching the board are outward actions: confirm with the user first.
