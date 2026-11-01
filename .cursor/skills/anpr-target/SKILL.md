---
name: anpr-target
description: >
  Runs i.MX95 ANPR board steps: Neutron delegate probe, GStreamer capture,
  systemd install, and INT8 TFLite export with a calibration set. Use when
  editing inference, capture, probe-npu, deploy units, neutron-converter, or
  tools/host/export_int8.py.
---

# ANPR on the i.MX95

Read [docs/architecture.md](../../../docs/architecture.md) and [docs/model-export.md](../../../docs/model-export.md) before changing runtime behavior. The notes below are the board-checked facts.

## Delegate

`probe-npu` searches `/usr/lib` and `/usr/lib/aarch64-linux-gnu` for `*neutron*delegate*.so` and `libtensorflow-lite-neutron-delegate.so`.

On the checked `imx-image-full` rootfs:

- Load `/usr/lib/libneutron_delegate.so` with `tflite_runtime.interpreter.load_delegate`.
- `/usr/lib/liblitert_neutron_delegate.so` is LiteRT. Report it and do not pass it to `tflite_runtime`.
- `libtensorflow-lite-neutron-delegate.so` is absent. `/usr/lib/aarch64-linux-gnu` is absent.
- `/dev/neutron0` is `root:video` mode `0660`.

Keep a single interpreter on one thread.

## Capture

Primary open path is PyGObject. Import `gi` inside the open function. The bench publisher is `tools/host/run-rtsp-server.sh`, path `videoCapture`, port 8554. The board reads it with `rtspsrc` over TCP. Hardware H.264 decode is `v4l2h264dec` (Wave6 via gst-plugins-good) followed by `imxvideoconvert_g2d`.

If appsink mapping fails, open the same pipeline with `cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)`. OpenCV 4.13.0.imx on this image is built with GStreamer.

## INT8 export

Run `tools/host/export_int8.py`. It exits when `datasets/calibration/images` has fewer than 100 `.jpg`, `.jpeg`, or `.png` frames from the real cameras. Then run `neutron-converter` from the eIQ build that matches BSP `6.18.37-2.1.0`, using flags from that binary's `--help`.

Stop when the converter log leaves ops on the CPU through XNNPACK. Do not copy that `.tflite` to the board.

## Service install

Templates live in `deploy/systemd/anpr.service` and `deploy/udev/99-anpr-neutron.rules`. The process user is `anpr` in group `video`. Config path is `/etc/anpr/config.json`.
