# Architecture

Each enabled stream has one capture thread and one latest-frame slot. A single inference thread owns the `tflite_runtime` interpreter. The interpreter is not shared across threads.

```text
stream thread -> latest frame slot -> inference thread -> detector -> recognizer -> journal
```

The recognizer stays off until `recognizer.enabled` is true and a charset and model exist. Stock `yolov8n.pt` is a COCO detector and is only a pipeline smoke model.

## Runtime on the board

Checked on `root@10.0.0.165` with `imx-image-full`:

- Python 3.14.5 and `tflite_runtime` at `/usr/lib/python3.14/site-packages/tflite_runtime`
- `/dev/neutron0` is `root:video` mode `0660`
- TFLite delegate: `/usr/lib/libneutron_delegate.so` (`tensorflow-lite-neutron-delegate` 2.19.0)
- LiteRT delegate: `/usr/lib/liblitert_neutron_delegate.so` (reported by `probe-npu`, not loaded into `tflite_runtime`)
- There is no `/usr/lib/aarch64-linux-gnu` on this rootfs. The probe still searches it.

`probe-npu` searches `/usr/lib` and `/usr/lib/aarch64-linux-gnu` for `*neutron*delegate*.so` and `libtensorflow-lite-neutron-delegate.so`. It selects `libneutron_delegate.so` for `tflite_runtime`. `libtensorflow-lite-neutron-delegate.so` is not in this image.

## Capture

Primary path is PyGObject (`python3-pygobject` 3.56.1, GStreamer 1.28.1). `capture.py` imports `gi` only when a caller opens a pipeline.

Hardware H.264 decode is `v4l2h264dec` in `libgstvideo4linux2.so` (gst-plugins-good). The image firmware is Wave6 (`firmware-imx-vpu-wave`). `avdec_h264` is not installed. Colorspace conversion uses `imxvideoconvert_g2d`.

RTP pipeline:

```text
udpsrc port=5004 ! application/x-rtp,encoding-name=H264 ! rtph264depay ! h264parse ! v4l2h264dec ! imxvideoconvert_g2d ! video/x-raw,format=BGR ! appsink drop=true max-buffers=1
```

OpenCV 4.13.0.imx is built with GStreamer (`cv2.CAP_GSTREAMER`). If appsink mapping gets in the way, open the same string with:

```python
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
```

## Config and install

Sample config: `configs/config.example.json`. On the board the service reads `/etc/anpr/config.json`. Device paths and thresholds live only in that file.

`deploy/systemd/anpr.service` runs as user `anpr` in group `video`. `deploy/udev/99-anpr-neutron.rules` keeps `/dev/neutron0` on group `video`.
