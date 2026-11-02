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

Primary path is PyGObject (`python3-pygobject` 3.56.1, GStreamer 1.28.1). `capture.py` imports `gi` only when a caller opens a pipeline. `anpr run` keeps one thread per enabled stream and republishes the latest frame until SIGTERM.

The bench publisher is `tools/host/run-rtsp-server.sh`. It serves H.264 MP4 files on `rtsp://<pc-address>:8554/videoCapture`. From this PC that address is `10.0.0.117`. The board opens it with `rtspsrc` over TCP. Hardware decode is `v4l2h264dec` in `libgstvideo4linux2.so` (gst-plugins-good, Wave6 firmware `firmware-imx-vpu-wave`). Colorspace conversion uses `imxvideoconvert_g2d`.

```text
rtspsrc location=rtsp://10.0.0.117:8554/videoCapture latency=100 protocols=tcp ! rtph264depay ! h264parse ! v4l2h264dec ! imxvideoconvert_g2d ! video/x-raw,format=RGB ! videoconvert ! video/x-raw,format=BGR ! appsink name=anpr drop=true max-buffers=1 sync=false
```

A raw RTP listener remains available for `type: "rtp"`. OpenCV 4.13.0.imx is built with GStreamer (`cv2.CAP_GSTREAMER`) if the appsink path has to be replaced with the same pipeline string:

```python
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
```

## Config and install

Sample config: `configs/config.example.json`. On the board the service reads `/etc/anpr/config.json`. Device paths and thresholds live only in that file.

`deploy/systemd/anpr.service` runs as user `anpr` in group `video`. `deploy/udev/99-anpr-neutron.rules` keeps `/dev/neutron0` on group `video`.
