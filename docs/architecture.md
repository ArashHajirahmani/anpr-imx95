# Architecture

Each enabled stream has one capture thread and one latest-frame slot. A single inference thread owns the `tflite_runtime` interpreter. The interpreter is not shared across threads.

```text
stream thread -> latest frame slot -> inference thread -> detector -> recognizer
recognizer -> journal
recognizer -> result store -> HTTP on a Unix socket -> anpr web -> browser
```

When `recognizer.enabled` is true, one thread runs the plate detector and OCR on the latest frame. `service.runtime` is `cpu` for the float ONNX models or `npu` for the INT8 models on ONNX Runtime's Neutron provider. The journal line includes `runtime`, `det_ms`, `ocr_ms`, `cpu`, and `rss_mb`.

After each plate crop, an image-quality gate can reject blurry, dark, bright, or low-contrast boxes before OCR (`recognizer.quality`). An optional experimental enhancer (`recognizer.enhancer`, off by default) may CLAHE / gamma / sharpen the crop, but it still must feed CCT as 128×64 RGB NHWC uint8 with no Python normalization, and it does not override confidence rejection. A plate is logged only when every kept character clears `recognizer.min_char_confidence` (0.07) and the mean character probability clears `recognizer.min_confidence` (0.07). Those two checks are independent of the quality gate. The npu path subtracts 0.005 from both floors, because the INT8 OCR score for the same crop sits just under the float score. Repeated copies of the same text are not logged again. A frame with no accepted plate does not clear that memory, so the same text is not logged again after a miss.

## Result API and web client

`anpr run` stores the latest JPEG and the plates found on that frame. Accepted plates also fill a history of `service.history_limit` events (4 in the sample). The image stays binary. Metadata is JSON. Both are HTTP resources on the Unix socket `service.api_socket` (`/run/anpr/api.sock`). The recognition process does not listen on TCP.

```text
GET /health
GET /v1/latest
GET /v1/latest.jpg
GET /v1/history
GET /v1/history/<id>.jpg
```

A plate object has `text`, `confidence`, `characters` (each `{ "char", "confidence" }`), and `box` (`x`, `y`, `width`, `height`, `confidence`). The box is in pixels of that JPEG. Before the first frame, `/v1/latest` returns 503.

```bash
curl --unix-socket /run/anpr/api.sock http://localhost/v1/latest
curl --unix-socket /run/anpr/api.sock http://localhost/v1/latest.jpg -o latest.jpg
```

`anpr web` is a separate process, installed as `anpr-web.service`. It serves one page on `web.listen_host` and `web.listen_port` (`0.0.0.0:8080` in the sample). The left side is the latest frame, the plate in large type, and the other fields in small type. A green box is placed from the box pixels. The right side is the last four plates, each cropped to its box. The page keeps serving when the socket is down and shows the last good frame with the error. Journal lines cover bind, reachability, each new plate, and a status line every 30 seconds. `ANPR_LOG_LEVEL=DEBUG` adds one line per API request and per poll.

`anpr.service` sets `RuntimeDirectory=anpr`, which creates `/run/anpr` for the socket.

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

The bench publisher is `tools/host/run-rtsp-server.sh`. MediaMTX listens on `0.0.0.0:8554` and ffmpeg publishes each H.264 file to `rtsp://<pc-address>:8554/videoCapture`. From this PC that address is `10.0.0.117`. The board opens it with `rtspsrc` over TCP. Hardware decode is `v4l2h264dec` in `libgstvideo4linux2.so` (gst-plugins-good, Wave6 firmware `firmware-imx-vpu-wave`). Colorspace conversion uses `imxvideoconvert_g2d`.

```text
rtspsrc location=rtsp://10.0.0.117:8554/videoCapture latency=100 protocols=tcp ! rtph264depay ! h264parse ! v4l2h264dec ! imxvideoconvert_g2d ! video/x-raw,format=RGB ! videoconvert ! video/x-raw,format=BGR ! appsink name=anpr drop=true max-buffers=1 sync=false
```

A raw RTP listener remains available for `type: "rtp"`. OpenCV 4.13.0.imx is built with GStreamer (`cv2.CAP_GSTREAMER`) if the appsink path has to be replaced with the same pipeline string:

```python
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
```

## Config and install

Sample config: `configs/config.example.json`. On the board the service reads `/etc/anpr/config.json`. Device paths and thresholds live only in that file, including `recognizer.min_char_confidence`, `recognizer.quality`, and `recognizer.enhancer`.

`deploy/systemd/anpr.service` and `deploy/systemd/anpr-web.service` run as user `anpr` in group `video`. `deploy/udev/99-anpr-neutron.rules` keeps `/dev/neutron0` on group `video`. The web unit does not stop when recognition is down.
