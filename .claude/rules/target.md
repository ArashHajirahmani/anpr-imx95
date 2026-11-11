---
paths:
  - "deploy/**"
  - "tools/host/**"
  - "src/anpr/capture.py"
  - "src/anpr/infer.py"
  - "src/anpr/pipeline.py"
---

# Board runtime

- Load `tflite_runtime` with `/usr/lib/libneutron_delegate.so` after `probe-npu` selects it.
- `liblitert_neutron_delegate.so` is the LiteRT delegate. Do not pass it to `tflite_runtime`.
- Search both `/usr/lib` and `/usr/lib/aarch64-linux-gnu` for `*neutron*delegate*.so` and `libtensorflow-lite-neutron-delegate.so`.
- The bench stream is RTSP path `videoCapture` on port 8554. Decode with `rtspsrc protocols=tcp`, `v4l2h264dec`, and `imxvideoconvert_g2d`. `cv2.CAP_GSTREAMER` is the fallback with the same pipeline string.
- `anpr.service` uses `User=anpr` and `Group=video`. `/dev/neutron0` stays group `video` mode `0660`.
- `tools/host/export_int8.py` must refuse to run when `datasets/calibration/images` has fewer than 100 frames.
- Host-side converter versions (`eiq-neutron-sdk`) must match the board Neutron driver; a microcode mismatch means recompile.
