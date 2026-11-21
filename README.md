# ANPR on i.MX95

Automatic number plate recognition on the NXP FRDM i.MX95 board. The plate detector and the OCR run on the eIQ Neutron NPU. Recognition combines evidence across frames, so one bad frame does not decide the plate.

[![ANPR demo on the i.MX95: live web page with the detected plate CAKW568](docs/media/anpr-demo.gif)](docs/media/anpr-demo.mp4)

*Click the preview for the full video with sound ([docs/media/anpr-demo.mp4](docs/media/anpr-demo.mp4)).*

## Results

Same board, same video, 100 s per run. Details and raw samples are in [docs/benchmark-npu-vs-cpu.md](docs/benchmark-npu-vs-cpu.md).

| | NPU | CPU | |
| --- | ---: | ---: | --- |
| Plate detector (YOLOv9-t) | **68 ms** | 972 ms | 14× faster |
| OCR (CCT-XS) | **49 ms** | 94 ms | 1.9× faster |
| Whole pipeline | **11.5 FPS** | 0.92 FPS | 12× |
| CPU time per frame | **0.15 s** | 2.5 s | 16× less |

NPU means the INT8 Neutron detector plus the XS OCR on the NPU. CPU means the float ONNX models on the Cortex-A55 cores. Source: a 17 fps H.264 clip with one key frame per second; every frame is decoded. 2 OCR workers.

## How it works

```text
IP camera (RTSP, H.264)
  -> capture thread      VPU decode (v4l2h264dec) + G2D convert; keeps only the newest frame
  -> detector thread     letterbox 640x640, YOLOv9-t INT8 on the NPU, box decode in numpy
  -> OCR workers (1-8)   ROI refine, quality gates, enhancer, CCT OCR (NPU conv + linear, CPU attention)
  -> publisher           frame order, multi-frame evidence fusion, release at posterior 0.9
  -> anpr run            latest JPEG + plate JSON on a Unix socket
  -> anpr web            browser page on port 8080 (separate process)
```

- Stale frames are dropped, not queued, so latency does not build up.
- One thread owns the detector interpreter. OCR workers share one OCR session.
- The recognition process opens no TCP port. Only `anpr web` listens.
- The page footer reports the device each stage really runs on.

Full details are in [docs/architecture.md](docs/architecture.md). The model walkthrough is in [docs/learning-guide.md](docs/learning-guide.md).

### What made it work on the NPU

- **Split detector head.** One `[1,5,N]` output quantized every score to 0. Separate box and score outputs keep their own INT8 scales.
- **DFL decode on the host.** The compiled detector ends at raw DFL logits and scores. Numpy decodes only the anchors above the threshold (detector about 43 ms on the board).
- **OCR split between NPU and CPU.** The XS OCR conv stem and linear layers run on the NPU. Attention stays on the CPU. INT8 ADD gave wrong results on the NPU, so ADD stays float.
- **Per-slot decoding.** The OCR has 10 character slots, not a CTC alignment. Repeated characters such as `077` and `DJXX` survive.
- **Evidence fusion.** Per-slot probabilities from up to 16 frames score a pool of candidate texts. Frames the OCR is sure about weigh more. A few clear frames beat many weak ones.

Export steps are in [docs/model-export.md](docs/model-export.md) and [models/README.md](models/README.md).

## Layout

```text
src/anpr/                       application package
configs/config.example.json     board config; videoCapture RTSP source
deploy/                         systemd units and udev rule
datasets/calibration/           INT8 calibration frames (images are gitignored)
datasets/videos/                H.264 bench MP4s for RTSP (files are gitignored)
docs/                           architecture, model export, benchmark, learning guide
docs/bench/                     benchmark sampler and raw per-second samples
docs/media/                     demo video
models/                         local weights (gitignored); see models/README.md
tools/host/export_int8.py       host-side INT8 export preflight
tools/host/fetch_models.sh      download the float detector and OCR into models/
tools/host/quantize_onnx_int8.py   INT8 ONNX
tools/host/onnx_split_head.py   split the detector head into box and score outputs
tools/host/convert_neutron_tflite.py   detector to a Neutron TFLite
tools/host/convert_neutron_ocr.py      XS OCR to a Neutron TFLite
tools/host/neutron_graph_report.py     which ops land on the NPU
tools/host/convert_video.sh     phone video to a bench H.264 MP4
tools/host/run-rtsp-server.sh   publish MP4 files for the board
```

The board checked here is `imx95-15x15-lpddr4x-frdm` running `imx-image-full` from Linux `6.18.37-2.1.0`, Python 3.14 and Neutron 3.1.1. The BSP stays in the sibling `imx-docker` checkout and its `yocto-builds` tree. See [docs/upstreams.md](docs/upstreams.md) and [ROADMAP.md](ROADMAP.md).

Install paths on the board are `/opt/anpr` and `/etc/anpr/config.json`.

## Host checks

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
# Host export tools (NXP eIQ Neutron SDK, onnx, onnxruntime, opencv, netron):
.venv/bin/pip install -r requirements-host.txt
.venv/bin/ruff check src tests tools
.venv/bin/ruff format --check src tests tools
.venv/bin/pytest
.venv/bin/python -m anpr check-config --config configs/config.example.json
```

`probe-npu` searches the machine it runs on. On a development host the Neutron delegate is absent and the command exits with an error. On the board it should select `/usr/lib/libneutron_delegate.so`.

## Commands

```bash
python -m anpr check-config --config configs/config.example.json
python -m anpr run --config configs/config.example.json
python -m anpr web --config configs/config.example.json
python -m anpr probe-npu
```

`run` stays up and reads each enabled stream until SIGTERM. It publishes the latest JPEG and plate JSON on the Unix socket in `service.api_socket`. The sample stream is `rtsp://10.0.0.117:8554/videoCapture`.

`web` is a separate process. Open `http://<board>:8080/` for the live frame, the FPS, and the recent plates. The reset button clears history and vote memory. `web` keeps serving if `run` is down.

```bash
journalctl -u anpr -u anpr-web -f
curl --unix-socket /run/anpr/api.sock http://localhost/v1/latest
```

Set `ANPR_LOG_LEVEL=DEBUG` on either unit for one journal line per request (includes crop quality rejects).

## Throughput settings

| Config | Sample | Effect |
| --- | --- | --- |
| `service.runtime` | `cpu` | `npu` loads the Neutron TFLite models through `detector.delegate_path` |
| `service.ocr_workers` | 4 | OCR worker threads (1-8). More workers raise FPS until the CPU saturates (about 5 on the CPU OCR path); a `.tflite` OCR shares one interpreter, so 2 is enough there |
| `service.inference_fps_limit` | 2 | Cap on how often the detector takes a frame. The board uses 30 |
| `web.poll_seconds` | 0.02 | Minimum gap between page polls |

## Recognition gates

After the detector crops a plate, these checks run. Thresholds live in `/etc/anpr/config.json` (see `configs/config.example.json`).

| Gate | Config | Default behavior |
| --- | --- | --- |
| Minimum plate width | `detector.min_plate_width` | Skip boxes narrower than 32 px before OCR (`0` turns it off) |
| Crop sharpness | `recognizer.quality.min_sharpness` | Skip crops below 60 (Laplacian variance of the raw crop at 128x64) before OCR; `0` turns it off |
| Image quality | `recognizer.quality` | Dark, bright, or low-contrast crops are only logged |
| ROI refine | `recognizer.roi_refine` | L/R border snap (−20%…+20%) + 5% pad; perspective off by default |
| Enhancer | `recognizer.enhancer` | On in the sample: CLAHE, gamma 1.1, mild sharpen; never overrides a reject gate |
| Evidence fusion | `recognizer.vote` | 16-frame window; release when the posterior reaches `release_posterior` (0.9); the posterior is the published confidence |
| Confidence floors | `recognizer.min_char_confidence`, `min_confidence` | Applied to the fused read before publishing (0.70; the NPU path subtracts 0.005) |

Accepted plate JSON includes `characters: [{ "char", "confidence" }, ...]`. Uncertain plates are not published. Plates too small for the OCR to resolve every glyph are a resolution limit: the bench plate CAKW568 reads AKW568 below about 130 px box width. When to re-quantize after a camera or lighting change is in [docs/model-export.md](docs/model-export.md).

## Bench stream

Publish MP4 files from the host, then install the same JSON on the board as `/etc/anpr/config.json`:

```bash
tools/host/run-rtsp-server.sh datasets/videos
tools/host/run-rtsp-server.sh datasets/videos/IMG_9475.mp4
```

MediaMTX is not in git. Put the executable at `tools/host/mediamtx`, next to `tools/host/run-rtsp-server.sh`. The other release files (`mediamtx.yml`, `LICENSE`, `auto.crt`, `auto.key`) can stay in that directory; they are gitignored. `MEDIAMTX_BIN` or a `mediamtx` binary on `PATH` also works. The script prints the board URL from the host's current address.

Video tracks must be H.264. Phone `.MOV` files are often HEVC. Convert them with `tools/host/convert_video.sh INPUT [OUTPUT]` (H.264, 18 fps, one key frame per second, no B-frames, no audio).

## Switching models

`recognizer.models` (and `detector.models`) name the available model files; `recognizer.model` picks the active one. Each entry has `path` (used with `service.runtime=cpu`) and `npu_path` (used with `runtime=npu`). The backend follows the file type under `npu`: `.onnx` runs on ORT CPU, a Neutron `.tflite` runs on the NPU. Change `"model": "s"` to `"xs"` or `"xs_npu"` in `/etc/anpr/config.json` and run `systemctl restart anpr`. The page footer shows the device each stage runs on and the active profile.

| Profile | OCR model | Runs on |
| --- | --- | --- |
| `s` | CCT-S, float ONNX (`cpu`) or INT8 ONNX (`npu`) | CPU; it does not fit the NPU |
| `xs` | CCT-XS, float ONNX (`cpu`) or INT8 ONNX (`npu`) | CPU |
| `xs_npu` | CCT-XS, float ONNX (`cpu`) or Neutron TFLite (`npu`) | NPU conv + linear, CPU attention |
