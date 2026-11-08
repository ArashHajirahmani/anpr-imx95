# ANPR on i.MX95

Config-driven automatic number plate recognition for the FRDM i.MX95 board. Capture threads publish the latest frame. One inference thread owns the TensorFlow Lite interpreter and the Neutron delegate.

This repository is the application. The BSP stays in the sibling `imx-docker` checkout and its `yocto-builds` tree. See [docs/upstreams.md](docs/upstreams.md) for the git boundaries and [ROADMAP.md](ROADMAP.md) for what is still ahead.

The flashed board checked for this scaffold is `imx95-15x15-lpddr4x-frdm` running `imx-image-full` from Linux `6.18.37-2.1.0`, Python 3.14, and Neutron 3.1.1.

## Layout

```text
src/anpr/                     application package
configs/config.example.json   board config; videoCapture RTSP source
deploy/                       systemd units and udev rule
datasets/calibration/         INT8 calibration frames (images are gitignored)
datasets/videos/              H.264 bench MP4s for RTSP (files are gitignored)
tools/host/export_int8.py     host-side INT8 export preflight
tools/host/run-rtsp-server.sh publish MP4 files for the board
docs/                         architecture, upstreams, model export
```

Install paths on the board, used later, are `/opt/anpr` and `/etc/anpr/config.json`.

## Host checks

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
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

`run` stays up and reads each enabled stream until SIGTERM. It also publishes the latest JPEG and plate JSON on the Unix socket in `service.api_socket`. The sample stream is `rtsp://10.0.0.117:8554/videoCapture`.

`web` is a separate process. Open `http://<board>:8080/` for the live frame and the last four plates. It keeps serving if `run` is down.

```bash
journalctl -u anpr -u anpr-web -f
curl --unix-socket /run/anpr/api.sock http://localhost/v1/latest
```

Set `ANPR_LOG_LEVEL=DEBUG` on either unit for one journal line per request (includes crop quality rejects).

## Recognition gates

After YOLO crops a plate, the runtime applies three independent checks. Device thresholds live in `/etc/anpr/config.json` (see `configs/config.example.json`).

| Gate | Config | Default behavior |
| --- | --- | --- |
| Image quality | `recognizer.quality` | Reject blurry, dark, bright, or low-contrast crops before OCR |
| Optional enhancer | `recognizer.enhancer` | Off by default; CLAHE / gamma / mild sharpen; never overrides reject gates |
| Per-character confidence | `recognizer.min_char_confidence` | Reject the whole plate if any kept character is below the floor |
| Mean confidence | `recognizer.min_confidence` | Existing mean softmax floor (NPU subtracts 0.005) |

Accepted plate JSON includes `characters: [{ "char", "confidence" }, ...]`. Uncertain plates are not published. Details are in [docs/architecture.md](docs/architecture.md). When to re-quantize after a camera or lighting change is in [docs/model-export.md](docs/model-export.md).

Publish MP4 files from this PC, then install the same JSON on the board as `/etc/anpr/config.json`:

```bash
tools/host/run-rtsp-server.sh datasets/videos
tools/host/run-rtsp-server.sh datasets/videos/IMG_9475.mp4
```

MediaMTX is not in git. Put the executable at `tools/host/mediamtx`, next to `tools/host/run-rtsp-server.sh`. The other release files (`mediamtx.yml`, `LICENSE`, `auto.crt`, `auto.key`) can stay in that directory; they are gitignored. `MEDIAMTX_BIN` or a `mediamtx` binary on `PATH` also works. The script prints the board URL from this PC's current address. Video tracks must be H.264. Phone `.MOV` files are often HEVC; convert before dropping them in `datasets/videos`.
