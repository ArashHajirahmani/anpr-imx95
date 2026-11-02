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

Set `ANPR_LOG_LEVEL=DEBUG` on either unit for one journal line per request.

Publish MP4 files from this PC, then install the same JSON on the board as `/etc/anpr/config.json`:

```bash
tools/host/run-rtsp-server.sh /path/to/videos
```

MediaMTX is not in git. Set `MEDIAMTX_BIN` or place the `mediamtx` binary next to the script. The script prints the board URL from this PC's current address. Video tracks must be H.264.
