# ANPR on i.MX95

Config-driven automatic number plate recognition for the FRDM i.MX95 board. Capture threads publish the latest frame. One inference thread owns the TensorFlow Lite interpreter and the Neutron delegate.

This repository is the application. The BSP stays in the sibling `imx-docker` checkout and its `yocto-builds` tree. See [docs/upstreams.md](docs/upstreams.md) for the git boundaries and [ROADMAP.md](ROADMAP.md) for what is still ahead.

The flashed board checked for this scaffold is `imx95-15x15-lpddr4x-frdm` running `imx-image-full` from Linux `6.18.37-2.1.0`, Python 3.14, and Neutron 3.1.1.

## Layout

```text
src/anpr/                  application package
configs/config.example.json
deploy/                    systemd unit and udev rule
datasets/calibration/      INT8 calibration frames (images are gitignored)
tools/host/export_int8.py  host-side INT8 export preflight
docs/                      architecture, upstreams, model export
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
python -m anpr probe-npu
```

`run` wires stream pipelines and frame slots, then returns. The live capture loop is a later phase.
