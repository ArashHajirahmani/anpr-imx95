# ANPR agent map

This repo is the ANPR application for the FRDM i.MX95. Read it before editing.

## Boundaries

- Change application code, config samples, deploy templates, and host export tooling here.
- Leave `imx-docker` and `yocto-builds` alone during application tasks. The BSP is a separate `repo` checkout. See [docs/upstreams.md](docs/upstreams.md).
- Do not commit `*.pt`, `*.tflite`, or files under `datasets/calibration/images/`.

## Runtime facts

Board check against `imx-image-full` on `imx95-15x15-lpddr4x-frdm`:

- Import `tflite_runtime`. The TFLite Neutron library is `/usr/lib/libneutron_delegate.so`.
- Report `/usr/lib/liblitert_neutron_delegate.so` and do not pass it to `tflite_runtime`.
- One interpreter thread. Capture threads only publish the latest frame.
- H.264 decode is `v4l2h264dec` then `imxvideoconvert_g2d`. Details are in [docs/architecture.md](docs/architecture.md).
- INT8 export goes through `tools/host/export_int8.py` and needs 100 real calibration frames. See [docs/model-export.md](docs/model-export.md).

Install paths used later: `/opt/anpr` and `/etc/anpr/config.json`. Device paths and thresholds belong in that JSON file.

## Tracking

Phase status is in [ROADMAP.md](ROADMAP.md). The live capture loop, Neutron inference, and the board install are not part of the scaffold.
