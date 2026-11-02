# Roadmap

Status of the i.MX95 ANPR service. Phase 0 is this repository scaffold.

## Phase 0 — Scaffold

- [x] Single application repo, separate from the Yocto `repo` checkout
- [x] Agent map, rules, and the `anpr-target` skill
- [x] Config loader, latest-frame slot, capture string builder, delegate ranking
- [x] INT8 export preflight that requires 100 calibration frames
- [x] systemd and udev templates
- [x] Live RTSP capture loop for the videoCapture bench stream
- [ ] Neutron inference on a converted model
- [ ] Board install under `/opt/anpr`

## Phase 1 — INT8 export

Run `tools/host/export_int8.py` once a plate-trained checkpoint and at least 100 real gate or parking frames are in `datasets/calibration/images`. Then run `neutron-converter` from the eIQ release that matches BSP `6.18.37-2.1.0`. Keep the model only when the converter log shows the graph on Neutron. A log that leaves ops on the CPU through XNNPACK means the export failed.

## Phase 2 — Board probe

Run `python -m anpr probe-npu` on the board and write the selected delegate path into the installed config. Libraries and `v4l2h264dec` were confirmed on `10.0.0.165`.

## Phase 3 — Capture and detector

Run one capture thread per enabled stream, drop stale frames, and detect plates at the configured inference rate.

## Phase 4 — Recognizer

Add a plate recognizer and charset. Export that model with its own calibration set through the same INT8 preflight.

## Browser

`anpr run` serves plate JSON and JPEG frames on a Unix socket. `anpr web` is a separate process and shows the latest frame, the plate, and the last four recognitions.

## Phase 5 — Install

Install `/opt/anpr`, `/etc/anpr/config.json`, the `anpr` user, the udev rule, `anpr.service`, and `anpr-web.service`. A `meta-anpr` Yocto layer comes only after that install is known.
