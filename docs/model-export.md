# Model export

Float ONNX detector and OCR weights are downloaded into `models/` (gitignored).
INT8 ONNX for Neutron is produced on the host with calibration frames. See
[models/README.md](../models/README.md) and `tools/host/fetch_models.sh`.

## ONNX INT8 (board `service.runtime=npu`)

```bash
tools/host/fetch_models.sh
.venv/bin/python tools/host/quantize_onnx_int8.py \
  --detector models/plate_detector.onnx \
  --ocr models/cct_s_v2_global.onnx \
  --images datasets/calibration/images \
  --output-dir models
```

Copy `*.onnx` to `/opt/anpr/models` on the board.

## TFLite / neutron-converter (legacy path)

Neutron compilation on the host needs a fully quantized INT8 TFLite model. `yolo export format=tflite int8=True` without a representative image set can leave layers unquantized. Those layers stay on the CPU through XNNPACK, and `/dev/neutron0` does not run the heavy ops.

## Calibration set

Put at least 100 `.jpg`, `.jpeg`, or `.png` frames in `datasets/calibration/images`. Use frames from the cameras this service will see (gate, parking lane, the same resolution and lighting). A COCO sample does not qualify. The image files are gitignored.

`tools/host/export_int8.py` counts those files and exits before calling Ultralytics when the count is under 100. It writes a temporary data yaml whose `train` and `val` entries point at that directory, then runs:

```bash
yolo export model=<weights> format=tflite int8=True data=<yaml>
```

```bash
python3 tools/host/export_int8.py --weights /path/to/plate-detector.pt
```

## When to re-quantize

INT8 scales track the calibration distribution. Tuning `recognizer.quality`, `recognizer.enhancer`, or confidence floors does **not** require a new export.

| Change | Re-quantize? |
| --- | --- |
| New camera (sensor, lens, FOV, or stream compression) | Yes — collect ≥100 frames from that camera and re-export |
| Day / night / weather on the same camera | Not for every shift; include those looks in the original ≥100-frame set. If night or rain was never calibrated, add frames and re-export |
| Threshold or enhancer config only | No |

## neutron-converter

Run `neutron-converter` from the eIQ Toolkit build that matches BSP `6.18.37-2.1.0`. Copy flags from that binary’s `--help`. The tool is not installed on the board.

Read the converter log before copying the `.tflite` file to the board. Keep the model only when the log shows the graph on Neutron. A report that leaves ops on the CPU or XNNPACK means the export failed; do not deploy that file.

The recognizer, when it exists, goes through the same preflight with its own calibration frames.
