# Model export

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

## neutron-converter

Run `neutron-converter` from the eIQ Toolkit build that matches BSP `6.18.37-2.1.0`. Copy flags from that binary’s `--help`. The tool is not installed on the board.

Read the converter log before copying the `.tflite` file to the board. Keep the model only when the log shows the graph on Neutron. A report that leaves ops on the CPU or XNNPACK means the export failed; do not deploy that file.

The recognizer, when it exists, goes through the same preflight with its own calibration frames.
