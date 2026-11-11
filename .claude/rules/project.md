# ANPR project rules

Edit this repository for the application. Leave `imx-docker` and `yocto-builds` unchanged on application tasks.

- Device paths, thresholds, and pipeline strings belong in the JSON config.
- Do not commit `*.pt`, `*.tflite`, `*.onnx`, or `datasets/calibration/images/`.
- INT8 export runs `tools/host/export_int8.py`. It requires at least 100 real calibration frames. Skip the preflight and the export is invalid.
- A neutron-converter/neutron_compiler log that leaves ops on the CPU through XNNPACK means the model is not ready to deploy.

```bash
# valid
python3 tools/host/export_int8.py --weights models/plate.pt

# invalid: yolo export int8=True with no calibration data yaml
```
