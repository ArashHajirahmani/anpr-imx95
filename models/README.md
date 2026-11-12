# Models (local cache)

Weight files under this directory are **not committed**. They are downloaded or
regenerated on the host, then copied to `/opt/anpr/models` on the board.

`configs/config.example.json` expects these names:

| File | Role | Source |
| --- | --- | --- |
| `plate_detector.onnx` | Float detector (640×640) | [open-image-models](https://github.com/ankandrew/open-image-models) YOLOv9-t 640 |
| `plate_detector.int8.onnx` | INT8 detector (ORT CPU) | Local quantize (see below) |
| `plate_detector.neutron.tflite` | Neutron NPU detector | `tools/host/convert_neutron_tflite.py` |
| `cct_s_v2_global.onnx` | Float OCR (sample default) | [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) CCT-S v2 global |
| `cct_s_v2_global.int8.onnx` | INT8 OCR (ORT CPU under npu) | Local quantize |
| `cct_xs_v2_global.onnx` | Optional faster float OCR | fast-plate-ocr CCT-XS v2 global |
| `cct_xs_v2_global.int8.onnx` | Optional faster INT8 OCR | Local quantize |
| `cct_xs_v2_global.neutron.tflite` | XS OCR on the NPU (`xs_npu` profile) | `tools/host/convert_neutron_ocr.py` |
| `*_plate_config.yaml` | OCR charset / size metadata | Same release as the matching OCR ONNX |

## Download float weights

From the repo root:

```bash
mkdir -p models
cd models

# Detector → rename to the path used in config
curl -fL -o plate_detector.onnx \
  https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-640-license-plates-end2end.onnx

# OCR S (sample default) + plate config
curl -fL -O https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v2_global.onnx
curl -fL -O https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_s_v2_global_plate_config.yaml

# Optional OCR XS
curl -fL -O https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_xs_v2_global.onnx
curl -fL -O https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/cct_xs_v2_global_plate_config.yaml
```

Or use the helper:

```bash
tools/host/fetch_models.sh
```

## NPU detector + INT8 OCR (`service.runtime=npu`)

Needs ≥100 real calibration frames in `datasets/calibration/images` (gitignored).
Details: [docs/model-export.md](../docs/model-export.md).

```bash
# Neutron TFLite detector (match eiq-neutron-sdk to board driver, e.g. 3.2.1)
.venv/bin/python tools/host/convert_neutron_tflite.py \
  --onnx models/plate_detector.onnx \
  --images datasets/calibration/images \
  --output models/plate_detector.neutron.tflite

# INT8 OCR for ORT CPU under npu mode
.venv/bin/python tools/host/quantize_onnx_int8.py \
  --detector models/plate_detector.onnx \
  --ocr models/cct_s_v2_global.onnx \
  --images datasets/calibration/images \
  --output-dir models
```

Copy `plate_detector.neutron.tflite` and `*.int8.onnx` to `/opt/anpr/models/…`.
Point `detector.npu_path` at the `.tflite`, `detector.delegate_path` at
`/usr/lib/libneutron_delegate.so`, and `recognizer.npu_path` at the INT8 OCR.

## Do not commit

`*.onnx`, `*.pt`, `*.tflite`, and other files under `models/` stay local. Only this
README is tracked.
