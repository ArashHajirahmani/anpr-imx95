# Model export

Float ONNX detector and OCR weights are downloaded into `models/` (gitignored).
Board `service.runtime=npu` needs a **Neutron-compiled TFLite detector** plus an
INT8 ONNX OCR. Plain ORT INT8 ONNX is not enough for the detector: Neutron EP
only accelerates `NeutronGraph` / MatMulInteger-style ops, and YOLO is Conv-heavy.

## NPU detector (TFLite + neutron-compiler)

Host packages from https://eiq.nxp.com/repository/ must match the board Neutron
driver. On LF `6.18-wrynose` the board reports driver `3.2.1`:

```bash
pip install -r requirements-host.txt   # pins eiq-neutron-sdk==3.2.1 and eiq-onnx2tflite
```

Needs ≥100 real calibration frames in `datasets/calibration/images`:

```bash
.venv/bin/python tools/host/convert_neutron_tflite.py \
  --onnx models/plate_detector.onnx \
  --images datasets/calibration/images \
  --output models/plate_detector.neutron.tflite \
  --target imx95
```

The script first splits the detector head (`tools/host/onnx_split_head.py`) so
`boxes` (pixels) and `scores` (0..1) are separate outputs. Left as one `[1, 5, N]`
tensor they share one INT8 scale of about 2.6 and every score quantizes to 0: the
model compiles, loads on the NPU, runs fast, and never detects a plate.
By default (`host_decode`) the cut goes further back, to the raw DFL box logits
`[1, 64, N]` and the scores, so the softmax, 1x1 conv, and anchor arithmetic are not
compiled as CPU fallback ops (the compiled graph shrinks from 38 operators to 16).
`PlateReader` dequantizes both outputs and decodes only the anchors above the score
threshold in numpy (`decode_dfl_head`; verified against the ONNX head to 6e-5 px).
Detector time on the board fell from about 110 ms to about 43 ms in the live service.

Always compare against the float model before deploying: run the 100 calibration
frames through both and require similar detections (float 48/100, NPU 43/100 on
the bench set). A max score of exactly 0 on every frame means the head was not split.

The script runs `onnx2tflite` → `tflite_profiler` → `tflite_quantizer`
(DIV/SOFTMAX left float) → `neutron_compiler --target imx95`. Keep the model
only when the compiler log shows a high conversion ratio and at least one
`NeutronGraph`. A microcode version mismatch against the board driver means
recompile with the matching `eiq-neutron-sdk` version.

Copy `plate_detector.neutron.tflite` to `/opt/anpr/models/` and set
`detector.npu_path` / `detector.delegate_path` to that file and
`/usr/lib/libneutron_delegate.so`. Verify with `neutron_perf.sh` while
inferring: NPU load should rise above 0%.

## OCR INT8 ONNX (CPU under `runtime=npu`)

```bash
tools/host/fetch_models.sh
.venv/bin/python tools/host/quantize_onnx_int8.py \
  --detector models/plate_detector.onnx \
  --ocr models/cct_s_v2_global.onnx \
  --images datasets/calibration/images \
  --output-dir models
```

Copy `cct_*_global.int8.onnx` to the board and point `recognizer.npu_path` at
it. OCR stays on ORT CPU for now; Neutron compilation of the CCT graph is not
wired yet.

## Calibration set

Put at least 100 `.jpg`, `.jpeg`, or `.png` frames in `datasets/calibration/images`.
Use frames from the cameras this service will see (gate, parking lane, the same
resolution and lighting). A COCO sample does not qualify. The image files are
gitignored.

## When to re-quantize

INT8 scales track the calibration distribution. Tuning `recognizer.quality`,
`recognizer.enhancer`, or confidence floors does **not** require a new export.

| Change | Re-quantize? |
| --- | --- |
| New camera (sensor, lens, FOV, or stream compression) | Yes — collect ≥100 frames from that camera and re-export |
| Day / night / weather on the same camera | Not for every shift; include those looks in the original ≥100-frame set. If night or rain was never calibrated, add frames and re-export |
| Threshold or enhancer config only | No |

## Why plain INT8 ONNX looked like “npu” but stayed on CPU

`service.runtime=npu` only selects INT8 paths and registers Neutron EP. On this
BSP, Neutron EP falls back to “Only NeutronGraph op will be assigned to NPU”
when its 2 GiB CMA stack alloc fails (default 4×512 MiB; FRDM CMA is ~960 MiB).
Even with CMA fixed (`NEUTRON_CMA_512SLOTS=1`), YOLO Convs are not claimed —
only MatMulInteger / QLinearMatMul / NeutronGraph are. Offline
`neutron-compiler` insertion of `NeutronGraph` into TFLite is required for the
detector.

## Inspect which ops run on the NPU

Open the compiled model in Netron: `netron models/plate_detector.neutron.tflite`. Fused `NeutronGraph` nodes run on the NPU; any other node stays on the CPU.
