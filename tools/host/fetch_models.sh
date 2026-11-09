#!/usr/bin/env bash
# Download float detector + OCR weights into models/ (gitignored binaries).
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
dest="${root}/models"
mkdir -p "${dest}"

detector_url="https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-640-license-plates-end2end.onnx"
ocr_base="https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates"

fetch() {
  local url="$1"
  local out="$2"
  if [[ -f "${out}" ]]; then
    echo "keep ${out}"
    return
  fi
  echo "get  ${out}"
  curl -fL --retry 3 -o "${out}.partial" "${url}"
  mv "${out}.partial" "${out}"
}

fetch "${detector_url}" "${dest}/plate_detector.onnx"
fetch "${ocr_base}/cct_s_v2_global.onnx" "${dest}/cct_s_v2_global.onnx"
fetch "${ocr_base}/cct_s_v2_global_plate_config.yaml" "${dest}/cct_s_v2_global_plate_config.yaml"
fetch "${ocr_base}/cct_xs_v2_global.onnx" "${dest}/cct_xs_v2_global.onnx"
fetch "${ocr_base}/cct_xs_v2_global_plate_config.yaml" "${dest}/cct_xs_v2_global_plate_config.yaml"

echo "float weights in ${dest}"
echo "INT8: see models/README.md and tools/host/quantize_onnx_int8.py"
