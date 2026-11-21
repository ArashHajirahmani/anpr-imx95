# Benchmark: NPU vs CPU, and OCR model size

Measured on the FRDM i.MX95 (7.7 GB RAM) on 2026-11-12/13. Raw per-second samples and the sampler are in `docs/bench/`.

## 1. Full pipeline, NPU vs CPU (100 s each)

Source: `IMG_9475_18if.mp4` (17 fps, 1080x864, 27.6 s loop), 2 OCR workers, same fusion voter.
NPU = INT8 Neutron detector + XS OCR on the NPU. CPU = float ONNX detector + float XS OCR on ORT CPU.
Decode (VPU) is identical in both modes and was not timed separately.

| Metric | NPU | CPU | NPU vs CPU |
|---|---|---|---|
| Detector, mean (median) | 68 ms (68) | 972 ms (970) | 14x faster |
| OCR, mean (median) | 49 ms (40) | 94 ms (88) | 1.9x faster |
| Detector + OCR per frame | ~117 ms | ~1,066 ms | 9x faster |
| FPS, 100 s average | 11.5 | 0.92 | 12x |
| FPS while frames flow | 13.6 | 1.0 | 13.6x |
| Process CPU (% of one core) | 176% (peak 233) | 228% (peak 263) | 23% less |
| CPU time per frame | ~0.15 s | ~2.5 s | 16x less |
| NPU load | 15.5% avg, 100% peak | 0% | |
| A55 temperature mean (max) | 56.1 C (57.1) | 58.3 C (59.9) | rise +1.9 C in both |
| RSS mean (max) | 226 MB (232) | 297 MB (300) | 24% less |

Accuracy vs plate CAKW568 (indicative only):

| Metric | NPU | CPU |
|---|---|---|
| OCR reads | 1,166 | 102 |
| Per-frame exact match | 24.0% | 19.6% |
| Character accuracy | 65.7% | 79.4% |
| Published after voting, exact | 63.8% (328/514) | 24.7% (21/85) |

Caveats: the NPU average includes ~8 s gaps when the bench clip loops. The CPU run started warmer (it ran second). The CPU run saw only ~93 frames (about +-8 points). The voter window spans ~16 s of video at 1 fps but ~1.2 s at 13 fps, so published accuracy is not a pure model comparison. Energy was not measured (temperature only).

## 2. OCR model size, S vs XS (CPU only, ORT, service idle)

110 plate crops from the bench clip (box padded 12%).

| Model | 1 thread | 4 threads | Reads ending KW568 | Exact CAKW568 |
|---|---|---|---|---|
| XS float | 64 ms | 24 ms | 98/110 | 26/110 |
| XS INT8 | 22 ms | 14 ms | 81/110 | 19/110 |
| S float | 633 ms | 310 ms | 75/110 | 25/110 |
| S INT8 | 205 ms | 88 ms | 87/110 | 20/110 |

XS float is ~13x faster than S float. All models mostly miss the faint leading C, so it is a resolution problem, not model size. One clip, one plate, correlated frames.

Gap: S was never run on the NPU (attention blocks and static INT8 of CCT-S do not work there), so there is no S-on-NPU row and no XS-on-NPU row in table 2.
