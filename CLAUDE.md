# Claude Code guide: ANPR on i.MX95

@AGENTS.md

The imported `AGENTS.md` is the shared agent map (boundaries, runtime facts, tracking). Everything below is specific to Claude Code. Path-scoped rules are in `.claude/rules/`, workflows in `.claude/skills/`, subagents in `.claude/agents/`.

## What this project is

Config-driven ANPR service for the FRDM i.MX95. `anpr run` captures RTSP, runs plate detector + OCR (CTC-greedy, ROI refine, quality gate, temporal vote) on one inference thread, and publishes the latest JPEG + plate JSON on a Unix socket. `anpr web` is a separate process serving the browser page from that socket. Pure-stdlib package (`dependencies = []`); `tflite_runtime`, `onnxruntime`, `cv2`, `gi` exist only on the board/host environment.

## Source map (`src/anpr/`)

| Module | Role |
| --- | --- |
| `cli.py`, `__main__.py` | `check-config`, `run`, `web`, `probe-npu` entry points |
| `config.py` | JSON config dataclasses + validation (`ConfigError`) |
| `capture.py` | GStreamer pipeline string + `LatestFrameSlot`; `gi` imported lazily |
| `infer.py` | Neutron delegate search/ranking (`probe_delegates`) |
| `pipeline.py` | Stream threads + the single inference thread |
| `recognize.py` | Detector, ROI refine, quality, enhancer, OCR decode |
| `roi_refine.py`, `quality.py` | Crop L/R snap, perspective; blur/dark/bright gates |
| `vote.py`, `plate_match.py`, `plates.py` | Temporal voting, alias matching, plate text helpers |
| `results.py` | `ResultStore`, sticky history |
| `api.py`, `ipc.py`, `web.py`, `static/index.html` | Unix-socket API and browser page |

Tests mirror modules in `tests/test_<module>.py`.

## Commands

```bash
.venv/bin/ruff check src tests tools
.venv/bin/ruff format --check src tests tools
.venv/bin/pytest
.venv/bin/python -m anpr check-config --config configs/config.example.json
```

Run all four before calling a Python change done (the `anpr-verify` skill does this). `probe-npu` fails on a dev host by design.

## Working rules

- Thresholds, device paths, pipeline strings go in the JSON config (and `configs/config.example.json`), never hard-coded. When you add a config key, update `config.py`, the sample config, `tests/test_config.py`, and the README/architecture docs together.
- Behavior changes in recognition must update `docs/architecture.md`; export/quantization changes must update `docs/model-export.md` and `models/README.md`.
- Never commit weights, calibration images, videos, or MediaMTX files (all gitignored). `auto.crt`/`auto.key` in the repo root are MediaMTX leftovers: never stage them.
- Do not touch `imx-docker` or `yocto-builds` on application tasks.
- Tick `ROADMAP.md` only for things verified on hardware or by tests.
- Commits: small, imperative subject ending in a period (match `git log`). Work on a branch, not `main`.

## Current state (update when it changes)

- Branch `wip/ocr-vote-roi` carries uncommitted work: ROI/vote tuning, the Neutron TFLite detector path (`convert_neutron_tflite.py`, `onnx_split_head.py`, split box/score outputs), and the staged pipeline (`staged.py`, `publisher.py`, `service.ocr_workers`).
- Detector on the NPU: the head must be split into `boxes` and `scores` outputs (one `[1,5,N]` tensor quantizes every score to 0). Always compare NPU vs float detections on the calibration frames before deploying. See `docs/model-export.md`.
- OCR stays on CPU (static INT8 of the CCT model diverges from float; weights-only quantization tracks it far better). The NPU cannot run it.
- Staged pipeline: 1 detector thread (owns the interpreter) + `service.ocr_workers` (default 4) OCR workers + in-order publisher. The detector is compiled to end at raw DFL logits + scores and decoded in numpy (`decode_dfl_head`), so `det_ms` is about 43 ms. Board throughput with 4 workers: about 16 fps while frames flow (single-threaded was about 3.4); 3 workers gave about 9.4, 5 saturate the CPU. `/v1/latest` carries a backend-measured `fps` (3 s window) that the UI shows; the old UI estimate measured the 500 ms poll interval (about 2).
- Ahead: OCR throughput (a faster OCR model, or OCR on fewer pixels), then ROADMAP.md Phase 5 closure and a commit of the WIP.

## Tooling

- Host tool install location: `/opt` was full; the SDK is symlinked into `/opt`, but install other large host tools (eIQ Toolkit / `neutron-converter`, MediaMTX, toolchains, `gh` if bulky) under `/run/media/arash/Projects/iMX95/host-tools/`, never `/opt`. The Yocto SDK is at `/run/media/arash/Projects/iMX95/host-tools/fsl-imx-wayland/6.18-wrynose` (source its `environment-setup-armv8-2a-poky-linux` for cross builds; root-owned). Small packages may use default system paths. Keep all of this outside the repo.

- GitHub: `gh` CLI is not installed on this machine. Project `.mcp.json` registers the GitHub MCP server; it needs `GITHUB_PERSONAL_ACCESS_TOKEN` in the environment.
- Board and bench addresses, accounts, and remote names are kept out of git: see `CLAUDE.local.md` (gitignored). Outward actions (pushing, opening PRs, publishing RTSP, changing the board) need the user's go-ahead; read-only ssh checks on the board are fine.
- Privacy: never write credentials, personal names, emails, or new IP addresses into tracked files, commits, or PRs. Put them in `CLAUDE.local.md`.
