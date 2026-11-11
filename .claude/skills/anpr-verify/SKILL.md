---
name: anpr-verify
description: Run the host lint, format, test, and config checks for the ANPR repo and report results. Use before finishing any Python, config, or test change, or when asked to verify/check the repo.
allowed-tools: Bash(.venv/bin/ruff *) Bash(.venv/bin/pytest *) Bash(.venv/bin/python -m anpr check-config *) Bash(git status *) Bash(git diff *)
---

# Verify the ANPR repo

Run from the repo root, in order, and report each result faithfully:

```bash
.venv/bin/ruff check src tests tools
.venv/bin/ruff format --check src tests tools
.venv/bin/pytest -q
.venv/bin/python -m anpr check-config --config configs/config.example.json
```

Notes:
- If `.venv` is missing: `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`.
- `tests/test_api.py::test_api_serves_metadata_and_jpeg` is known flaky. If it is the only failure, rerun that file 3 times, report the flake, and do not mask a real failure in it.
- If `configs/config.example.json` changed, also confirm `tests/test_config.py` and README/architecture docs still agree with it.
- Do not run `probe-npu` on the host; it errors by design off-board.
- Finish with `git status --short` and flag any `*.pt`, `*.tflite`, `*.onnx`, calibration image, or `auto.crt`/`auto.key` that would be staged.
