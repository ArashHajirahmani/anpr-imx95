---
paths:
  - "**/*.py"
---

# Python

- Package code lives in `src/anpr`. Type public functions. Use stdlib `logging`, not `print`, in library modules.
- Keep the `tflite_runtime` and `gi` imports inside the functions that open the device or the pipeline (also `onnxruntime`/`cv2` if the module must import on a bare dev host).
- One thread owns the TFLite interpreter. Capture code publishes frames into `LatestFrameSlot`.
- Line length 100, ruff rules E, F, I, UP, target py311. Run `ruff check` and `ruff format` before finishing.
- Add or update the matching `tests/test_<module>.py`; tests must run on a host with no NPU, GStreamer, or weights.

```python
# preferred
def open_gstreamer(pipeline: str) -> object:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    return Gst.parse_launch(pipeline)
```
