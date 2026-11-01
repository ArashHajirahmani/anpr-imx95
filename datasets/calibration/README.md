# INT8 calibration frames

`tools/host/export_int8.py` reads images from `datasets/calibration/images`.

- Add at least 100 `.jpg`, `.jpeg`, or `.png` files before an INT8 export.
- Use real frames from the gate or parking cameras this service will run on.
- A COCO sample or any other generic set is not a stand-in.
- Image files in `images/` are gitignored. This README is the tracked part of the directory.

The Neutron converter accepts a fully quantized INT8 model. Export without this set can leave layers on the CPU through XNNPACK.
