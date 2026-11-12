"""Split a YOLO detection head so boxes and scores quantize separately.

Ultralytics-style exports end in ``Concat(boxes_px, sigmoid_scores) -> output0`` with
shape ``[1, 5, N]``. Boxes are in pixels (0..640) and scores are 0..1. After INT8
quantization both share one scale of about 2.6, so every score rounds to 0 and the
detector never fires. Exposing the two Concat inputs as separate graph outputs gives
each its own scale.

`host_decode=True` (default) cuts further back: the outputs are the raw DFL box logits
``[1, 64, N]`` (4 sides x 16 bins) and the scores. Everything between them and the
boxes (softmax over bins, 1x1 conv, anchor arithmetic) then no longer runs as CPU
fallback inside the TFLite graph; the application decodes only the anchors whose score
passes the threshold (`anpr.recognize.decode_dfl_head`).
"""

from pathlib import Path


class SplitError(Exception):
    """The ONNX graph does not end in the expected boxes+scores Concat."""


def split_detection_head(
    source: Path, target: Path, *, host_decode: bool = True
) -> tuple[str, str]:
    """Write ``target`` with two outputs and return their names.

    ``host_decode=False``: (``boxes`` ``[1,4,N]`` pixels, ``scores`` ``[1,1,N]``).
    ``host_decode=True``: (``box_dfl`` ``[1,64,N]`` raw logits, ``scores``).
    """
    import onnx
    from onnx import TensorProto, helper, shape_inference

    model = onnx.load(str(source))
    graph = model.graph
    if len(graph.output) != 1:
        raise SplitError(f"expected one graph output, found {len(graph.output)}")
    producers = {name: node for node in graph.node for name in node.output}
    concat = producers.get(graph.output[0].name)
    if concat is None or concat.op_type != "Concat" or len(concat.input) != 2:
        raise SplitError("graph output is not a two-input Concat")

    inferred = shape_inference.infer_shapes(model)
    shapes = {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in [*inferred.graph.value_info, *inferred.graph.output]
    }
    first, second = concat.input
    by_channels = {}
    for name in (first, second):
        shape = shapes.get(name)
        if not shape or len(shape) != 3:
            raise SplitError(f"cannot infer a [1, C, N] shape for {name}")
        by_channels[shape[1]] = name
    if set(by_channels) != {4, 1}:
        raise SplitError(
            f"expected 4-channel boxes and 1-channel scores, got {sorted(by_channels)}"
        )
    boxes, scores = by_channels[4], by_channels[1]
    count = shapes[boxes][2]
    first_name = "boxes"
    first_tensor = boxes
    first_channels = 4
    if host_decode:
        first_tensor = _dfl_logits(boxes, producers, shapes, count)
        first_name = "box_dfl"
        first_channels = 64

    graph.node.remove(concat)
    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info(
                first_name, TensorProto.FLOAT, [1, first_channels, count]
            ),
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, count]),
        ]
    )
    for node in graph.node:
        for index, name in enumerate(node.output):
            if name == first_tensor:
                node.output[index] = first_name
            elif name == scores:
                node.output[index] = "scores"
        for index, name in enumerate(node.input):
            if name == first_tensor:
                node.input[index] = first_name
            elif name == scores:
                node.input[index] = "scores"
    _prune(graph)
    onnx.checker.check_model(model)
    onnx.save(model, str(target))
    return first_name, "scores"


def _dfl_logits(boxes: str, producers: dict, shapes: dict, count: int) -> str:
    """Find the ``[1, 64, N]`` tensor that feeds the DFL ``Reshape -> [1, 4, 16, N]``."""
    pending = [boxes]
    seen: set[str] = set()
    while pending:
        name = pending.pop()
        node = producers.get(name)
        if node is None or name in seen:
            continue
        seen.add(name)
        if node.op_type == "Reshape" and shapes.get(node.output[0]) == [1, 4, 16, count]:
            logits = node.input[0]
            if shapes.get(logits) == [1, 64, count]:
                return logits
        pending.extend(node.input)
    raise SplitError("no DFL Reshape to [1, 4, 16, N] found behind the box branch")


def _prune(graph: object) -> None:
    """Drop nodes that no graph output depends on."""
    needed = {output.name for output in graph.output}
    keep = []
    for node in reversed(list(graph.node)):
        if any(name in needed for name in node.output):
            keep.append(node)
            needed.update(node.input)
    keep.reverse()
    del graph.node[:]
    graph.node.extend(keep)
    used = {name for node in keep for name in node.input}
    kept_initializers = [init for init in graph.initializer if init.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
