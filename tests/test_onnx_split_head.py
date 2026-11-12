from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper  # noqa: E402

from onnx_split_head import SplitError, split_detection_head  # noqa: E402


def _head_model(path: Path, boxes_channels: int = 4) -> None:
    nodes = [
        helper.make_node("Identity", ["x"], ["boxes_px"]),
        helper.make_node("Sigmoid", ["x_scores"], ["scores"]),
        helper.make_node("Concat", ["boxes_px", "scores"], ["output0"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "head",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, boxes_channels, 6]),
            helper.make_tensor_value_info("x_scores", TensorProto.FLOAT, [1, 1, 6]),
        ],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, boxes_channels + 1, 6])],
    )
    onnx.save(helper.make_model(graph), str(path))


def test_split_exposes_boxes_and_scores(tmp_path: Path) -> None:
    source = tmp_path / "in.onnx"
    target = tmp_path / "out.onnx"
    _head_model(source)
    assert split_detection_head(source, target, host_decode=False) == ("boxes", "scores")
    graph = onnx.load(str(target)).graph
    assert [output.name for output in graph.output] == ["boxes", "scores"]
    assert all(node.op_type != "Concat" for node in graph.node)


def test_split_rejects_unexpected_head(tmp_path: Path) -> None:
    source = tmp_path / "in.onnx"
    _head_model(source, boxes_channels=3)
    with pytest.raises(SplitError):
        split_detection_head(source, tmp_path / "out.onnx", host_decode=False)


def _dfl_model(path: Path) -> None:
    count = 8
    nodes = [
        helper.make_node("Identity", ["x"], ["logits"]),
        helper.make_node(
            "Constant",
            [],
            ["shape"],
            value=helper.make_tensor("shape", TensorProto.INT64, [4], [1, 4, 16, count]),
        ),
        helper.make_node("Reshape", ["logits", "shape"], ["bins"]),
        helper.make_node("ReduceSum", ["bins"], ["boxes_px"], axes=[2], keepdims=0),
        helper.make_node("Sigmoid", ["x_scores"], ["scores"]),
        helper.make_node("Concat", ["boxes_px", "scores"], ["output0"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "dfl",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64, count]),
            helper.make_tensor_value_info("x_scores", TensorProto.FLOAT, [1, 1, count]),
        ],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, count])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)]), str(path))


def test_split_host_decode_exposes_dfl_logits_and_prunes_the_tail(tmp_path: Path) -> None:
    source = tmp_path / "in.onnx"
    target = tmp_path / "out.onnx"
    _dfl_model(source)
    assert split_detection_head(source, target) == ("box_dfl", "scores")
    graph = onnx.load(str(target)).graph
    assert [output.name for output in graph.output] == ["box_dfl", "scores"]
    op_types = {node.op_type for node in graph.node}
    assert not op_types & {"Reshape", "ReduceSum", "Concat"}
    assert "Sigmoid" in op_types


def test_split_host_decode_requires_a_dfl_reshape(tmp_path: Path) -> None:
    source = tmp_path / "in.onnx"
    _head_model(source)
    with pytest.raises(SplitError):
        split_detection_head(source, tmp_path / "out.onnx")
