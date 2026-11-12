#!/usr/bin/env python3
"""List which operators of a Neutron-compiled TFLite model run on the NPU or the CPU.

`NeutronGraph` operators execute on the NPU. Everything else is a CPU fallback.
For compiler-side detail run `neutron_compiler ... --dump-graphs` and read its
"Overall statistics" block (operator conversion ratio, NPU latency estimate).

  .venv/bin/python tools/host/neutron_graph_report.py models/plate_detector.neutron.tflite
"""

import argparse
from collections import Counter
from pathlib import Path


def report(path: Path) -> str:
    from tflite import Model
    from tflite.BuiltinOperator import BuiltinOperator

    names = {value: key for key, value in vars(BuiltinOperator).items() if not key.startswith("_")}
    model = Model.GetRootAsModel(path.read_bytes(), 0)
    graph = model.Subgraphs(0)

    def shape(index: int) -> list[int]:
        tensor = graph.Tensors(index)
        return [int(d) for d in tensor.ShapeAsNumpy()] if tensor.ShapeLength() else []

    def name_of(operator: object) -> str:
        code = model.OperatorCodes(operator.OpcodeIndex())
        builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
        if builtin == BuiltinOperator.CUSTOM:
            return code.CustomCode().decode()
        return names.get(builtin, str(builtin))

    lines = [
        f"model   {path}",
        "inputs  "
        + ", ".join(f"{graph.Tensors(i).Name().decode()}{shape(i)}" for i in graph.InputsAsNumpy()),
        "outputs "
        + ", ".join(
            f"{graph.Tensors(i).Name().decode()}{shape(i)}" for i in graph.OutputsAsNumpy()
        ),
        "",
    ]
    counts: Counter[str] = Counter()
    for index in range(graph.OperatorsLength()):
        operator = graph.Operators(index)
        name = name_of(operator)
        counts[name] += 1
        where = "NPU" if name == "NeutronGraph" else "CPU"
        first = next((shape(i) for i in operator.InputsAsNumpy() if i >= 0), [])
        out = shape(operator.OutputsAsNumpy()[0])
        lines.append(f"{index:3d} {where} {name:16s} {first} -> {out}")
    npu = counts["NeutronGraph"]
    total = sum(counts.values())
    lines += [
        "",
        f"{npu} NeutronGraph (NPU), {total - npu} other operators (CPU)",
        str(dict(counts)),
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    print(report(parser.parse_args().model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
