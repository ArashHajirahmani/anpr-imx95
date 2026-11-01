"""Neutron delegate discovery for the TFLite runtime.

`tflite_runtime` is imported only when an interpreter is created.
"""

from dataclasses import dataclass
from pathlib import Path

SEARCH_DIRECTORIES = (Path("/usr/lib"), Path("/usr/lib/aarch64-linux-gnu"))
DELEGATE_GLOBS = ("*neutron*delegate*.so", "libtensorflow-lite-neutron-delegate.so")
TFLITE_DELEGATE_PREFERENCE = (
    "libneutron_delegate.so",
    "libtensorflow-lite-neutron-delegate.so",
)
LITERT_DELEGATE_NAME = "liblitert_neutron_delegate.so"
NEUTRON_DEVICE = Path("/dev/neutron0")


class DelegateNotFoundError(FileNotFoundError):
    """No TFLite Neutron delegate was found in the search directories."""


@dataclass(frozen=True)
class DelegateProbe:
    candidates: tuple[Path, ...]
    selected: Path | None
    device_node: Path
    device_present: bool


def discover_delegates(roots: tuple[Path, ...] | list[Path] = SEARCH_DIRECTORIES) -> list[Path]:
    found: dict[Path, None] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in DELEGATE_GLOBS:
            for path in root.glob(pattern):
                if path.is_file():
                    found[path.resolve()] = None
    return sorted(found, key=lambda path: path.name)


def select_tflite_delegate(candidates: list[Path] | tuple[Path, ...]) -> Path:
    by_name = {path.name: path for path in candidates}
    for name in TFLITE_DELEGATE_PREFERENCE:
        if name in by_name:
            return by_name[name]
    raise DelegateNotFoundError(
        "no TFLite Neutron delegate found "
        f"({LITERT_DELEGATE_NAME} is not loadable by tflite_runtime)"
    )


def probe_delegates(
    roots: tuple[Path, ...] | list[Path] = SEARCH_DIRECTORIES,
    device_node: Path = NEUTRON_DEVICE,
) -> DelegateProbe:
    candidates = tuple(discover_delegates(roots))
    try:
        selected: Path | None = select_tflite_delegate(candidates)
    except DelegateNotFoundError:
        selected = None
    return DelegateProbe(
        candidates=candidates,
        selected=selected,
        device_node=device_node,
        device_present=device_node.exists(),
    )


def load_interpreter(model_path: Path, delegate_path: Path) -> object:
    """Create a TFLite interpreter bound to the Neutron delegate."""
    from tflite_runtime.interpreter import Interpreter, load_delegate

    delegate = load_delegate(str(delegate_path))
    interpreter = Interpreter(model_path=str(model_path), experimental_delegates=[delegate])
    interpreter.allocate_tensors()
    return interpreter
