from pathlib import Path

import pytest

from anpr.infer import (
    LITERT_DELEGATE_NAME,
    DelegateNotFoundError,
    discover_delegates,
    select_tflite_delegate,
)


def test_prefers_tflite_delegate_over_litert(tmp_path: Path) -> None:
    neutron = _touch(tmp_path, "libneutron_delegate.so")
    litert = _touch(tmp_path, LITERT_DELEGATE_NAME)
    found = discover_delegates([tmp_path])
    assert set(found) == {neutron, litert}
    assert select_tflite_delegate(found) == neutron


def test_ignores_litert_only_directory(tmp_path: Path) -> None:
    _touch(tmp_path, LITERT_DELEGATE_NAME)
    found = discover_delegates([tmp_path])
    with pytest.raises(DelegateNotFoundError):
        select_tflite_delegate(found)


def test_accepts_tensorflow_lite_named_delegate(tmp_path: Path) -> None:
    named = _touch(tmp_path, "libtensorflow-lite-neutron-delegate.so")
    found = discover_delegates([tmp_path])
    assert select_tflite_delegate(found) == named


def test_prefers_libneutron_when_both_tflite_names_exist(tmp_path: Path) -> None:
    preferred = _touch(tmp_path, "libneutron_delegate.so")
    _touch(tmp_path, "libtensorflow-lite-neutron-delegate.so")
    assert select_tflite_delegate(discover_delegates([tmp_path])) == preferred


def test_searches_second_root_and_skips_missing(tmp_path: Path) -> None:
    multiarch = tmp_path / "aarch64-linux-gnu"
    multiarch.mkdir()
    library = _touch(multiarch, "libneutron_delegate.so")
    found = discover_delegates([tmp_path / "missing", multiarch])
    assert found == [library]


def _touch(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"")
    return path.resolve()
