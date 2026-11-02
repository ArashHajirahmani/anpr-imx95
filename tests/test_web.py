import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path

from anpr.api import ApiService
from anpr.recognize import PlateRead
from anpr.results import ResultStore
from anpr.types import Frame, PlateBox
from anpr.web import WebService

JPEG = b"\xff\xd8\xff\xd9plate"


def test_page_survives_a_missing_api_then_shows_the_plate(tmp_path: Path, caplog: object) -> None:
    caplog.set_level(logging.INFO, logger="anpr.web")
    socket_path = tmp_path / "api.sock"
    web = WebService(socket_path, "127.0.0.1", 0, 0.05)
    _host, port = web.start()
    api: ApiService | None = None
    try:
        page = _get(port, "/")
        assert b'id="plate"' in page
        assert b'id="history"' in page
        assert b'id="error"' in page
        assert b"#39ff14" in page
        assert b"box.width / width" in page
        assert b"Character confidence" in page
        assert b"Detector time" in page
        assert b"Plate read time" in page
        _wait(lambda: "anpr api unreachable" in caplog.text)
        state = json.loads(_get(port, "/api/state"))
        assert state["ok"] is False
        assert state["latest"] is None

        store = ResultStore(4)
        api = ApiService(store, socket_path)
        api.start()
        api.wait_until_ready(2)
        _wait(lambda: json.loads(_get(port, "/api/state"))["ok"] is True)
        assert json.loads(_get(port, "/api/state"))["latest"] is None

        store.record(
            Frame("videoCapture", b"raw", 9, 200, 100),
            [PlateRead("ABC123", 0.82, PlateBox(20, 30, 80, 24, 0.77))],
            min_confidence=0.07,
            jpeg=JPEG,
            last_texts={},
        )
        _wait(lambda: _plate_text(port) == "ABC123")
        state = json.loads(_get(port, "/api/state"))
        plate = state["latest"]["plates"][0]
        assert plate["box"] == {
            "x": 20,
            "y": 30,
            "width": 80,
            "height": 24,
            "confidence": 0.77,
        }
        assert state["latest"]["width"] == 200
        assert state["latest"]["height"] == 100
        assert state["history_limit"] == 4
        assert state["history"][0]["text"] == "ABC123"
        assert _get(port, "/media/latest.jpg") == JPEG
        image = state["history"][0]["image"]
        assert _get(port, image) == JPEG
        assert "anpr api reachable" in caplog.text
        assert "plate ABC123" in caplog.text
        assert "ANPR_LOG_LEVEL=DEBUG" in caplog.text
    finally:
        if api is not None:
            api.close()
        web.close()


def _plate_text(port: int) -> str | None:
    latest = json.loads(_get(port, "/api/state")).get("latest")
    if not latest or not latest.get("plates"):
        return None
    return latest["plates"][0]["text"]


def _get(port: int, path: str) -> bytes:
    deadline = time.monotonic() + 2
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=1) as response:
                return response.read()
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last = exc
            time.sleep(0.02)
    raise AssertionError(last)


def _wait(predicate: object) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("timed out")
