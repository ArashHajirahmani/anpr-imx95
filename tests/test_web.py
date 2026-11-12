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
        assert b'id="live-meta"' in page
        assert b'id="history"' in page
        assert b'id="error"' in page
        assert b"#39ff14" in page
        assert b"STATS_HOLD_MS" in page
        assert b"HISTORY_SLOTS" in page
        assert b"DETECTION" in page
        assert b"PERFORMANCE" in page
        assert b"box-label" in page
        assert b"box-shadow" not in page
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


def test_poll_is_one_request_when_nothing_changed(tmp_path: Path, monkeypatch: object) -> None:
    import anpr.web as web_module
    from anpr.web import ClientCache

    socket_path = tmp_path / "api.sock"
    store = ResultStore(4)
    api = ApiService(store, socket_path)
    api.start()
    api.wait_until_ready(2)
    paths: list[str] = []
    real = web_module.unix_request

    def counting(socket: str, path: str, timeout: float) -> tuple[int, bytes]:
        paths.append(path)
        return real(socket, path, timeout)

    monkeypatch.setattr(web_module, "unix_request", counting)
    try:
        cache = ClientCache(str(socket_path))
        frame = Frame("videoCapture", b"raw", 9, 200, 100)
        plate = PlateRead("ABC123", 0.82, PlateBox(20, 30, 80, 24, 0.77))
        store.record(frame, [plate], min_confidence=0.07, jpeg=JPEG, last_texts={})
        cache.poll(1)
        assert "/v1/latest.jpg" in paths
        assert "/v1/history/1.jpg" in paths

        paths.clear()
        cache.poll(1)
        assert paths == ["/v1/latest"]
        assert cache.state()["ok"] is True

        paths.clear()
        store.record(frame, [], min_confidence=0.07, jpeg=JPEG, last_texts={})
        cache.poll(1)
        assert "/v1/latest.jpg" in paths
        assert "/v1/history" in paths
        assert not any(path.startswith("/v1/history/") for path in paths)
    finally:
        api.close()
