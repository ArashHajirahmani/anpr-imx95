import json
import logging
from pathlib import Path

from anpr.api import ApiService
from anpr.ipc import unix_request
from anpr.recognize import PlateRead
from anpr.results import ResultStore
from anpr.types import Frame, PlateBox

JPEG = b"\xff\xd8\xff\xd9"


def test_api_serves_metadata_and_jpeg(tmp_path: Path, caplog: object) -> None:
    caplog.set_level(logging.INFO, logger="anpr.api")
    store = ResultStore(4)
    socket_path = tmp_path / "api.sock"
    service = ApiService(store, socket_path)
    service.start()
    try:
        service.wait_until_ready(2)
        assert "curl --unix-socket" in caplog.text
        assert "/v1/latest" in caplog.text
        status, body = unix_request(str(socket_path), "/health", 1)
        assert status == 200
        assert json.loads(body)["ok"] is True
        status, body = unix_request(str(socket_path), "/v1/latest", 1)
        assert status == 503
        assert json.loads(body)["error"] == "no frame yet"
        status, _body = unix_request(str(socket_path), "/v1/latest.jpg", 1)
        assert status == 404

        store.record(
            Frame("videoCapture", b"raw", 5, 100, 50),
            [PlateRead("ABC123", 0.91, PlateBox(4, 6, 30, 12, 0.8))],
            min_confidence=0.07,
            jpeg=JPEG,
            last_texts={},
        )
        status, body = unix_request(str(socket_path), "/v1/latest", 1)
        assert status == 200
        payload = json.loads(body)
        plate = payload["plates"][0]
        assert plate["text"] == "ABC123"
        assert plate["confidence"] == 0.91
        assert plate["box"] == {
            "x": 4,
            "y": 6,
            "width": 30,
            "height": 12,
            "confidence": 0.8,
        }
        status, image = unix_request(str(socket_path), "/v1/latest.jpg", 1)
        assert status == 200
        assert image == JPEG

        status, body = unix_request(str(socket_path), "/v1/history", 1)
        history = json.loads(body)
        assert history["limit"] == 4
        assert history["items"][0]["text"] == "ABC123"
        event_id = history["items"][0]["id"]
        status, image = unix_request(str(socket_path), f"/v1/history/{event_id}.jpg", 1)
        assert status == 200
        assert image == JPEG
        status, missing = unix_request(str(socket_path), "/v1/history/999.jpg", 1)
        assert status == 404
        assert json.loads(missing)["error"] == "unknown history id"
        status, _body = unix_request(str(socket_path), "/missing", 1)
        assert status == 404
    finally:
        service.close()


def test_reset_clears_history_and_notifies_listeners(tmp_path: Path) -> None:
    store = ResultStore(4)
    notified: list[bool] = []
    store.on_reset(lambda: notified.append(True))
    socket_path = tmp_path / "api.sock"
    service = ApiService(store, socket_path)
    service.start()
    try:
        service.wait_until_ready(2)
        store.record(
            Frame("videoCapture", b"raw", 5, 100, 50),
            [PlateRead("ABC123", 0.91, PlateBox(4, 6, 30, 12, 0.8))],
            min_confidence=0.07,
            jpeg=JPEG,
            last_texts={},
        )
        _status, body = unix_request(str(socket_path), "/v1/history", 1)
        assert json.loads(body)["items"]
        status, _body = unix_request(str(socket_path), "/v1/reset", 1)
        assert status == 404  # GET is not allowed to reset
        status, body = unix_request(str(socket_path), "/v1/reset", 1, method="POST")
        assert status == 200 and json.loads(body)["ok"] is True
        _status, body = unix_request(str(socket_path), "/v1/history", 1)
        assert json.loads(body)["items"] == []
        assert notified == [True]
    finally:
        service.close()
