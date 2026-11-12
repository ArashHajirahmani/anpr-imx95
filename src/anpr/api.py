"""HTTP API on a Unix socket.

JSON metadata and the JPEG are separate GET resources, so the image stays
binary. The recognition process does not open a TCP port. Inspect it with
`curl --unix-socket /run/anpr/api.sock http://localhost/v1/latest`.
"""

import json
import logging
import os
import re
import socket
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

from anpr.results import ResultStore

logger = logging.getLogger("anpr.api")
_HEARTBEAT_SECONDS = 30.0
_RETRY_SECONDS = 2.0
_JSON = "application/json; charset=utf-8"
_HISTORY_JPEG = re.compile(r"^/v1/history/(\d+)\.jpg$")


class ApiService:
    """Serve the result store until close. Bind failures are retried."""

    def __init__(self, store: ResultStore, socket_path: Path) -> None:
        self.store = store
        self.socket_path = socket_path
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="anpr-api", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wait_until_ready(self, timeout: float) -> None:
        if not self._ready.wait(timeout):
            raise TimeoutError(f"api did not listen on {self.socket_path}")

    def close(self) -> None:
        self._stop.set()
        if not self._thread.is_alive():
            return
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            logger.warning("api thread did not stop")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                server = _bind(self.socket_path, self.store)
            except OSError as exc:
                logger.error("api bind failed for %s: %s", self.socket_path, exc)
                self._stop.wait(_RETRY_SECONDS)
                continue
            logger.info("api listening on unix socket %s", self.socket_path)
            logger.info("curl --unix-socket %s http://localhost/v1/latest", self.socket_path)
            self._ready.set()
            heartbeat = time.monotonic()
            try:
                server.timeout = 0.2
                while not self._stop.is_set():
                    try:
                        server.handle_request()
                    except OSError as exc:
                        if self._stop.is_set():
                            break
                        logger.warning("api accept failed: %s", exc)
                    now = time.monotonic()
                    if now - heartbeat >= _HEARTBEAT_SECONDS:
                        _heartbeat(self.store)
                        heartbeat = now
            except Exception:
                logger.exception("api server failed")
            finally:
                server.server_close()
                self.socket_path.unlink(missing_ok=True)
            if not self._stop.is_set():
                logger.warning("api server stopped; binding again")
                self._ready.clear()


class ApiServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 16

    def __init__(self, path: str, store: ResultStore) -> None:
        self.store = store
        super().__init__(path, ApiHandler)


class ApiHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server: ApiServer

    def setup(self) -> None:
        super().setup()
        self.request.settimeout(5)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            self._route(path)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("client closed the connection")
        except Exception:
            logger.exception("api request failed for %s", path)
            try:
                self._send(500, _JSON, _dump({"error": "internal error"}))
            except OSError:
                logger.debug("client closed the connection during the error response")

    def _route(self, path: str) -> None:
        if path == "/health":
            self._send(200, _JSON, _dump({"ok": True, "service": "anpr"}))
            return
        if path == "/v1/latest":
            self._latest()
            return
        if path == "/v1/latest.jpg":
            self._latest_jpeg()
            return
        if path == "/v1/history":
            self._history()
            return
        match = _HISTORY_JPEG.match(path)
        if match is None:
            self._send(404, _JSON, _dump({"error": "not found"}))
            return
        self._history_jpeg(int(match.group(1)))

    def _latest(self) -> None:
        latest = self.server.store.latest()
        if latest is None:
            self._send(503, _JSON, _dump({"error": "no frame yet"}))
            return
        self._send(200, _JSON, _dump(latest.json()))

    def _latest_jpeg(self) -> None:
        latest = self.server.store.latest()
        if latest is None or not latest.jpeg:
            self._send(404, _JSON, _dump({"error": "no frame yet"}))
            return
        self._send(200, "image/jpeg", latest.jpeg)

    def _history(self) -> None:
        store = self.server.store
        body = {
            "limit": store.history_limit,
            "items": [item.json() for item in store.history()],
        }
        self._send(200, _JSON, _dump(body))

    def _history_jpeg(self, event_id: int) -> None:
        for item in self.server.store.history():
            if item.id == event_id:
                self._send(200, "image/jpeg", item.jpeg)
                return
        self._send(404, _JSON, _dump({"error": "unknown history id"}))

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        logger.debug("GET %s -> %s (%d bytes)", self.path, status, len(body))

    def log_message(self, fmt: str, *args: object) -> None:
        logger.debug("%s", fmt % args)


def _bind(path: Path, store: ResultStore) -> ApiServer:
    path.parent.mkdir(parents=True, exist_ok=True)
    _clear_stale_socket(path)
    server = ApiServer(str(path), store)
    try:
        os.chmod(path, 0o660)
    except OSError:
        server.server_close()
        path.unlink(missing_ok=True)
        raise
    return server


def _clear_stale_socket(path: Path) -> None:
    if not path.exists():
        return
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.2)
        probe.connect(str(path))
    except OSError:
        path.unlink(missing_ok=True)
    else:
        raise OSError(f"api socket is already being served: {path}")
    finally:
        probe.close()


def _heartbeat(store: ResultStore) -> None:
    latest = store.latest()
    count = len(store.history())
    if latest is None:
        logger.info("api heartbeat frame=none history=%d", count)
        return
    plate = latest.plates[0].text if latest.plates else "-"
    logger.info(
        "api heartbeat frame=%s plate=%s history=%d stream=%s",
        latest.frame_id,
        plate,
        count,
        latest.stream,
    )


def _dump(payload: object) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")
