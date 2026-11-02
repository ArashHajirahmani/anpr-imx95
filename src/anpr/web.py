"""Browser client for the ANPR Unix-socket API.

This process is independent of recognition. It keeps serving the page when
the socket is missing, and it logs reachability to the journal.
"""

import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from anpr.ipc import unix_request

logger = logging.getLogger("anpr.web")
_PAGE_PATH = Path(__file__).resolve().parent / "static" / "index.html"
_PAGE = _PAGE_PATH.read_bytes()
_HEARTBEAT_SECONDS = 30.0
_JSON = "application/json; charset=utf-8"
_HISTORY_MEDIA = re.compile(r"^/media/history/(\d+)\.jpg$")


class WebService:
    """Serve the single page and poll the recognition socket."""

    def __init__(self, api_socket: Path, host: str, port: int, poll_seconds: float) -> None:
        self.api_socket = api_socket
        self.host = host
        self.port = port
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._cache = ClientCache(str(api_socket))
        self._server: WebServer | None = None
        self._http = threading.Thread(target=self._serve, name="anpr-web", daemon=True)
        self._poller = threading.Thread(target=self._poll_loop, name="anpr-web-poll", daemon=True)

    def start(self) -> tuple[str, int]:
        server = WebServer((self.host, self.port), self._cache)
        self._server = server
        self.host = server.server_address[0]
        self.port = server.server_address[1]
        self._http.start()
        self._poller.start()
        logger.info("web client listening on http://%s:%s", self.host, self.port)
        logger.info("reading plates from unix socket %s", self.api_socket)
        logger.info("set ANPR_LOG_LEVEL=DEBUG for per-poll api traces")
        return self.host, self.port

    def close(self) -> None:
        self._stop.set()
        if self._poller.is_alive():
            self._poller.join(timeout=5)
        if self._http.is_alive():
            self._http.join(timeout=5)
        if self._http.is_alive() or self._poller.is_alive():
            logger.warning("web client threads did not stop")

    def _serve(self) -> None:
        server = self._server
        if server is None:
            return
        try:
            server.timeout = 0.2
            while not self._stop.is_set():
                try:
                    server.handle_request()
                except OSError as exc:
                    if self._stop.is_set():
                        break
                    logger.warning("web accept failed: %s", exc)
        except Exception:
            logger.exception("web server failed")
        finally:
            server.server_close()

    def _poll_loop(self) -> None:
        timeout = max(2.0, self.poll_seconds)
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._cache.poll(timeout)
            except OSError as exc:
                self._cache.fail(f"{exc.__class__.__name__}: {exc}")
            except Exception:
                logger.exception("api poll failed")
                self._cache.fail("api poll failed")
            self._cache.maybe_heartbeat()
            remaining = self.poll_seconds - (time.monotonic() - started)
            if remaining > 0 and self._stop.wait(remaining):
                return


class ClientCache:
    """Last good snapshot. A failed poll keeps it and records the error."""

    def __init__(self, api_socket: str) -> None:
        self.api_socket = api_socket
        self._lock = threading.Lock()
        self.ok = False
        self.error: str | None = "waiting for anpr api"
        self.failures = 0
        self._reachable: bool | None = None
        self._latest: dict[str, object] | None = None
        self._latest_jpeg = b""
        self._history: list[dict[str, object]] = []
        self._history_jpeg: dict[int, bytes] = {}
        self._history_limit = 4
        self._last_heartbeat = time.monotonic()

    def fail(self, message: str) -> None:
        with self._lock:
            self.failures += 1
            self.ok = False
            self.error = message
            changed = self._reachable is not False
            self._reachable = False
        if changed:
            logger.warning("anpr api unreachable at %s: %s", self.api_socket, message)

    def poll(self, timeout: float) -> None:
        status, _body = unix_request(self.api_socket, "/health", timeout)
        if status != 200:
            raise OSError(f"health status {status}")
        status, body = unix_request(self.api_socket, "/v1/latest", timeout)
        latest: dict[str, object] | None
        jpeg = b""
        if status == 503:
            latest = None
        elif status != 200:
            raise OSError(f"latest status {status}")
        else:
            latest = _object(body)
            status, jpeg = unix_request(self.api_socket, "/v1/latest.jpg", timeout)
            if status != 200 or not jpeg:
                raise OSError(f"latest frame status {status}")
        status, body = unix_request(self.api_socket, "/v1/history", timeout)
        if status != 200:
            raise OSError(f"history status {status}")
        history = _object(body)
        items = history.get("items")
        if not isinstance(items, list):
            raise OSError("history items missing")
        limit = history.get("limit")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            limit = len(items) or 4
        parsed: list[dict[str, object]] = []
        images: dict[int, bytes] = {}
        for item in items:
            if not isinstance(item, dict):
                raise OSError("history item is not an object")
            event_id = item.get("id")
            if isinstance(event_id, bool) or not isinstance(event_id, int):
                raise OSError("history item is missing an id")
            status, image = unix_request(
                self.api_socket,
                f"/v1/history/{event_id}.jpg",
                timeout,
            )
            if status != 200 or not image:
                raise OSError(f"history frame {event_id} status {status}")
            images[event_id] = image
            parsed.append(item)
        self._commit(latest, jpeg, parsed, images, limit)
        frame_id = latest.get("frame_id") if latest is not None else "none"
        logger.debug("poll frame=%s history=%d", frame_id, len(parsed))

    def maybe_heartbeat(self) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._last_heartbeat < _HEARTBEAT_SECONDS:
                return
            self._last_heartbeat = now
            ok = self.ok
            error = self.error
            failures = self.failures
            frame_id = self._latest.get("frame_id") if self._latest is not None else "none"
            count = len(self._history)
        logger.info(
            "web poll ok=%s frame=%s history=%d failures=%d error=%s",
            ok,
            frame_id,
            count,
            failures,
            error,
        )

    def state(self) -> dict[str, object]:
        with self._lock:
            latest = None
            if self._latest is not None and self._latest_jpeg:
                latest = dict(self._latest)
                latest["image"] = f"/media/latest.jpg?v={self._latest.get('frame_id')}"
            history = []
            for item in self._history:
                shown = dict(item)
                shown["image"] = f"/media/history/{item.get('id')}.jpg"
                history.append(shown)
            return {
                "ok": self.ok,
                "error": self.error,
                "failures": self.failures,
                "api_socket": self.api_socket,
                "history_limit": self._history_limit,
                "latest": latest,
                "history": history,
            }

    def latest_jpeg(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    def history_jpeg(self, event_id: int) -> bytes:
        with self._lock:
            return self._history_jpeg.get(event_id, b"")

    def _commit(
        self,
        latest: dict[str, object] | None,
        jpeg: bytes,
        items: list[dict[str, object]],
        images: dict[int, bytes],
        limit: int,
    ) -> None:
        with self._lock:
            previous = {item.get("id") for item in self._history}
            changed = self._reachable is not True
            self._reachable = True
            self.ok = True
            self.error = None
            self._latest = latest
            self._latest_jpeg = jpeg
            self._history = items
            self._history_jpeg = images
            self._history_limit = limit
        if changed:
            logger.info("anpr api reachable at %s", self.api_socket)
        for item in items:
            if item.get("id") in previous:
                continue
            _log_plate(item)


class WebServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], cache: ClientCache) -> None:
        self.cache = cache
        super().__init__(address, WebHandler)


class WebHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server: WebServer

    def setup(self) -> None:
        super().setup()
        self.request.settimeout(5)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            self._route(path)
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("browser closed the connection")
        except Exception:
            logger.exception("web request failed for %s", path)
            try:
                self._send(500, _JSON, _dump({"error": "internal error"}))
            except OSError:
                logger.debug("browser closed the connection during the error response")

    def _route(self, path: str) -> None:
        if path in {"/", "/index.html"}:
            self._send(200, "text/html; charset=utf-8", _PAGE)
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Connection", "close")
            self.end_headers()
            return
        if path == "/api/state":
            self._send(200, _JSON, _dump(self.server.cache.state()))
            return
        if path == "/media/latest.jpg":
            payload = self.server.cache.latest_jpeg()
            if not payload:
                self._send(404, _JSON, _dump({"error": "no frame yet"}))
                return
            self._send(200, "image/jpeg", payload)
            return
        match = _HISTORY_MEDIA.match(path)
        if match is not None:
            payload = self.server.cache.history_jpeg(int(match.group(1)))
            if not payload:
                self._send(404, _JSON, _dump({"error": "unknown history id"}))
                return
            self._send(200, "image/jpeg", payload)
            return
        self._send(404, _JSON, _dump({"error": "not found"}))

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


def _object(body: bytes) -> dict[str, object]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OSError(f"invalid json: {exc}") from exc
    if not isinstance(payload, dict):
        raise OSError("api response was not a JSON object")
    return payload


def _log_plate(item: dict[str, object]) -> None:
    box = item.get("box")
    if not isinstance(box, dict):
        box = {}
    try:
        confidence = float(item.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    logger.info(
        "plate %s confidence=%.2f box=%s,%s %sx%s id=%s",
        item.get("text"),
        confidence,
        box.get("x"),
        box.get("y"),
        box.get("width"),
        box.get("height"),
        item.get("id"),
    )


def _dump(payload: object) -> bytes:
    return json.dumps(payload, indent=2).encode("utf-8")
