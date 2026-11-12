"""HTTP over a Unix domain socket.

The recognition process is the server. The web client is the only caller.
"""

import http.client
import socket


class UnixHTTPConnection(http.client.HTTPConnection):
    """Speak HTTP to a Unix socket. The host header is unused."""

    def __init__(self, socket_path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


def unix_request(
    socket_path: str,
    path: str,
    timeout: float,
    method: str = "GET",
) -> tuple[int, bytes]:
    """Request one path (GET by default). Raises OSError when the socket cannot be used."""
    connection = UnixHTTPConnection(socket_path, timeout)
    try:
        connection.request(method, path)
        response = connection.getresponse()
        body = response.read()
        return response.status, body
    finally:
        connection.close()
