"""A loopback HTTP server that serves in-memory files, with byte ranges, for the fetch tests."""

import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_RANGE = re.compile(r"bytes=(\d+)-(\d+)")


@dataclass
class FileServer:
    base_url: str
    files: dict[str, bytes] = field(default_factory=dict)
    redirects: dict[str, str] = field(default_factory=dict)
    requests: list[str] = field(default_factory=list)
    served_bytes: dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    honour_ranges: bool = True

    def url(self, path: str) -> str:
        return f"{self.base_url}/{path}"

    def serve(self, path: str, content: bytes) -> str:
        self.files[f"/{path}"] = content
        return self.url(path)

    def redirect(self, path: str, to: str) -> str:
        self.redirects[f"/{path}"] = f"/{to}"
        return self.url(path)


@contextmanager
def running_file_server() -> Iterator[FileServer]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    state = FileServer(base_url=f"http://127.0.0.1:{server.server_address[1]}")
    server.state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        state: FileServer = self.server.state  # type: ignore[attr-defined]
        with state.lock:
            state.requests.append(self.path)
        if self.path in state.redirects:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", state.redirects[self.path])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        content = state.files.get(self.path)
        if content is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        requested = state.honour_ranges and _RANGE.fullmatch(self.headers.get("Range", ""))
        if requested:
            start, end = int(requested[1]), min(int(requested[2]), len(content) - 1)
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
            content = content[start : end + 1]
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)
        with state.lock:
            state.served_bytes[self.path] = state.served_bytes.get(self.path, 0) + len(content)

    def log_message(self, format: str, *args: object) -> None:
        pass
