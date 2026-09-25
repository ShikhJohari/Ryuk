"""HTTP reads for the fetch commands: whole files and byte ranges, over the standard library."""

import io
import urllib.error
import urllib.request
from http.client import HTTPException, HTTPResponse
from typing import override

from ryuk.fetch import FetchError

USER_AGENT = "ryuk-fetch/0.1 (+https://github.com/ShikhJohari/Ryuk)"
TIMEOUT_SECONDS = 60
_PARTIAL_CONTENT = 206


def open_url(url: str, *, byte_range: tuple[int, int] | None = None) -> HTTPResponse:
    """GET `url`, following redirects; `byte_range` is inclusive, as in the Range header."""
    if not url.startswith(("https://", "http://")):
        raise FetchError(f"refusing to fetch {url}: only http(s) sources are pinned")
    headers = {"User-Agent": USER_AGENT}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 (scheme checked above)
    try:
        response: HTTPResponse = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)  # noqa: S310
    except (urllib.error.URLError, TimeoutError) as error:
        raise FetchError(f"could not fetch {url}: {error}") from error
    if byte_range is not None and response.status != _PARTIAL_CONTENT:
        response.close()
        raise FetchError(f"{url} ignored the byte range request (HTTP {response.status})")
    return response


class RangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file, read with one range request per read.

    Reads served from `prefetched` spans cost nothing; everything else goes to the network.
    The redirect is resolved once up front, so each read goes straight to the final host.
    """

    def __init__(self, url: str, size: int) -> None:
        super().__init__()
        with open_url(url, byte_range=(0, 0)) as response:
            self._url = response.url
            total = response.headers.get("Content-Range", "").rpartition("/")[2]
        if total != str(size):
            raise FetchError(f"{url} is {total or 'an unknown number of'} bytes, pinned at {size}")
        self._size = size
        self._position = 0
        self._spans: list[tuple[int, bytes]] = []

    def fetch(self, start: int, length: int) -> bytes:
        """Read `length` bytes at `start` from the network, bypassing the prefetched spans."""
        try:
            with open_url(self._url, byte_range=(start, start + length - 1)) as response:
                data: bytes = response.read()
        except (OSError, HTTPException) as error:
            raise FetchError(f"reading {self._url} was cut short: {error!r}") from error
        if len(data) != length:
            raise FetchError(f"{self._url} returned {len(data)} bytes for a {length}-byte range")
        return data

    def prefetched(self, start: int, data: bytes) -> None:
        """Keep a span read ahead of time, so later reads inside it skip the network."""
        self._spans.append((start, data))

    @override
    def readable(self) -> bool:
        return True

    @override
    def seekable(self) -> bool:
        return True

    @override
    def tell(self) -> int:
        return self._position

    @override
    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self._position, io.SEEK_END: self._size}[whence]
        self._position = max(0, origin + offset)
        return self._position

    @override
    def readinto(self, buffer: memoryview | bytearray) -> int:  # type: ignore[override]
        length = min(len(buffer), self._size - self._position)
        if length <= 0:
            return 0
        data = self._cached(self._position, length) or self.fetch(self._position, length)
        buffer[:length] = data
        self._position += length
        return length

    def _cached(self, start: int, length: int) -> bytes | None:
        for span_start, span in self._spans:
            offset = start - span_start
            if offset >= 0 and offset + length <= len(span):
                return span[offset : offset + length]
        return None
