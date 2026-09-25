"""A remote file read by byte ranges, as the CelebA label fetch reads the train shards."""

import io

import pytest

from file_server import FileServer
from ryuk.fetch import FetchError
from ryuk.fetch.http import RangeFile

CONTENT = bytes(range(256)) * 40


def test_reads_are_byte_ranges_of_the_remote_file(file_server: FileServer) -> None:
    remote = RangeFile(file_server.serve("shard.parquet", CONTENT), len(CONTENT))

    remote.seek(100)
    assert remote.read(50) == CONTENT[100:150]
    assert remote.tell() == 150
    remote.seek(-10, io.SEEK_END)
    assert remote.read(50) == CONTENT[-10:]
    assert remote.read(50) == b""
    remote.seek(-5, io.SEEK_CUR)
    assert remote.read() == CONTENT[-5:]


def test_the_redirect_is_resolved_once(file_server: FileServer) -> None:
    file_server.serve("cdn/shard.parquet", CONTENT)
    url = file_server.redirect("resolve/shard.parquet", "cdn/shard.parquet")
    remote = RangeFile(url, len(CONTENT))
    file_server.requests.clear()

    remote.read(10)
    remote.read(10)

    assert file_server.requests == ["/cdn/shard.parquet", "/cdn/shard.parquet"]


def test_reads_inside_a_prefetched_span_skip_the_network(file_server: FileServer) -> None:
    remote = RangeFile(file_server.serve("shard.parquet", CONTENT), len(CONTENT))
    remote.prefetched(1000, remote.fetch(1000, 500))
    file_server.requests.clear()

    remote.seek(1100)
    assert remote.read(400) == CONTENT[1100:1500]
    assert file_server.requests == []

    remote.seek(1400)
    assert remote.read(200) == CONTENT[1400:1600]
    assert file_server.requests == ["/shard.parquet"]


def test_a_remote_file_of_another_size_is_refused(file_server: FileServer) -> None:
    url = file_server.serve("shard.parquet", CONTENT)

    with pytest.raises(FetchError, match=f"{len(CONTENT)} bytes, pinned at 99"):
        RangeFile(url, 99)


def test_a_server_that_ignores_ranges_is_refused(file_server: FileServer) -> None:
    file_server.honour_ranges = False

    with pytest.raises(FetchError, match="ignored the byte range"):
        RangeFile(file_server.serve("shard.parquet", CONTENT), len(CONTENT))


def test_a_short_range_is_refused(file_server: FileServer) -> None:
    remote = RangeFile(file_server.serve("shard.parquet", CONTENT), len(CONTENT))

    with pytest.raises(FetchError, match="returned 10 bytes for a 20-byte range"):
        remote.fetch(len(CONTENT) - 10, 20)
