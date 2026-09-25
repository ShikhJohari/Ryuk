"""A pinned file lands in place only once its size and checksum match; a verified one is kept."""

import hashlib
from pathlib import Path, PurePosixPath

import pytest

from file_server import FileServer
from ryuk.fetch.pinned import (
    Algorithm,
    Checksum,
    ChecksumMismatchError,
    FetchError,
    PinnedFile,
    fetch_pinned,
)

CONTENT = b"thirteen thousand faces" * 1000


def pinned(url: str, content: bytes = CONTENT, *, algorithm: Algorithm = "sha256") -> PinnedFile:
    digest = hashlib.new(algorithm, content).hexdigest()
    return PinnedFile(
        url=url,
        path=PurePosixPath("lfw/pairs.txt"),
        size=len(content),
        checksum=Checksum(algorithm, digest),
    )


def leftovers(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


@pytest.mark.parametrize("algorithm", ["md5", "sha256"])
def test_a_download_whose_checksum_matches_is_renamed_into_place(
    file_server: FileServer, tmp_path: Path, algorithm: Algorithm
) -> None:
    url = file_server.serve("pairs.txt", CONTENT)

    outcome = fetch_pinned(pinned(url, algorithm=algorithm), tmp_path)

    assert outcome.updated
    assert outcome.path == tmp_path / "lfw" / "pairs.txt"
    assert outcome.path.read_bytes() == CONTENT
    assert leftovers(tmp_path) == ["lfw/pairs.txt"]


def test_a_redirected_download_is_followed(file_server: FileServer, tmp_path: Path) -> None:
    file_server.serve("s3/pairs.txt", CONTENT)
    url = file_server.redirect("files/5976006", "s3/pairs.txt")

    assert fetch_pinned(pinned(url), tmp_path).path.read_bytes() == CONTENT


def test_an_existing_verified_file_is_skipped(file_server: FileServer, tmp_path: Path) -> None:
    url = file_server.serve("pairs.txt", CONTENT)
    fetch_pinned(pinned(url), tmp_path)
    file_server.requests.clear()

    outcome = fetch_pinned(pinned(url), tmp_path)

    assert not outcome.updated
    assert file_server.requests == []


def test_an_existing_file_that_fails_verification_is_replaced(
    file_server: FileServer, tmp_path: Path
) -> None:
    url = file_server.serve("pairs.txt", CONTENT)
    target = tmp_path / "lfw" / "pairs.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(CONTENT[:-1] + b"!")

    outcome = fetch_pinned(pinned(url), tmp_path)

    assert outcome.updated
    assert target.read_bytes() == CONTENT


def test_a_corrupted_download_is_never_renamed_into_place(
    file_server: FileServer, tmp_path: Path
) -> None:
    url = file_server.serve("pairs.txt", CONTENT[:-1] + b"!")

    with pytest.raises(ChecksumMismatchError) as raised:
        fetch_pinned(pinned(url), tmp_path)

    assert "lfw/pairs.txt" in str(raised.value)
    assert hashlib.sha256(CONTENT).hexdigest() in str(raised.value)
    assert leftovers(tmp_path) == []


def test_a_corrupted_download_leaves_a_previous_file_untouched(
    file_server: FileServer, tmp_path: Path
) -> None:
    url = file_server.serve("pairs.txt", b"tampered")
    target = tmp_path / "lfw" / "pairs.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"stale")

    with pytest.raises(ChecksumMismatchError):
        fetch_pinned(pinned(url), tmp_path)

    assert target.read_bytes() == b"stale"
    assert leftovers(tmp_path) == ["lfw/pairs.txt"]


def test_a_download_of_the_wrong_size_fails_before_hashing(
    file_server: FileServer, tmp_path: Path
) -> None:
    url = file_server.serve("pairs.txt", CONTENT + b"extra")

    with pytest.raises(ChecksumMismatchError, match="bytes"):
        fetch_pinned(pinned(url), tmp_path)

    assert leftovers(tmp_path) == []


def test_a_missing_source_fails_with_its_url(file_server: FileServer, tmp_path: Path) -> None:
    url = file_server.url("gone.txt")

    with pytest.raises(FetchError, match=r"gone\.txt"):
        fetch_pinned(pinned(url), tmp_path)

    assert leftovers(tmp_path) == []


def test_only_web_urls_are_fetched(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match="file:///etc/passwd"):
        fetch_pinned(pinned("file:///etc/passwd"), tmp_path)
