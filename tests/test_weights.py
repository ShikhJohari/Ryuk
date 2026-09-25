"""The weights are fetched to pinned paths and checked against pinned sha256s."""

import hashlib
import io
import zipfile
from pathlib import Path, PurePosixPath

import pytest

from file_server import FileServer
from ryuk.fetch.pinned import Checksum, ChecksumMismatchError, FetchError, PinnedFile
from ryuk.weights import WEIGHTS, YUNET, ExtractedFile, Weights, fetch_weights

YUNET_BYTES = b"yunet onnx"
ARCFACE_BYTES = b"arcface onnx" * 100


def sha256(content: bytes) -> Checksum:
    return Checksum("sha256", hashlib.sha256(content).hexdigest())


def buffalo(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def weights(server: FileServer, *, arcface: bytes = ARCFACE_BYTES) -> tuple[Weights, ...]:
    archive = buffalo({"det_10g.onnx": b"detector", "w600k_r50.onnx": arcface})
    return (
        Weights(
            name="yunet",
            file=PinnedFile(
                url=server.serve("yunet.onnx", YUNET_BYTES),
                path=PurePosixPath("face_detection_yunet_2026may.onnx"),
                size=len(YUNET_BYTES),
                checksum=sha256(YUNET_BYTES),
            ),
        ),
        Weights(
            name="arcface",
            file=ExtractedFile(
                archive=PinnedFile(
                    url=server.serve("buffalo_l.zip", archive),
                    path=PurePosixPath("buffalo_l.zip"),
                    size=len(archive),
                    checksum=sha256(archive),
                ),
                member="w600k_r50.onnx",
                path=PurePosixPath("w600k_r50.onnx"),
                size=len(ARCFACE_BYTES),
                checksum=sha256(ARCFACE_BYTES),
            ),
        ),
    )


def files(folder: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(folder.iterdir())}


def test_every_weight_file_lands_at_its_pinned_path(
    file_server: FileServer, tmp_path: Path
) -> None:
    fetched = fetch_weights(tmp_path, weights(file_server))

    assert [item.path for item in fetched] == [
        tmp_path / "face_detection_yunet_2026may.onnx",
        tmp_path / "w600k_r50.onnx",
    ]
    # Only the recognition model is kept from buffalo_l; the archive itself is not.
    assert files(tmp_path) == {
        "face_detection_yunet_2026may.onnx": YUNET_BYTES,
        "w600k_r50.onnx": ARCFACE_BYTES,
    }


def test_a_second_run_downloads_nothing(file_server: FileServer, tmp_path: Path) -> None:
    pinned = weights(file_server)
    fetch_weights(tmp_path, pinned)
    file_server.requests.clear()

    fetched = fetch_weights(tmp_path, pinned)

    assert not any(item.updated for item in fetched)
    assert file_server.requests == []


def test_an_extracted_file_that_fails_its_hash_fails_loudly(
    file_server: FileServer, tmp_path: Path
) -> None:
    # A verified buffalo_l.zip whose ArcFace member is not the pinned ArcFace.
    pinned = weights(file_server, arcface=ARCFACE_BYTES[:-1] + b"!")

    with pytest.raises(ChecksumMismatchError, match=r"w600k_r50\.onnx"):
        fetch_weights(tmp_path, pinned)

    assert files(tmp_path) == {"face_detection_yunet_2026may.onnx": YUNET_BYTES}


def test_an_archive_without_the_member_fails_loudly(
    file_server: FileServer, tmp_path: Path
) -> None:
    pinned = weights(file_server)
    arcface = pinned[1].file
    assert isinstance(arcface, ExtractedFile)
    missing = ExtractedFile(
        archive=arcface.archive,
        member="w600k_r100.onnx",
        path=arcface.path,
        size=arcface.size,
        checksum=arcface.checksum,
    )

    with pytest.raises(FetchError, match=r"w600k_r100\.onnx"):
        fetch_weights(tmp_path, (Weights(name="arcface", file=missing),))

    assert files(tmp_path) == {}


def test_a_corrupted_download_fails_loudly(file_server: FileServer, tmp_path: Path) -> None:
    pinned = weights(file_server)
    file_server.serve("yunet.onnx", b"yunet onnY")

    with pytest.raises(ChecksumMismatchError, match=r"face_detection_yunet_2026may\.onnx"):
        fetch_weights(tmp_path, pinned)

    assert files(tmp_path) == {}


def test_the_pinned_weights_are_the_four_the_toolchain_spike_verified() -> None:
    assert {entry.name: entry.file.checksum.hexdigest for entry in WEIGHTS} == {
        "yunet": "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0",
        "sface": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "arcface": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        "facenet": "281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b",
    }


def test_the_committed_yunet_fixture_is_the_pinned_detector() -> None:
    fixture = YUNET.path(Path(__file__).parent / "fixtures")

    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == YUNET.file.checksum.hexdigest
