"""LFW is rebuilt from pinned files: pairs lists and the funneled archive, extracted once."""

import hashlib
import io
import tarfile
from pathlib import Path, PurePosixPath

import pytest

from file_server import FileServer
from ryuk.fetch import FetchError
from ryuk.fetch.lfw import LFW, LfwSource, fetch_lfw
from ryuk.fetch.pinned import Checksum, ChecksumMismatchError, PinnedFile

IMAGES = {
    "lfw_funneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg": b"jpeg one",
    "lfw_funneled/Zico/Zico_0001.jpg": b"jpeg two",
}


def tarball(members: dict[str, bytes], *, symlink: tuple[str, str] | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        if symlink is not None:
            info = tarfile.TarInfo(symlink[0])
            info.type = tarfile.SYMTYPE
            info.linkname = symlink[1]
            archive.addfile(info)
    return buffer.getvalue()


def pin(server: FileServer, relative: str, content: bytes) -> PinnedFile:
    return PinnedFile(
        url=server.serve(relative, content),
        path=PurePosixPath("lfw", relative),
        size=len(content),
        checksum=Checksum("md5", hashlib.md5(content, usedforsecurity=False).hexdigest()),
    )


def source(server: FileServer, archive: bytes) -> LfwSource:
    return LfwSource(
        archive=pin(server, "archives/lfw-funneled.tgz", archive),
        pairs=(pin(server, "pairs.txt", b"10\t300\n"), pin(server, "pairsDevTest.txt", b"500\n")),
        extracted=PurePosixPath("lfw/lfw_funneled"),
        images=len(IMAGES),
    )


def images(root: Path) -> dict[str, bytes]:
    folder = root / "lfw" / "lfw_funneled"
    return {
        str(path.relative_to(root / "lfw")): path.read_bytes()
        for path in sorted(folder.rglob("*.jpg"))
    }


def test_the_archive_is_extracted_beside_the_pairs_lists(
    file_server: FileServer, tmp_path: Path
) -> None:
    fetched = fetch_lfw(tmp_path, source(file_server, tarball(IMAGES)))

    assert images(tmp_path) == IMAGES
    assert (tmp_path / "lfw" / "pairs.txt").read_bytes() == b"10\t300\n"
    assert {item.path.name for item in fetched} == {
        "lfw-funneled.tgz",
        "pairs.txt",
        "pairsDevTest.txt",
        "lfw_funneled",
    }
    assert all(item.updated for item in fetched)


def test_a_second_run_downloads_and_extracts_nothing(
    file_server: FileServer, tmp_path: Path
) -> None:
    lfw = source(file_server, tarball(IMAGES))
    fetch_lfw(tmp_path, lfw)
    extracted_at = (tmp_path / "lfw" / "lfw_funneled").stat().st_mtime_ns
    file_server.requests.clear()

    fetched = fetch_lfw(tmp_path, lfw)

    assert not any(item.updated for item in fetched)
    assert file_server.requests == []
    assert (tmp_path / "lfw" / "lfw_funneled").stat().st_mtime_ns == extracted_at


def test_an_extraction_from_another_archive_is_replaced(
    file_server: FileServer, tmp_path: Path
) -> None:
    stale = tmp_path / "lfw" / "lfw_funneled" / "Someone" / "Someone_0001.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"left over")

    fetch_lfw(tmp_path, source(file_server, tarball(IMAGES)))

    assert images(tmp_path) == IMAGES


def test_an_extraction_missing_images_is_extracted_again(
    file_server: FileServer, tmp_path: Path
) -> None:
    lfw = source(file_server, tarball(IMAGES))
    fetch_lfw(tmp_path, lfw)
    (tmp_path / "lfw" / "lfw_funneled" / "Zico" / "Zico_0001.jpg").unlink()

    fetched = fetch_lfw(tmp_path, lfw)

    assert images(tmp_path) == IMAGES
    assert [item.updated for item in fetched] == [False, False, False, True]


def test_an_archive_holding_the_wrong_number_of_images_is_refused(
    file_server: FileServer, tmp_path: Path
) -> None:
    lfw = source(file_server, tarball(IMAGES))
    lfw = LfwSource(lfw.archive, lfw.pairs, lfw.extracted, images=3)

    with pytest.raises(FetchError, match="holds 2 images, not 3"):
        fetch_lfw(tmp_path, lfw)

    assert not (tmp_path / "lfw" / "lfw_funneled").exists()


def test_a_corrupted_archive_is_never_extracted(file_server: FileServer, tmp_path: Path) -> None:
    lfw = source(file_server, tarball(IMAGES))
    file_server.serve("archives/lfw-funneled.tgz", tarball({**IMAGES, "extra.jpg": b"!"}))

    with pytest.raises(ChecksumMismatchError, match=r"lfw-funneled\.tgz"):
        fetch_lfw(tmp_path, lfw)

    assert not (tmp_path / "lfw" / "lfw_funneled").exists()
    assert not (tmp_path / "lfw" / "archives" / "lfw-funneled.tgz").exists()


def test_an_archive_that_would_write_outside_its_folder_is_refused(
    file_server: FileServer, tmp_path: Path
) -> None:
    hostile = tarball(IMAGES, symlink=("lfw_funneled/escape", "/etc/passwd"))

    with pytest.raises(FetchError, match="escape"):
        fetch_lfw(tmp_path, source(file_server, hostile))

    assert sorted(path.name for path in (tmp_path / "lfw").iterdir()) == [
        "archives",
        "pairs.txt",
        "pairsDevTest.txt",
    ]


def test_an_archive_without_the_expected_folder_is_refused(
    file_server: FileServer, tmp_path: Path
) -> None:
    with pytest.raises(FetchError, match="lfw_funneled"):
        fetch_lfw(tmp_path, source(file_server, tarball({"lfw/Zico/Zico_0001.jpg": b"x"})))

    assert not (tmp_path / "lfw" / "lfw_funneled").exists()


def test_the_pinned_sources_are_the_figshare_files_the_dataset_fetch_recorded() -> None:
    assert LFW.archive.url == "https://ndownloader.figshare.com/files/5976015"
    assert LFW.archive.checksum == Checksum("md5", "1b42dfed7d15c9b2dd63d5e5840c86ad")
    assert {pinned.path.name: pinned.checksum.hexdigest for pinned in LFW.pairs} == {
        "pairs.txt": "9f1ba174e4e1c508ff7cdf10ac338a7d",
        "pairsDevTrain.txt": "4f27cbf15b2da4a85c1907eb4181ad21",
        "pairsDevTest.txt": "5132f7440eb68cf58910c8a45a2ac10b",
    }
