"""Enrolling persons of interest through the HTTP seam: rejections, warnings and acknowledgement."""

import io
import struct
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from starlette.types import Message

from ryuk.api.uploads import MAX_UPLOAD_BYTES
from synthetic import fake
from watchlist_service import (
    blank,
    distant,
    encode,
    evaluated,
    group,
    portrait,
    serve,
    upload,
    with_bystander,
)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    model = fake("sface")
    evaluation = evaluated(model.key, first_active=model.key)
    with serve(tmp_path / "ryuk.sqlite3", [model], evaluation) as served:
        yield served


def enroll(client: TestClient, name: str, photo: Any, acknowledged: list[str] | None = None) -> Any:
    return client.post(
        "/api/persons",
        data={"name": name, "acknowledgedWarnings": acknowledged or []},
        files=upload(photo),
    )


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png_header(width: int, height: int) -> bytes:
    """A PNG that declares its size but has no pixel data: decoding it would fail, so a refusal
    by size shows the size was checked before any decoding."""
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header) + png_chunk(b"IEND", b"")


def corrupt_png() -> bytes:
    """A PNG whose first IDAT chunk declares the wrong length, which Pillow reports as a
    SyntaxError rather than an OSError."""
    data = bytearray(encode(portrait(0), "PNG"))
    assert data[37:41] == b"IDAT"  # the signature and IHDR take the first 33 bytes
    data[36] = 16
    return bytes(data)


def test_a_person_of_interest_is_created_from_a_name_and_one_photo(client: TestClient) -> None:
    response = enroll(client, "Ada Lovelace", portrait(0))

    assert response.status_code == 201
    person = response.json()
    assert person["name"] == "Ada Lovelace"
    assert person["status"] == "on_watchlist"
    assert [photo["mediaType"] for photo in person["photos"]] == ["image/jpeg"]
    assert client.get(f"/api/persons/{person['id']}").json() == person
    [listed] = client.get("/api/persons").json()
    assert listed["id"] == person["id"]
    assert listed["photoCount"] == 1
    assert listed["coverPhotoId"] == person["photos"][0]["id"]


@pytest.mark.parametrize(
    ("image_format", "media_type"),
    [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")],
)
def test_jpeg_png_and_webp_are_accepted_and_keep_their_format(
    client: TestClient, image_format: str, media_type: str
) -> None:
    response = enroll(client, "Ada", encode(portrait(0), image_format))

    assert response.status_code == 201
    [photo] = response.json()["photos"]
    image = client.get(f"/api/persons/{response.json()['id']}/photos/{photo['id']}/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == media_type
    assert image.headers["cache-control"] == "no-store"
    assert (photo["width"], photo["height"]) == (400, 400)


def mpo(*frames: Any) -> bytes:
    """A multi-picture JPEG, as some cameras and phones write, which Pillow reads as MPO."""
    first, *rest = (PILImage.fromarray(np.ascontiguousarray(f[:, :, ::-1])) for f in frames)
    buffer = io.BytesIO()
    first.save(buffer, format="MPO", save_all=True, append_images=rest)
    return buffer.getvalue()


def test_a_multi_picture_jpeg_is_enrolled_from_its_first_picture_as_a_jpeg(
    client: TestClient,
) -> None:
    # The second picture has no face, so enrolling it would be no_face.
    photo = mpo(portrait(0), blank())
    with PILImage.open(io.BytesIO(photo)) as opened:
        assert opened.format == "MPO"

    response = enroll(client, "Ada", photo)

    assert response.status_code == 201
    [stored_photo] = response.json()["photos"]
    assert stored_photo["mediaType"] == "image/jpeg"
    image = client.get(f"/api/persons/{response.json()['id']}/photos/{stored_photo['id']}/image")
    assert image.headers["content-type"] == "image/jpeg"
    with PILImage.open(io.BytesIO(image.content)) as stored:
        assert stored.format == "JPEG"
        assert stored.size == (400, 400)
        assert getattr(stored, "n_frames", 1) == 1


@pytest.mark.parametrize(
    ("photo", "status", "code"),
    [
        (encode(blank()), 422, "no_face"),
        (encode(group(0, 1)), 422, "multiple_faces"),
        (encode(distant(0)), 422, "face_too_small"),
        (b"not an image at all", 422, "unsupported_image"),
        (encode(portrait(0), "GIF"), 422, "unsupported_image"),
        (encode(portrait(0), "BMP"), 422, "unsupported_image"),
        (corrupt_png(), 422, "unsupported_image"),
        (b"\xff\xd8\xff" + bytes(10 * 1024 * 1024), 413, "photo_too_large"),
        (png_header(7000, 7000), 413, "photo_too_large"),
    ],
    ids=[
        "no-face",
        "two-faces",
        "small-face",
        "garbage",
        "gif",
        "bmp",
        "corrupt-png",
        "over-10-mb",
        "over-40-megapixels",
    ],
)
def test_a_photo_that_cannot_be_enrolled_is_rejected_and_nobody_is_created(
    client: TestClient, photo: bytes, status: int, code: str
) -> None:
    response = enroll(client, "Ada", photo)

    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["code"] == code
    assert client.get("/api/persons", params={"status": "all"}).json() == []


@pytest.mark.parametrize(
    "size",
    # 49 MP, then 400 MP: far enough over Pillow's own bomb limit that it refuses the header.
    [(7000, 7000), (20000, 20000)],
    ids=["over-the-limit", "decompression-bomb"],
)
def test_a_photo_over_the_pixel_limit_is_refused_before_it_is_decoded(
    client: TestClient, size: tuple[int, int]
) -> None:
    response = enroll(client, "Ada", png_header(*size))

    assert response.status_code == 413
    assert response.json()["code"] == "photo_too_large"
    assert response.json()["detail"] == "The photo is over 40 megapixels."


def test_a_body_declared_over_the_ceiling_is_refused_before_the_route_runs(
    client: TestClient,
) -> None:
    # A photo that would enroll, sent with a Content-Length over the ceiling: the route never
    # sees it, so nobody is created.
    response = client.post(
        "/api/persons",
        data={"name": "Ada"},
        files=upload(portrait(0)),
        headers={"content-length": str(MAX_UPLOAD_BYTES + 1)},
    )

    assert response.status_code == 413
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["connection"] == "close"
    assert response.json()["code"] == "photo_too_large"
    assert client.get("/api/persons", params={"status": "all"}).json() == []


def test_a_chunked_body_stops_being_read_once_it_passes_the_ceiling(client: TestClient) -> None:
    # The test client reads a streamed body whole, so the app is called directly to watch how
    # much of a chunked body (no Content-Length) it reads.
    mebibyte = 1024 * 1024
    head = (
        b'--b\r\nContent-Disposition: form-data; name="name"\r\n\r\nAda\r\n'
        b'--b\r\nContent-Disposition: form-data; name="photo"; filename="photo.jpg"\r\n'
        b"Content-Type: image/jpeg\r\n\r\n\xff\xd8\xff"
    )
    reads = 0
    sent: list[Message] = []

    async def receive() -> Message:
        nonlocal reads
        reads += 1
        if reads == 1:
            return {"type": "http.request", "body": head, "more_body": True}
        if reads <= 64:
            return {"type": "http.request", "body": bytes(mebibyte), "more_body": True}
        return {"type": "http.request", "body": b"\r\n--b--\r\n", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/persons",
        "raw_path": b"/api/persons",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"127.0.0.1"),
            (b"content-type", b"multipart/form-data; boundary=b"),
            (b"transfer-encoding", b"chunked"),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 80),
    }

    anyio.run(client.app, scope, receive, send)

    # The head and 11 MiB of photo: the read that passes the ceiling is the last.
    assert reads == 1 + MAX_UPLOAD_BYTES // mebibyte + 1
    start, body = sent
    assert start["status"] == 413
    assert (b"connection", b"close") in start["headers"]
    assert b'"code":"photo_too_large"' in body["body"]
    assert client.get("/api/persons", params={"status": "all"}).json() == []


def test_a_bystanders_small_face_is_ignored(client: TestClient) -> None:
    response = enroll(client, "Ada", with_bystander(0, bystander=1))

    assert response.status_code == 201


@pytest.mark.parametrize("name", ["", "   ", "x" * 201])
def test_a_person_of_interest_needs_a_name(client: TestClient, name: str) -> None:
    response = enroll(client, name, portrait(0))

    assert response.status_code == 422
    assert response.json()["code"] in {"invalid_name", "invalid_request"}


@pytest.mark.parametrize(
    "name",
    [
        "Ada\x00Lovelace",
        "Ada\tLovelace",
        "Ada\nLovelace",
        "Ada\x1b[31m",
        "Ada\x7f",
        "Ada\x85Lovelace",
        "Ada\x9b",
        "Ada \u202eecalevoL",
        "\u202aAda\u202c",
        "Ada \u2066Lovelace\u2069",
        "Ada \u2067Lovelace",
    ],
    ids=[
        "nul",
        "tab",
        "newline",
        "escape",
        "delete",
        "c1-next-line",
        "c1-csi",
        "rlo",
        "lre",
        "lri",
        "rli",
    ],
)
def test_a_name_with_control_or_text_direction_characters_is_refused(
    client: TestClient, name: str
) -> None:
    response = enroll(client, name, portrait(0))

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_name"
    assert client.get("/api/persons", params={"status": "all"}).json() == []


def test_a_name_is_trimmed_and_its_spaces_collapsed(client: TestClient) -> None:
    response = enroll(client, "  Ada   Lovelace ", portrait(0))

    assert response.json()["name"] == "Ada Lovelace"


def test_a_duplicate_name_warns_until_acknowledged(client: TestClient) -> None:
    first = enroll(client, "Ada Lovelace", portrait(0)).json()

    warned = enroll(client, " ada  LOVELACE", portrait(1))

    assert warned.status_code == 409
    problem = warned.json()
    assert problem["code"] == "warnings"
    assert [(w["code"], w["personId"]) for w in problem["warnings"]] == [
        ("duplicate_name", first["id"])
    ]
    assert "Ada Lovelace" in problem["warnings"][0]["detail"]
    assert len(client.get("/api/persons").json()) == 1

    acknowledged = enroll(client, " ada  LOVELACE", portrait(1), ["duplicate_name"])

    assert acknowledged.status_code == 201
    assert len(client.get("/api/persons").json()) == 2


def test_a_photo_that_looks_like_another_person_of_interest_warns(client: TestClient) -> None:
    first = enroll(client, "Ada", portrait(0)).json()

    warned = enroll(client, "Grace", portrait(0, shot=3))

    assert warned.status_code == 409
    assert [(w["code"], w["personId"]) for w in warned.json()["warnings"]] == [
        ("looks_like_other", first["id"])
    ]
    assert enroll(client, "Grace", portrait(0, shot=3), ["looks_like_other"]).status_code == 201


def test_a_different_face_does_not_warn(client: TestClient) -> None:
    enroll(client, "Ada", portrait(0))

    assert enroll(client, "Grace", portrait(1)).status_code == 201


def test_every_warning_is_listed_until_all_are_acknowledged(client: TestClient) -> None:
    enroll(client, "Ada", portrait(0))

    partly = enroll(client, "ada", portrait(0, shot=3), ["duplicate_name"])

    assert partly.status_code == 409
    assert {w["code"] for w in partly.json()["warnings"]} == {"duplicate_name", "looks_like_other"}
    fully = enroll(client, "ada", portrait(0, shot=3), ["duplicate_name", "looks_like_other"])
    assert fully.status_code == 201


def test_an_unknown_warning_code_is_an_invalid_request(client: TestClient) -> None:
    response = enroll(client, "Ada", portrait(0), ["whatever"])

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_face_warnings_are_skipped_when_no_model_can_be_active(tmp_path: Path) -> None:
    model = fake("sface")
    with serve(tmp_path / "ryuk.sqlite3", [model], evaluated()) as client:
        enroll(client, "Ada", portrait(0))

        response = enroll(client, "Grace", portrait(0, shot=3))

        assert response.status_code == 201
        # The name check needs no model, so it still applies.
        assert enroll(client, "ADA", portrait(1)).json()["code"] == "warnings"


def test_enrollment_needs_the_detector_but_the_watchlist_still_reads(tmp_path: Path) -> None:
    model = fake("sface")
    with serve(tmp_path / "ryuk.sqlite3", [model], evaluated(), detector=False) as client:
        response = enroll(client, "Ada", portrait(0))

        assert response.status_code == 503
        assert response.json()["code"] == "no_detector"
        assert client.get("/api/persons").json() == []
