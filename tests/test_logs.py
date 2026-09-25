import io
import json
import logging
from collections.abc import Iterator

import pytest

from ryuk.logs import configure_logging


@pytest.fixture
def stream() -> Iterator[io.StringIO]:
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    stream = io.StringIO()
    configure_logging(stream=stream)
    yield stream
    root.handlers[:], _ = saved
    root.setLevel(saved[1])


def test_each_record_is_one_json_line(stream: io.StringIO) -> None:
    logging.getLogger("ryuk.test").info("sighting %s opened", "abc")

    record = json.loads(stream.getvalue())
    assert record["level"] == "INFO"
    assert record["logger"] == "ryuk.test"
    assert record["message"] == "sighting abc opened"
    assert record["time"].endswith("Z")


def test_an_exception_is_logged_with_its_traceback(stream: io.StringIO) -> None:
    error = RuntimeError("boom")
    logging.getLogger("ryuk.test").error("failed", exc_info=error)

    record = json.loads(stream.getvalue())
    assert record["message"] == "failed"
    assert "RuntimeError: boom" in record["exception"]
