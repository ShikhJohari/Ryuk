from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from file_server import FileServer, running_file_server
from ryuk.api import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    # The service only answers to localhost Host headers, so tests talk to it as 127.0.0.1.
    with TestClient(
        create_app(), base_url="http://127.0.0.1", raise_server_exceptions=False
    ) as test_client:
        yield test_client


@pytest.fixture
def file_server() -> Iterator[FileServer]:
    with running_file_server() as server:
        yield server
