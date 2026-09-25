from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ryuk.api import create_app
from ryuk.settings import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    # The service only answers to localhost Host headers, so tests talk to it as 127.0.0.1.
    with TestClient(
        create_app(settings), base_url="http://127.0.0.1", raise_server_exceptions=False
    ) as test_client:
        yield test_client
