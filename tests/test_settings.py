from pathlib import Path

import pytest
from pydantic import ValidationError

from ryuk.settings import Settings


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
def test_the_service_binds_to_loopback(host: str) -> None:
    assert Settings(host=host).host == host


@pytest.mark.parametrize(
    "host", ["0.0.0.0", "::", "192.168.1.20", "10.0.0.5", "ryuk.example.com", ""]
)
def test_the_service_refuses_to_bind_beyond_localhost(host: str) -> None:
    with pytest.raises(ValidationError, match="loopback"):
        Settings(host=host)


def test_the_bind_address_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RYUK_HOST", "0.0.0.0")

    with pytest.raises(ValidationError, match="loopback"):
        Settings()


def test_data_and_weights_live_at_pinned_paths_in_the_repository() -> None:
    settings = Settings()

    assert settings.data_dir == Path("data/raw")
    assert settings.weights_dir == Path("models/weights")


def test_the_weights_directory_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RYUK_WEIGHTS_DIR", str(tmp_path))

    assert Settings().weights_dir == tmp_path
