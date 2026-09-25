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
