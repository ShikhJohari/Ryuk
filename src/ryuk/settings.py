"""Service settings, read from `RYUK_*` environment variables."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ryuk.loopback import is_loopback_host


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RYUK_", frozen=True)

    # The API has no authentication, so it must never be reachable from another machine.
    host: str = "127.0.0.1"
    port: int = 8000
    # Relative paths resolve against the working directory: run Ryuk from the repository root.
    data_dir: Path = Path("data/raw")
    weights_dir: Path = Path("models/weights")
    # Benchmark embeddings and other derived evaluation data; never the app database.
    cache_dir: Path = Path("data/cache")
    # The watchlist: persons of interest, their photos and embeddings. Holds face images.
    database: Path = Path("data/ryuk.sqlite3")
    # The committed evaluation outputs; the service reads the thresholds from here at startup.
    results: Path = Path("evaluation/results.json")

    @field_validator("host")
    @classmethod
    def _host_is_loopback(cls, host: str) -> str:
        if not is_loopback_host(host):
            raise ValueError(f"{host!r} is not a loopback address; Ryuk binds to localhost only")
        return host
