"""The service the browser smoke test runs against (`pnpm e2e`, web/playwright.config.ts).

The real service on 127.0.0.1:8000 with the real YuNet detector from the fixtures and a fake
recognition model standing in for SFace, evaluated and active, on a fresh database each run. The
fake stays out of `ryuk serve`: the tests that need a stand-in own it.
"""

import tempfile
from pathlib import Path

import uvicorn

from ryuk.api import create_app
from ryuk.api.frames import MAX_FRAME_MESSAGE_BYTES
from ryuk.detector import Detector
from ryuk.logs import configure_logging
from ryuk.watchlist.database import open_database
from ryuk.watchlist.service import Watchlist, start_watchlist
from synthetic import YUNET, fake
from watchlist_service import evaluated


def main() -> None:
    sface = fake("sface")
    configure_logging()
    with tempfile.TemporaryDirectory(prefix="ryuk-e2e-") as directory:

        def start() -> Watchlist:
            return start_watchlist(
                open_database(Path(directory) / "ryuk.sqlite3"),
                Detector(YUNET),
                [sface],
                evaluated(sface.key, first_active=sface.key),
            )

        uvicorn.run(
            create_app(start),
            host="127.0.0.1",
            port=8000,
            log_config=None,
            ws_max_size=MAX_FRAME_MESSAGE_BYTES,
        )


if __name__ == "__main__":
    main()
