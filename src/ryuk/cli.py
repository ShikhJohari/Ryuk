"""`ryuk` command line: thin wrappers over the package, excluded from coverage."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from pydantic import ValidationError

from ryuk.api import create_app
from ryuk.api.contract import openapi_schema, render_openapi
from ryuk.datasets import DatasetError
from ryuk.eda.build import ProvenanceError, build_summary
from ryuk.eda.files import FIGURES_DIR, read_summary, write_schema, write_summary
from ryuk.eda.scan import default_workers
from ryuk.fetch import FetchError
from ryuk.fetch.celeba import fetch_celeba
from ryuk.fetch.lfw import fetch_lfw
from ryuk.fetch.pinned import Fetched
from ryuk.logs import configure_logging
from ryuk.plotting.eda import save_eda_figures
from ryuk.settings import Settings
from ryuk.weights import fetch_weights

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="Benchmark datasets (LFW, CelebA).")
weights_app = typer.Typer(no_args_is_help=True, help="Detector and recognition model weights.")
app.add_typer(data_app, name="data")
app.add_typer(weights_app, name="weights")


@app.command()
def serve() -> None:
    """Run the service on RYUK_HOST:RYUK_PORT (loopback only)."""
    settings = _settings()
    configure_logging()
    # log_config=None leaves uvicorn's loggers propagating to the JSON handler.
    uvicorn.run(create_app(), host=settings.host, port=settings.port, log_config=None)


@app.command()
def openapi(
    output: Annotated[Path, typer.Option(help="Where to write the contract.")] = Path(
        "openapi.json"
    ),
) -> None:
    """Write the OpenAPI contract the client's types are generated from."""
    output.write_text(render_openapi(openapi_schema(create_app())))
    typer.echo(f"Wrote {output}")


@data_app.command("fetch")
def fetch_data() -> None:
    """Fetch LFW and CelebA into RYUK_DATA_DIR from pinned sources, verifying every checksum."""
    root = _settings().data_dir
    _report(lambda: [*fetch_lfw(root), *fetch_celeba(root)])


@weights_app.command("fetch")
def fetch_all_weights() -> None:
    """Fetch the detector and recognition weights into RYUK_WEIGHTS_DIR, sha256-verified."""
    weights_dir = _settings().weights_dir
    _report(lambda: fetch_weights(weights_dir))


@app.command()
def eda(
    output: Annotated[Path, typer.Option(help="Where to write the summary and figures.")] = Path(
        "eda"
    ),
    workers: Annotated[
        int, typer.Option(min=1, help="Detector threads; defaults to one per CPU core.")
    ] = default_workers(),
    figures_only: Annotated[
        bool,
        typer.Option(
            help="Regenerate the schema and redraw the figures from the existing summary, "
            "without data."
        ),
    ] = False,
) -> None:
    """Summarise LFW and CelebA from RYUK_DATA_DIR: write summary.json, its schema and figures."""
    configure_logging()
    try:
        if figures_only:
            summary = read_summary(output)
            written = [write_schema(output)]
        else:
            settings = _settings()
            summary = build_summary(
                settings.data_dir, settings.weights_dir, workers=workers, repo=Path.cwd()
            )
            written = write_summary(summary, output)
    except (DatasetError, ProvenanceError, OSError, ValidationError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from None
    written += save_eda_figures(summary, output / FIGURES_DIR)
    typer.echo(f"Wrote {len(written)} files to {output}")


def _report(fetch: Callable[[], list[Fetched]]) -> None:
    configure_logging()
    try:
        fetched = fetch()
    except FetchError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from None
    updated = sum(item.updated for item in fetched)
    typer.echo(f"{len(fetched)} in place and verified, {updated} fetched or rebuilt this run")


def _settings() -> Settings:
    try:
        return Settings()
    except ValidationError as error:
        for problem in error.errors():
            field = "RYUK_" + "_".join(str(part) for part in problem["loc"]).upper()
            typer.echo(f"{field}: {problem['msg']}", err=True)
        raise typer.Exit(code=2) from None
