"""`ryuk` command line: thin wrappers over the package, excluded from coverage."""

from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from pydantic import ValidationError

from ryuk.api import create_app
from ryuk.api.contract import openapi_schema, render_openapi
from ryuk.logs import configure_logging
from ryuk.settings import Settings

app = typer.Typer(no_args_is_help=True, add_completion=False)


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


def _settings() -> Settings:
    try:
        return Settings()
    except ValidationError as error:
        for problem in error.errors():
            field = "RYUK_" + "_".join(str(part) for part in problem["loc"]).upper()
            typer.echo(f"{field}: {problem['msg']}", err=True)
        raise typer.Exit(code=2) from None
