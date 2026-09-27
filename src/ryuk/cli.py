"""`ryuk` command line: thin wrappers over the package, excluded from coverage."""

from collections.abc import Callable
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from pydantic import ValidationError

from ryuk.api import create_app
from ryuk.api.contract import openapi_schema, render_openapi
from ryuk.api.frames import MAX_FRAME_MESSAGE_BYTES
from ryuk.datasets import DatasetError
from ryuk.detector import MIN_USABLE_FACE_SIZE, Detector
from ryuk.eda.build import ProvenanceError, build_summary
from ryuk.eda.files import write_eda, write_from_summary
from ryuk.eda.scan import Scanner, default_workers
from ryuk.evaluation.active import assemble
from ryuk.evaluation.celeba import CelebaEvaluation
from ryuk.evaluation.embeddings import EmbeddingCache
from ryuk.evaluation.provenance import ProvenanceError as ResultsProvenanceError
from ryuk.evaluation.provenance import current_provenance
from ryuk.evaluation.results import (
    Results,
    identification_matches,
    json_schema,
    read_results,
    write_results,
)
from ryuk.evaluation.verification import LfwData, LfwEvaluation, Models, Pipeline
from ryuk.fetch import FetchError
from ryuk.fetch.celeba import fetch_celeba
from ryuk.fetch.lfw import fetch_lfw
from ryuk.fetch.pinned import Fetched, file_checksum
from ryuk.logs import configure_logging
from ryuk.recognition.load import NETWORKS, load_model
from ryuk.recognition.sface import SFace
from ryuk.settings import Settings
from ryuk.watchlist.load import open_watchlist
from ryuk.weights import EVALUATION_WEIGHTS, SFACE_INT8, WEIGHTS, YUNET, fetch_weights

app = typer.Typer(no_args_is_help=True, add_completion=False)
data_app = typer.Typer(no_args_is_help=True, help="Benchmark datasets (LFW, CelebA).")
weights_app = typer.Typer(no_args_is_help=True, help="Detector and recognition model weights.")
app.add_typer(data_app, name="data")
evaluate_app = typer.Typer(
    no_args_is_help=True, help="Measure the recognition models on the benchmarks."
)
app.add_typer(weights_app, name="weights")
app.add_typer(evaluate_app, name="evaluate")

RESULTS = Path("evaluation/results.json")
RESULTS_SCHEMA = Path("evaluation/results.schema.json")


class Dataset(StrEnum):
    lfw = "lfw"
    celeba = "celeba"


@app.command()
def serve() -> None:
    """Run the service on RYUK_HOST:RYUK_PORT (loopback only)."""
    settings = _settings()
    configure_logging()
    # log_config=None leaves uvicorn's loggers propagating to the JSON handler.
    app = create_app(lambda: open_watchlist(settings))
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_config=None,
        ws_max_size=MAX_FRAME_MESSAGE_BYTES,
    )


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
def fetch_data(
    dataset: Annotated[
        list[Dataset] | None,
        typer.Option(help="Fetch only this dataset; repeat for several. Default: all."),
    ] = None,
) -> None:
    """Fetch LFW and CelebA into RYUK_DATA_DIR from pinned sources, verifying every checksum."""
    root = _settings().data_dir
    chosen = set(dataset or Dataset)
    _report(
        lambda: [
            *(fetch_lfw(root) if Dataset.lfw in chosen else []),
            *(fetch_celeba(root) if Dataset.celeba in chosen else []),
        ]
    )


@weights_app.command("fetch")
def fetch_all_weights() -> None:
    """Fetch the detector and recognition weights into RYUK_WEIGHTS_DIR, sha256-verified."""
    weights_dir = _settings().weights_dir
    _report(lambda: fetch_weights(weights_dir, WEIGHTS + EVALUATION_WEIGHTS))


@evaluate_app.command("lfw")
def evaluate_lfw() -> None:
    """Score every recognition model on LFW View 2 and write evaluation/results.json.

    Takes a few minutes on Apple Silicon; embeddings are cached under RYUK_CACHE_DIR.
    """
    settings = _settings()
    configure_logging()
    weights_dir = settings.weights_dir
    try:
        yunet = YUNET.path(weights_dir)
        pipeline = Pipeline(
            detector=Detector(yunet),
            detector_sha256=file_checksum(yunet, "sha256"),
            min_face_size=MIN_USABLE_FACE_SIZE,
            crop="five-point",
        )
        models = Models(
            compared={network: partial(load_model, network, weights_dir) for network in NETWORKS},
            sface_int8=lambda: SFace(SFACE_INT8.path(weights_dir)),
        )
        evaluation = LfwEvaluation(
            LfwData.read(settings.data_dir),
            pipeline,
            EmbeddingCache(settings.cache_dir / "embeddings"),
        )
        verification = evaluation.run(models, current_provenance(Path.cwd()))
    except (OSError, DatasetError, ResultsProvenanceError) as error:
        typer.echo(f"error: {error}", err=True)
        typer.echo(
            "Fetch what is missing with `ryuk weights fetch` and `ryuk data fetch`.", err=True
        )
        raise typer.Exit(code=1) from None
    previous = _previous_results()
    identification = previous.identification if previous is not None else None
    if identification is not None and not identification_matches(verification, identification):
        typer.echo(
            "warning: the models or crops changed, so the CelebA results and thresholds no "
            "longer apply and are dropped; run `ryuk evaluate celeba` again",
            err=True,
        )
        identification = None
    write_results(RESULTS, assemble(verification, identification))
    for model in verification.models:
        flag = "" if model.reproduces_published else "  <- outside 0.5 points of published"
        typer.echo(
            f"{model.model.network:8} {model.accuracy * 100:6.2f} ± "
            f"{model.standard_error * 100:.2f}  (published {model.published.accuracy * 100:.2f})"
            f"{flag}"
        )
    typer.echo(f"Wrote {RESULTS}")


@evaluate_app.command("celeba")
def evaluate_celeba(
    workers: Annotated[
        int, typer.Option(min=1, help="Detector threads for the scan; one per CPU core by default.")
    ] = default_workers(),
) -> None:
    """Rehearse the watchlist on CelebA: freeze each threshold on validation, score test once.

    Needs `ryuk evaluate lfw` first: it chooses each network's crop and judges eligibility.
    Embeddings are cached under RYUK_CACHE_DIR.
    """
    settings = _settings()
    configure_logging()
    previous = _previous_results()
    if previous is None:
        typer.echo(f"error: no {RESULTS}; run `ryuk evaluate lfw` first", err=True)
        raise typer.Exit(code=1)
    crops = {model.model.network: model.crop for model in previous.verification.models}
    if missing := [network for network in NETWORKS if network not in crops]:
        typer.echo(
            f"error: LFW has no result for {', '.join(missing)}; run `ryuk evaluate lfw` first",
            err=True,
        )
        raise typer.Exit(code=1)
    weights_dir = settings.weights_dir
    try:
        yunet = YUNET.path(weights_dir)
        evaluation = CelebaEvaluation(
            root=settings.data_dir,
            pipeline=Pipeline(
                detector=Detector(yunet),
                detector_sha256=file_checksum(yunet, "sha256"),
                min_face_size=MIN_USABLE_FACE_SIZE,
                crop="five-point",
            ),
            cache=EmbeddingCache(settings.cache_dir / "embeddings"),
            scanner=Scanner(yunet, workers),
        )
        identification = evaluation.run(
            {network: partial(load_model, network, weights_dir) for network in NETWORKS},
            crops,
            current_provenance(Path.cwd()),
        )
    except (OSError, DatasetError) as error:
        typer.echo(f"error: {error}", err=True)
        typer.echo(
            "Fetch what is missing with `ryuk weights fetch` and `ryuk data fetch`.", err=True
        )
        raise typer.Exit(code=1) from None
    except (ResultsProvenanceError, ValueError) as error:
        # A draw that cannot be filled, a threshold that cannot be frozen, a stale cache.
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from None
    results = assemble(previous.verification, identification)
    write_results(RESULTS, results)
    for model in identification.models:
        at = model.test.at_threshold
        typer.echo(
            f"{model.model.network:8} threshold {model.threshold:.3f}  test TPIR "
            f"{at.tpir.value * 100:6.2f}  FPIR {at.fpir.value * 100:5.2f}  "
            f"rank-1 {model.test.rank_1.value * 100:6.2f}  {model.ms_per_face:.1f} ms"
        )
    if results.first_active_model is not None:
        typer.echo(results.first_active_model.reason)
    typer.echo(f"Wrote {RESULTS}")


def _previous_results() -> Results | None:
    try:
        return read_results(RESULTS)
    except ValidationError as error:
        typer.echo(f"error: {RESULTS} does not match its schema: {error}", err=True)
        raise typer.Exit(code=1) from None


@evaluate_app.command("schema")
def results_schema() -> None:
    """Write the JSON Schema of evaluation/results.json, generated from its model."""
    RESULTS_SCHEMA.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_SCHEMA.write_text(json_schema())
    typer.echo(f"Wrote {RESULTS_SCHEMA}")


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
            written = write_from_summary(output)
        else:
            settings = _settings()
            summary = build_summary(
                settings.data_dir, settings.weights_dir, workers=workers, repo=Path.cwd()
            )
            written = write_eda(summary, output)
    # Writes go through write_into_place, which reports a failed write as a FetchError.
    except (DatasetError, ProvenanceError, FetchError, OSError, ValidationError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from None
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
