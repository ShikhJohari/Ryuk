"""`ryuk` command line: thin wrappers over the package, excluded from coverage."""

from collections.abc import Callable, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

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
from ryuk.eda.summary import Draw
from ryuk.evaluation.active import Carried, assemble, carried
from ryuk.evaluation.celeba import CelebaEvaluation
from ryuk.evaluation.draws import DrawMismatchError
from ryuk.evaluation.embeddings import EmbeddingCache
from ryuk.evaluation.provenance import ProvenanceError as ResultsProvenanceError
from ryuk.evaluation.provenance import current_provenance
from ryuk.evaluation.results import (
    Identification,
    Results,
    identification_mismatch,
    json_schema,
    model_changes,
    read_results,
    write_results,
)
from ryuk.evaluation.verification import LfwData, LfwEvaluation, Models, Pipeline
from ryuk.fetch import FetchError
from ryuk.fetch.celeba import fetch_celeba
from ryuk.fetch.lfw import fetch_lfw
from ryuk.fetch.pinned import Fetched, file_checksum
from ryuk.logs import configure_logging
from ryuk.recognition import ModelKey, Network, RecognitionModel
from ryuk.recognition.load import NETWORKS, load_model
from ryuk.recognition.sface import SFace
from ryuk.settings import Settings
from ryuk.watchlist.errors import StartupError
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

RESULTS_SCHEMA = Path("evaluation/results.schema.json")


class Dataset(StrEnum):
    lfw = "lfw"
    celeba = "celeba"


@app.command()
def serve() -> None:
    """Run the service on RYUK_HOST:RYUK_PORT (loopback only)."""
    settings = _settings()
    configure_logging()
    # Opened before uvicorn starts, so a refusal is one line on stderr, not a lifespan traceback.
    try:
        watchlist = open_watchlist(settings)
    except StartupError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from None
    # log_config=None leaves uvicorn's loggers propagating to the JSON handler.
    app = create_app(lambda: watchlist)
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
def evaluate_lfw(
    replace_identification: Annotated[
        bool,
        typer.Option(
            help="Drop the CelebA results, thresholds and first active model when the models "
            "or crops no longer match them. Without it such a run fails and writes nothing."
        ),
    ] = False,
) -> None:
    """Score every recognition model on LFW View 2 and write RYUK_RESULTS (evaluation/results.json).

    Takes a few minutes on an Apple Silicon Mac, longer on CPU only; embeddings are cached under
    RYUK_CACHE_DIR.
    """
    settings = _settings()
    configure_logging()
    results_path = settings.results
    previous = _previous_results(results_path)
    identification = previous.identification if previous is not None else None
    weights_dir = settings.weights_dir
    loaded = _load_models(weights_dir)
    if identification is not None and (
        changes := model_changes(_keys(loaded), [m.model for m in identification.models])
    ):
        _drop_identification(changes, results_path, replace_identification)
        identification = None
    try:
        yunet = YUNET.path(weights_dir)
        pipeline = Pipeline(
            detector=Detector(yunet),
            detector_sha256=file_checksum(yunet, "sha256"),
            min_face_size=MIN_USABLE_FACE_SIZE,
            crop="five-point",
        )
        models = Models(
            compared=_preloaded(loaded),
            sface_int8=lambda: SFace(SFACE_INT8.path(weights_dir)),
        )
        evaluation = LfwEvaluation(
            LfwData.read(settings.data_dir),
            pipeline,
            EmbeddingCache(settings.cache_dir / "embeddings"),
        )
        verification = evaluation.run(models, current_provenance(Path.cwd()))
    except (OSError, DatasetError, ResultsProvenanceError) as error:
        _missing(error)
    if identification is not None and (
        mismatch := identification_mismatch(verification, identification)
    ):
        _drop_identification([mismatch], results_path, replace_identification)
        identification = None
    kept = _carried(identification, previous)
    write_results(results_path, assemble(verification, identification, kept.learning, kept.bias))
    for model in verification.models:
        flag = "" if model.reproduces_published else "  <- outside 0.5 points of published"
        typer.echo(
            f"{model.model.network:8} {model.accuracy * 100:6.2f} ± "
            f"{model.standard_error * 100:.2f}  (published {model.published.accuracy * 100:.2f})"
            f"{flag}"
        )
    typer.echo(f"Wrote {results_path}")


@evaluate_app.command("celeba")
def evaluate_celeba(
    workers: Annotated[
        int, typer.Option(min=1, help="Detector threads for the scan; one per CPU core by default.")
    ] = default_workers(),
    redraw: Annotated[
        bool,
        typer.Option(
            help="Make the draws afresh even where they differ from the ones in RYUK_RESULTS. "
            "Without it, each draw rebuilt must be the one recorded there."
        ),
    ] = False,
) -> None:
    """Rehearse the watchlist on CelebA: freeze each threshold on validation, score test once.

    Needs `ryuk evaluate lfw` first: it chooses each network's crop and judges eligibility.
    Embeddings are cached under RYUK_CACHE_DIR.
    """
    settings = _settings()
    configure_logging()
    results_path = settings.results
    previous = _previous_results(results_path)
    if previous is None:
        typer.echo(f"error: no {results_path}; run `ryuk evaluate lfw` first", err=True)
        raise typer.Exit(code=1)
    crops = {model.model.network: model.crop for model in previous.verification.models}
    if missing := [network for network in NETWORKS if network not in crops]:
        typer.echo(
            f"error: LFW has no result for {', '.join(missing)}; run `ryuk evaluate lfw` first",
            err=True,
        )
        raise typer.Exit(code=1)
    weights_dir = settings.weights_dir
    loaded = _load_models(weights_dir)
    if changes := model_changes(_keys(loaded), [m.model for m in previous.verification.models]):
        # Checked before anything runs: the results could never hold CelebA thresholds for
        # models LFW did not score.
        typer.echo(
            f"error: the recognition models are not the ones LFW scored in {results_path}:",
            err=True,
        )
        for change in changes:
            typer.echo(f"  {change}", err=True)
        typer.echo("Run `ryuk evaluate lfw` first. Nothing was run or written.", err=True)
        raise typer.Exit(code=1)
    committed = previous.identification
    selections = (
        {}
        if redraw or committed is None
        else {draw.draw: draw.selection_sha256 for draw in committed.draws}
    )
    try:
        evaluation = _celeba(settings, workers, selections)
        identification = evaluation.run(_preloaded(loaded), crops, current_provenance(Path.cwd()))
        kept = _carried(identification, previous)
        results = assemble(previous.verification, identification, kept.learning, kept.bias)
    except (OSError, DatasetError) as error:
        _missing(error)
    except DrawMismatchError as error:
        typer.echo(f"error: {error}", err=True)
        typer.echo(
            "Nothing was embedded or written. Pass --redraw to score new draws in their place.",
            err=True,
        )
        raise typer.Exit(code=1) from None
    except (ResultsProvenanceError, ValueError) as error:
        # A draw that cannot be filled, a threshold that cannot be frozen, a stale cache,
        # results that do not validate.
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from None
    write_results(results_path, results)
    for model in identification.models:
        at = model.test.at_threshold
        typer.echo(
            f"{model.model.network:8} threshold {model.threshold:.3f}  test TPIR "
            f"{at.tpir.value * 100:6.2f}  FPIR {at.fpir.value * 100:5.2f}  "
            f"rank-1 {model.test.rank_1.value * 100:6.2f}  {model.ms_per_face:.1f} ms"
        )
    if results.first_active_model is not None:
        typer.echo(results.first_active_model.reason)
    typer.echo(f"Wrote {results_path}")


def _previous_results(path: Path) -> Results | None:
    try:
        return read_results(path)
    except ValidationError as error:
        typer.echo(f"error: {path} does not match its schema: {error}", err=True)
        raise typer.Exit(code=1) from None


def _load_models(weights_dir: Path) -> dict[Network, RecognitionModel]:
    """Every compared network, loaded once up front so its key can be checked before any run."""
    try:
        return {network: load_model(network, weights_dir) for network in NETWORKS}
    except OSError as error:
        _missing(error)


def _keys(models: Mapping[Network, RecognitionModel]) -> dict[Network, ModelKey]:
    return {network: model.key for network, model in models.items()}


def _preloaded(
    models: Mapping[Network, RecognitionModel],
) -> dict[Network, Callable[[], RecognitionModel]]:
    def given(model: RecognitionModel) -> Callable[[], RecognitionModel]:
        return lambda: model

    return {network: given(model) for network, model in models.items()}


def _carried(identification: Identification | None, previous: Results | None) -> Carried:
    """The previous learning and bias sections that still apply, warning about any dropped."""
    kept = carried(
        identification,
        None if previous is None else previous.learning,
        None if previous is None else previous.bias,
    )
    for reason in kept.dropped:
        typer.echo(f"warning: {reason}", err=True)
    return kept


def _drop_identification(reasons: list[str], results_path: Path, replace: bool) -> None:
    """Warn that the CelebA results are being dropped, or, without `replace`, refuse to."""
    if replace:
        typer.echo(
            f"warning: {'; '.join(reasons)}. The CelebA results, thresholds and first active "
            "model no longer apply and are dropped; run `ryuk evaluate celeba` again",
            err=True,
        )
        return
    typer.echo(
        f"error: the CelebA results in {results_path} were measured on other models or crops:",
        err=True,
    )
    for reason in reasons:
        typer.echo(f"  {reason}", err=True)
    typer.echo(
        "Writing LFW now would drop every CelebA result, threshold and the first active model. "
        "Nothing was written. Pass --replace-identification to drop them and then run `ryuk "
        "evaluate celeba`, or evaluate on the machine those results came from.",
        err=True,
    )
    raise typer.Exit(code=1)


def _missing(error: Exception) -> NoReturn:
    typer.echo(f"error: {error}", err=True)
    typer.echo("Fetch what is missing with `ryuk weights fetch` and `ryuk data fetch`.", err=True)
    raise typer.Exit(code=1) from None


@evaluate_app.command("learn")
def evaluate_learn(
    workers: Annotated[
        int, typer.Option(min=1, help="Detector threads for the scan; one per CPU core by default.")
    ] = default_workers(),
) -> None:
    """Compare learning on frozen embeddings with best-photo on the CelebA draws (#10).

    Needs `ryuk evaluate celeba` first, and rebuilds its draws exactly. Fits on validation only,
    scores the test draw once, records each model's live rule, and writes per-probe scores
    under RYUK_CACHE_DIR/scores. The bias breakdown is dropped: run `ryuk evaluate bias` after.
    """
    settings = _settings()
    configure_logging()
    results_path = settings.results
    previous, identification = _identified(results_path)
    loaded = _load_models(settings.weights_dir)
    try:
        evaluation = _celeba(settings, workers, _selections(identification))
        learning = evaluation.learn(
            _preloaded(loaded),
            identification,
            current_provenance(Path.cwd()),
            settings.cache_dir / "scores",
        )
        results = assemble(previous.verification, identification, learning)
    except (OSError, DatasetError) as error:
        _missing(error)
    except (DrawMismatchError, ResultsProvenanceError, RuntimeError, ValueError) as error:
        # A draw that is not the committed one, a threshold that cannot be frozen, a fit that
        # did not converge, results that do not validate.
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from None
    if previous.bias is not None:
        typer.echo(
            "warning: the bias breakdown is dropped, as the live rules it covers may have "
            "changed; run `ryuk evaluate bias` again",
            err=True,
        )
    write_results(results_path, results)
    for model in learning.models:
        typer.echo(f"{model.model.network:8} live rule {model.live_rule}: {model.live_reason}")
    typer.echo(f"Wrote {results_path}")


@evaluate_app.command("bias")
def evaluate_bias(
    workers: Annotated[
        int, typer.Option(min=1, help="Detector threads for the scan; one per CPU core by default.")
    ] = default_workers(),
) -> None:
    """Break the test draw's rates down by group at each model's single frozen threshold (#10).

    Needs `ryuk evaluate learn` first, for each model's live rule.
    """
    settings = _settings()
    configure_logging()
    results_path = settings.results
    previous, identification = _identified(results_path)
    if previous.learning is None:
        typer.echo(
            f"error: no learning in {results_path}; run `ryuk evaluate learn` first", err=True
        )
        raise typer.Exit(code=1)
    loaded = _load_models(settings.weights_dir)
    try:
        evaluation = _celeba(settings, workers, _selections(identification))
        bias = evaluation.bias(
            _preloaded(loaded), identification, previous.learning, current_provenance(Path.cwd())
        )
        results = assemble(previous.verification, identification, previous.learning, bias)
    except (OSError, DatasetError) as error:
        _missing(error)
    except (DrawMismatchError, ResultsProvenanceError, ValueError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from None
    write_results(results_path, results)
    for model in bias.models:
        ratios = ", ".join(
            f"{a.attribute} {'—' if a.fpir_ratio is None else f'{a.fpir_ratio:.2f}'}"
            for a in model.attributes
        )
        typer.echo(f"{model.model.network:8} {model.rule:10} worst/best FPIR: {ratios}")
    typer.echo(f"Wrote {results_path}")


def _identified(results_path: Path) -> tuple[Results, Identification]:
    """The committed results, which must hold identification."""
    previous = _previous_results(results_path)
    if previous is None or previous.identification is None:
        typer.echo(
            f"error: no CelebA results in {results_path}; run `ryuk evaluate celeba` first",
            err=True,
        )
        raise typer.Exit(code=1)
    return previous, previous.identification


def _selections(identification: Identification) -> dict[Draw, str]:
    return {draw.draw: draw.selection_sha256 for draw in identification.draws}


def _celeba(settings: Settings, workers: int, selections: Mapping[Draw, str]) -> CelebaEvaluation:
    yunet = YUNET.path(settings.weights_dir)
    return CelebaEvaluation(
        root=settings.data_dir,
        pipeline=Pipeline(
            detector=Detector(yunet),
            detector_sha256=file_checksum(yunet, "sha256"),
            min_face_size=MIN_USABLE_FACE_SIZE,
            crop="five-point",
        ),
        cache=EmbeddingCache(settings.cache_dir / "embeddings"),
        scanner=Scanner(yunet, workers),
        selections=selections,
    )


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
