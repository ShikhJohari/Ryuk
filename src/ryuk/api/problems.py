"""RFC 9457 problem responses.

Every error the service returns is `application/problem+json` with a stable
machine-readable `code`; `title` is the HTTP status phrase and `detail` is for
the operator. Clients branch on `code`, never on `detail`.
"""

import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from ryuk.api.schema import ApiModel

PROBLEM_MEDIA_TYPE = "application/problem+json"

logger = logging.getLogger(__name__)

# RFC 9110 renamed these; Python 3.12's `http.HTTPStatus` still has the old phrases.
_RFC_9110_PHRASES = {413: "Content Too Large", 422: "Unprocessable Content"}


class Problem(ApiModel):
    """An RFC 9457 problem detail with Ryuk's `code` extension."""

    type: str
    title: str
    status: int
    detail: str
    code: str


class ProblemError(Exception):
    """Raise from a route to answer with a problem response."""

    def __init__(self, *, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.problem = problem_for_status(status, detail, code=code)


def status_phrase(status: int) -> str:
    if status in _RFC_9110_PHRASES:
        return _RFC_9110_PHRASES[status]
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Error"  # a status outside the registry, such as 499


def status_code_name(status: int) -> str:
    """`not_found` for 404: the default `code` when nothing more specific applies."""
    return status_phrase(status).lower().replace(" ", "_").replace("-", "_")


def problem_response(problem: Problem) -> JSONResponse:
    return JSONResponse(
        problem.model_dump(by_alias=True),
        status_code=problem.status,
        media_type=PROBLEM_MEDIA_TYPE,
    )


def problem_for_status(status: int, detail: str, code: str | None = None) -> Problem:
    return Problem(
        type="about:blank",
        title=status_phrase(status),
        status=status,
        detail=detail,
        code=code or status_code_name(status),
    )


def install_problem_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ProblemError, _on_problem)
    app.add_exception_handler(HTTPException, _on_http_exception)
    app.add_exception_handler(RequestValidationError, _on_invalid_request)
    app.add_exception_handler(Exception, _on_unhandled)


async def _on_problem(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, ProblemError)  # noqa: S101 - narrows Starlette's handler signature
    return problem_response(exc.problem)


async def _on_http_exception(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, HTTPException)  # noqa: S101
    response = problem_response(problem_for_status(exc.status_code, str(exc.detail)))
    if exc.headers:
        response.headers.update(exc.headers)
    return response


async def _on_invalid_request(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)  # noqa: S101
    detail = "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()
    )
    return problem_response(problem_for_status(422, detail, code="invalid_request"))


async def _on_unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return problem_response(
        problem_for_status(500, "The service hit an unexpected error.", code="internal_error")
    )
