"""Why a watchlist operation was refused. The API answers each with a problem response."""

from dataclasses import dataclass
from typing import Literal

type WarningCode = Literal["duplicate_name", "looks_like_other", "may_not_be_same_person"]
"""Enrollment warnings, confirmed by resending with the codes acknowledged (#12)."""


class WatchlistError(Exception):
    """A refusal with an HTTP status and a stable machine-readable code."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class EnrollmentWarning:
    code: WarningCode
    detail: str
    person_id: str | None
    """The other person of interest the warning is about, if any."""


class UnacknowledgedWarningsError(WatchlistError):
    """The photo or name raised warnings the operator has not acknowledged yet."""

    def __init__(self, warnings: list[EnrollmentWarning]) -> None:
        super().__init__(409, "warnings", "; ".join(warning.detail for warning in warnings))
        self.warnings = warnings


def not_found(what: str) -> WatchlistError:
    return WatchlistError(404, "not_found", f"No {what} has that ID.")
