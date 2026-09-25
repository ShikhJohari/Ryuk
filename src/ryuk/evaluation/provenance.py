"""The provenance stamped on every results section: commit, working tree state, time, machine."""

import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ryuk.evaluation.results import Provenance


class ProvenanceError(RuntimeError):
    """The commit a result came from cannot be established."""


def current_provenance(repository: Path) -> Provenance:
    return Provenance(
        commit=_git(repository, "rev-parse", "HEAD"),
        dirty=bool(_git(repository, "status", "--porcelain")),
        generated_at=datetime.now(UTC),
        machine=f"{platform.platform()} ({platform.machine()})",
    )


def _git(repository: Path, *arguments: str) -> str:
    git = shutil.which("git")
    if git is None:
        raise ProvenanceError("git is not on PATH; results must name the commit they came from")
    try:
        completed = subprocess.run(  # noqa: S603 - fixed arguments, no shell
            [git, *arguments], cwd=repository, capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as error:
        raise ProvenanceError(
            f"git {' '.join(arguments)} failed: {error.stderr.strip()}"
        ) from error
    return completed.stdout.strip()
