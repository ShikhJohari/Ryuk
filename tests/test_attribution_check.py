"""CI's attribution check: a commit may not credit Claude as its author, its committer or a
co-author (#47 Q17). Driven through the script CI runs, against throwaway repositories."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / ".github" / "scripts" / "check-attribution.sh"
SHIKHAR = ("Shikhar Johari", "shikharjohari6@icloud.com")
GIT = shutil.which("git") or "git"
BASH = shutil.which("bash") or "bash"
ISOLATED = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
"""Keeps the developer's own git config (signing, hooks, identity) out of the repositories."""


def git(
    repo: Path, *args: str, who: tuple[str, str] = SHIKHAR, by: tuple[str, str] | None = None
) -> str:
    committer = by or who
    env = {
        **os.environ,
        **ISOLATED,
        "GIT_AUTHOR_NAME": who[0],
        "GIT_AUTHOR_EMAIL": who[1],
        "GIT_COMMITTER_NAME": committer[0],
        "GIT_COMMITTER_EMAIL": committer[1],
    }
    done = subprocess.run(  # noqa: S603 - fixed arguments, no shell
        [GIT, *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def commit(
    repo: Path, message: str, who: tuple[str, str] = SHIKHAR, by: tuple[str, str] | None = None
) -> str:
    git(repo, "commit", "--allow-empty", "-m", message, who=who, by=by)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository with one clean commit, tagged `base`, for ranges to start from."""
    git(tmp_path, "init", "--quiet")
    commit(tmp_path, "chore: start")
    git(tmp_path, "tag", "base")
    return tmp_path


def check(
    repo: Path, *args: str, github: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run the script as CI does: with `args` (the range `base..HEAD` by default), or with
    `--github` and the event's values in the environment."""
    arguments = ["--github"] if github is not None else list(args or ("base..HEAD",))
    env = {**os.environ, **ISOLATED, **(github or {})}
    return subprocess.run(  # noqa: S603 - fixed arguments, no shell
        [BASH, str(SCRIPT), *arguments],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_commits_by_shikhar_alone_pass(repo: Path) -> None:
    commit(repo, "feat: a change")
    commit(repo, "fix: another\n\nCloses #47")

    assert check(repo).returncode == 0


@pytest.mark.parametrize(
    "trailer",
    [
        pytest.param("Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", id="harness"),
        pytest.param("co-authored-by: Claude <claude@example.com>", id="lower case"),
        pytest.param("Co-authored-by: Someone <noreply@anthropic.com>", id="anthropic address"),
        pytest.param("Co-authored-by : Claude <noreply@anthropic.com>", id="space before colon"),
    ],
)
def test_a_claude_co_author_fails_and_names_the_commit(repo: Path, trailer: str) -> None:
    sha = commit(repo, f"feat: a change\n\nBody.\n\n{trailer}")

    result = check(repo)

    assert result.returncode == 1
    assert sha[:7] in result.stdout
    assert "feat: a change" in result.stdout


def test_claude_as_author_fails(repo: Path) -> None:
    commit(repo, "feat: a change", who=("Claude", "noreply@example.com"))

    assert check(repo).returncode == 1


def test_an_anthropic_committer_fails(repo: Path) -> None:
    commit(repo, "feat: a change", by=("Bot", "noreply@anthropic.com"))

    assert check(repo).returncode == 1


def test_a_human_co_author_passes(repo: Path) -> None:
    commit(repo, "feat: pair work\n\nCo-authored-by: Ada Lovelace <ada@example.com>")

    assert check(repo).returncode == 0


def test_mentioning_claude_outside_a_co_author_line_passes(repo: Path) -> None:
    commit(repo, "docs: Claude never co-authors a commit\n\nSee the Claude Code notes.")

    assert check(repo).returncode == 0


def test_only_the_commits_in_the_range_are_checked(repo: Path) -> None:
    commit(repo, "feat: old\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
    git(repo, "tag", "reviewed")
    commit(repo, "feat: new")

    assert check(repo, "reviewed..HEAD").returncode == 0
    assert check(repo, "base..HEAD").returncode == 1


def test_a_range_that_does_not_resolve_fails_rather_than_checking_nothing(repo: Path) -> None:
    result = check(repo, "0123456789abcdef0123456789abcdef01234567..HEAD")

    assert result.returncode != 0
    assert "::error::" not in result.stdout


def test_a_percent_sign_in_the_subject_is_escaped_for_the_annotation(repo: Path) -> None:
    commit(repo, "fix: 100% wrong\n\nCo-Authored-By: Claude <noreply@anthropic.com>")

    assert "fix: 100%25 wrong" in check(repo).stdout


def test_on_a_pull_request_it_checks_the_commits_between_base_and_head(repo: Path) -> None:
    base = git(repo, "rev-parse", "HEAD")
    head = commit(repo, "feat: a change", who=("Claude", "noreply@anthropic.com"))
    event = {"PR_BASE": base, "PR_HEAD": head, "PUSH_BEFORE": "", "GITHUB_SHA": "unused"}

    assert check(repo, github=event).returncode == 1
    assert check(repo, github={**event, "PR_BASE": head}).returncode == 0


def test_on_a_push_it_checks_the_commits_after_before(repo: Path) -> None:
    offending = commit(repo, "feat: old", who=("Claude", "noreply@anthropic.com"))
    after = commit(repo, "feat: new")

    assert check(repo, github={"PUSH_BEFORE": offending, "GITHUB_SHA": after}).returncode == 0


def test_a_push_that_creates_the_branch_checks_its_last_commit(repo: Path) -> None:
    clean = git(repo, "rev-parse", "HEAD")
    offending = commit(repo, "feat: a change", who=("Claude", "noreply@anthropic.com"))
    zeros = "0" * 40

    assert check(repo, github={"PUSH_BEFORE": zeros, "GITHUB_SHA": offending}).returncode == 1
    assert check(repo, github={"PUSH_BEFORE": zeros, "GITHUB_SHA": clean}).returncode == 0


def test_every_offending_commit_is_reported(repo: Path) -> None:
    first = commit(repo, "feat: one\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
    commit(repo, "feat: clean")
    second = commit(repo, "feat: two", who=("Claude", "noreply@anthropic.com"))

    output = check(repo).stdout

    assert first[:7] in output
    assert second[:7] in output


def test_it_requires_a_range(repo: Path) -> None:
    result = subprocess.run(  # noqa: S603 - fixed arguments, no shell
        [BASH, str(SCRIPT)], cwd=repo, capture_output=True, text=True, check=False
    )

    assert result.returncode == 2
    assert "usage" in result.stderr
