#!/usr/bin/env bash
# Fails if any commit in a revision range credits Claude as its author, its committer or a
# co-author: every commit is attributed to Shikhar alone (#47 Q17). Only the range is checked,
# so the commits already on main before the rule was enforced are left alone.
#
# Usage:
#   check-attribution.sh <revision range>   e.g. origin/main..HEAD
#   check-attribution.sh --github           the commits a pull request or push adds, from
#                                           PR_BASE, PR_HEAD, PUSH_BEFORE and GITHUB_SHA
set -euo pipefail
# Case-insensitive [[ =~ ]], without bash 4's ${var,,}: macOS ships bash 3.2.
shopt -s nocasematch

usage='usage: check-attribution.sh <revision range> | --github'

github_range() {
  local before=${PUSH_BEFORE:-}
  if [[ -n ${PR_BASE:-} ]]; then
    echo "$PR_BASE..${PR_HEAD:?PR_HEAD is not set}"
  elif [[ -z ${before//0/} ]]; then
    # A push that created the branch: GitHub sends an all-zero `before`.
    echo "${GITHUB_SHA:?GITHUB_SHA is not set}^!"
  else
    echo "$before..${GITHUB_SHA:?GITHUB_SHA is not set}"
  fi
}

case ${1:-} in
  "") echo "$usage" >&2; exit 2 ;;
  --github) range=$(github_range) ;;
  *) range=$1 ;;
esac

claude='claude|anthropic\.com'
co_author="^[[:space:]]*co-authored-by[[:space:]]*:.*($claude)"
status=0

report() {
  local subject
  subject=$(git log -1 --format='%h %s' "$1")
  # `%` starts an escape in a workflow command's message.
  echo "::error::${subject//%/%25} credits Claude ($2). Commits are attributed to Shikhar alone (#47 Q17): drop the line or reset the author, then force-push the branch."
  status=1
}

# On its own line so that an unknown revision, e.g. a `before` lost to a force-push, fails the
# check instead of leaving nothing to check.
shas=$(git rev-list "$range")

for sha in $shas; do
  author=$(git log -1 --format='%an <%ae>' "$sha")
  committer=$(git log -1 --format='%cn <%ce>' "$sha")
  if [[ $author =~ $claude ]]; then report "$sha" "author: $author"; fi
  if [[ $committer =~ $claude ]]; then report "$sha" "committer: $committer"; fi
  # Any co-author line in the message, not only a well-formed trailer block.
  while IFS= read -r line || [[ -n $line ]]; do
    if [[ $line =~ $co_author ]]; then report "$sha" "$line"; fi
  done < <(git log -1 --format='%B' "$sha")
done

exit "$status"
