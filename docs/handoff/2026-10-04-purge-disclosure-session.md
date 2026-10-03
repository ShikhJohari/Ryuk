# Purge disclosure session, 4 October 2026

The record of a small docs session. A lead session had #51 (#47 Q13: disclose only what purge can and cannot erase) written by an implementer agent in its own worktree, fact-checked by a reviewer agent against the code and primary sources, and applied what it raised. It ran beside #52, in a separate worktree and PR. Read the photo delete and rename session record first. GitHub issues are the source of truth for state; this file is the narrative.

## State

- #51 is done on `ShikharJohari/51-purge-disclosure` (worktree `~/.t3/worktrees/Ryuk/51-purge-disclosure`, from `main` at `d7d160d`). Docs only; no code. Nothing merges without Shikhar's go-ahead (#47 Q16).
- The report renders both PDFs with no unresolved references; every commit passes the attribution check.

## What changed

- **ADR 0004** gains `## Amendment (2026-09-29, #47 Q13)`, in ADR 0003's convention. What purge erases, as the code does it: one transaction clearing the person as runner-up, then the cascade to photos, embeddings and sightings with their crops; `secure_delete` zeroing freed pages; the journal truncated at commit (`journal_mode=TRUNCATE`, #43). What it cannot, "at least": filesystem snapshots (APFS, Time Machine's hourly, last-backup and pre-update local snapshots, btrfs), blocks written elsewhere, blocks the journal freed (and the journals deleted before #43), an upload's spooled temporary file, and an SSD's unerased flash. Why neither moving the database nor `VACUUM` would help.
- **Report.** Section 8 gains "What purge cannot erase" (8.9, `@sec-limitation-purge`), before "No authentication". The Section 6 purge outline no longer promises "freed pages overwritten" and lists what purge cannot reach; Section 1's outline says purge erases from the database, not the disk, and the operator's need reads "erased from the application completely, with an honest account of what lies beyond it". `progress.qmd` credits #51 and no longer says #31 "will" add purge.
- **README.** Purge erases from the database; snapshots, backups and freed blocks are beyond its reach, with a link to ADR 0004.

## Interpretations

- **Two residues beyond the ruling's list are disclosed.** Starlette 1.7.0 spools an upload over 1 MB to an unnamed temporary file whose blocks are freed, not overwritten (`formparsers.py`, `spool_max_size`); it holds the photo as sent, metadata included, which ADR 0003 otherwise never keeps. And an SSD may keep old bytes in flash. Both are the same kind of residue as the journal; the ruling excluded neither. Closing the upload one in code (an in-memory limit) would be a new ticket.
- **#43, not #59**, is cited for `TRUNCATE`: #59 is the PR, `8d1b85d` and the repo's other fix commits cite the issue.
- **SSD wording stays general.** How the Mac's per-volume encryption bears on flash remanence is not stated, since no source makes it precise for this case.

## Review

A reviewer agent checked every claim against the code, sqlite.org, Apple's and btrfs's docs, and the installed Starlette, and ran the purge SQL under inotify to confirm no statement journal or other SQLite temp file is written. Nothing blocking. Applied: Time Machine's full retention, the SSD bullet and "at least", "local snapshots cover the whole startup disk, folders excluded from Time Machine included", "a Mac" rather than "the Mac" in the report, btrfs snapshots described as point-in-time copies of one subvolume, the test's claim narrowed to the samples it reads, the upload wording, the ruling's date, Section 1's "truly erased", Section 6's outline, the README sentence and the progress note.

## Resuming

Shikhar reviews the PR (`Closes #51`) and merges it with a merge commit. Working rules as before: one ticket per session; nothing merged without Shikhar; role agents only, never Haiku; never Playwright locally; write the handoff before merge.
