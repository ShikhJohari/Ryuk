---
status: accepted
---

# Face data lives inside SQLite, erased with secure_delete

Enrolled photos, embeddings and sighting crops are stored as BLOBs in the same SQLite database as the records that own them, not as files beside it, and the database runs with `PRAGMA secure_delete` on. Purge has to erase a person of interest and everything derived from them completely; with every byte in one database, a purge is one transaction that either happens entirely or not at all, and secure_delete overwrites the freed pages instead of leaving the faces recoverable in the file. At Ryuk's scale (hundreds of persons of interest, a few thousand photos, embeddings of 0.5 to 2 KB) BLOB storage costs nothing noticeable.

## Considered options

- Photos and crops as files, embeddings as sidecar `.npy`, the database holding paths. Rejected: a purge interrupted between the database commit and the file deletions leaves face photos on disk that nothing references, which is the worst place for a privacy failure to hide.
- Embeddings as BLOBs, photos as files. Rejected for the same reason; the photos are the more sensitive half.

## Consequences

The database file holds face images, so backing it up or copying it copies them; the report's ethics section says so. The live monitor never searches the database directly: the active model's embeddings for persons on the watchlist are loaded into memory at startup and after each watchlist change.

## Amendment (2026-10-04, #47 Q13)

Purge erases a person of interest from the database file, not from the disk beneath it. `DELETE /api/persons/{id}` runs one transaction: it clears the person as runner-up on everyone else's sightings, keeping the score, then deletes the person, and the foreign keys cascade to their enrolled photos, those photos' embeddings and their sightings, each sighting's JPEG crop included. `secure_delete` overwrites the deleted content with zeros, freelist pages included, and the rollback journal is truncated to zero length at commit (`journal_mode=TRUNCATE`, #43), so once the purge commits neither the database file nor its journal holds the faces. A test reads both files after a purge and finds neither the photo nor a crop that spans overflow pages (`tests/test_watchlist_removal.py`).

What purge cannot erase:

- Filesystem snapshots. An APFS snapshot is a read-only copy of a whole volume, and on a Mac with Time Machine on, Time Machine takes one of the startup disk about every hour and keeps it for 24 hours, besides the copies in its backups. A btrfs snapshot copies a subvolume in the same way. A snapshot taken while a person was enrolled keeps their faces until it is deleted.
- Blocks written elsewhere. A filesystem that writes changed data to new blocks, as btrfs does by default and APFS does for blocks a snapshot or clone shares, puts `secure_delete`'s zeros in new blocks, and the old ones keep the faces: held by the snapshot or clone that shares them, or freed without being overwritten.
- Blocks the journal freed. Before a transaction changes a page, SQLite copies the page's original content to the journal, so a purge's journal holds every page it zeroes. Truncating the journal frees its blocks without overwriting them, and what they held can stay on the disk until the filesystem reuses them. The same goes for the journals deleted at each commit before #43, when the journal mode was SQLite's default, `DELETE`.
- Uploads. The multipart parser spools an uploaded photo over 1 MB to a temporary file while the request is read; the file is deleted, not overwritten, when the request ends.

Neither moving the database nor running `VACUUM` after a purge would change this. On the Mac, the machine of record, local snapshots cover the whole startup disk wherever the database lives. SQLite offers `VACUUM` as the alternative to `secure_delete` for leaving no deleted content in the file, which `secure_delete` already does, and a `VACUUM` writes through the same filesystem. So these limits are disclosed, in the report's limitations, rather than worked around.
