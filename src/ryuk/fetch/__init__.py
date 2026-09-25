"""Fetching the benchmark datasets and model weights from pinned, checksum-verified sources."""


class FetchError(Exception):
    """A pinned source could not be fetched or did not match its pin."""
