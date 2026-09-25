"""The benchmark datasets as `ryuk data fetch` leaves them on disk: LFW's folders and pairs lists,
CelebA's label table and image shards. Read-only; nothing here downloads.
"""


class DatasetError(Exception):
    """A dataset file on disk is not in the form the fetch left it in."""
