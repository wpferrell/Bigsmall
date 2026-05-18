"""BigSmall public exception types.

`BigSmallVersionError` is raised when a .bs file (or codec inside it) was
produced by a newer release than the one installed. Catch this at the
application boundary and surface the embedded `pip install --upgrade
bigsmall` instruction; do NOT swallow it silently.
"""
from __future__ import annotations


class BigSmallVersionError(Exception):
    """Raised when a .bs container or codec requires a newer bigsmall.

    The message always contains an actionable upgrade instruction so users
    don't have to look up what to do next.
    """

    def __init__(self, required: str, installed: str, *, detail: str | None = None):
        self.required = required
        self.installed = installed
        msg = (
            f"This file requires bigsmall >= {required} (you have {installed}).\n"
            "Run: pip install --upgrade bigsmall"
        )
        if detail:
            msg += f"\n({detail})"
        super().__init__(msg)
