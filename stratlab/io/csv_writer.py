"""CSV writer utilities for streaming trial outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class TrialCsvWriter:
    """CSV writer that discovers headers from the first row and streams append writes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._f = path.open("w", newline="")
        self._writer: csv.DictWriter[str] | None = None
        self._header: list[str] = []

    def write_row(self, row: dict[str, Any]) -> None:
        if self._writer is None:
            self._header = list(row.keys())
            self._writer = csv.DictWriter(self._f, fieldnames=self._header, extrasaction="ignore")
            self._writer.writeheader()
        self._writer.writerow({k: row.get(k) for k in self._header})
        if self._f.tell() % (1024 * 128) < 1024:
            self._f.flush()

    def close(self) -> None:
        self._f.flush()
        self._f.close()

    def flush(self) -> None:
        self._f.flush()
