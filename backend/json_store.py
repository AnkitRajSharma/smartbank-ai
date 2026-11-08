"""Small lock-protected JSON array store for the local prototype."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import portalocker


class JsonTable:
    """Persist a list of objects with atomic replacements and file locking."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def _read_unlocked(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as source:
                rows = json.load(source)
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Could not read JSON data file {self.path.name}") from error
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise RuntimeError(f"JSON data file {self.path.name} must contain an array of objects")
        return rows

    def _write_unlocked(self, rows: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent,
                                             prefix=f".{self.path.name}.", delete=False) as target:
                temporary_name = target.name
                json.dump(rows, target, indent=2, ensure_ascii=False)
                target.write("\n")
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary_name, self.path)
        except OSError as error:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
            raise RuntimeError(f"Could not write JSON data file {self.path.name}") from error

    def read(self) -> list[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a", timeout=5):
            return self._read_unlocked()

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a", timeout=5):
            rows = self._read_unlocked()
            rows.append(record)
            self._write_unlocked(rows)

    def replace(self, rows: list[dict[str, Any]]) -> None:
        """Atomically replace a table while holding its process-safe file lock."""
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("JSON tables must be arrays of objects")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a", timeout=5):
            self._write_unlocked(rows)

    def append_unique(self, record: dict[str, Any], key: str) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with portalocker.Lock(str(self.lock_path), mode="a", timeout=5):
            rows = self._read_unlocked()
            value = str(record.get(key, "")).lower()
            if any(str(row.get(key, "")).lower() == value for row in rows):
                return False
            rows.append(record)
            self._write_unlocked(rows)
            return True
