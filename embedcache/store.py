"""SQLite-backed vector store.

SQLite is in the standard library and gives the two things a cache actually
needs: atomic writes and concurrent readers. A JSON file gives neither — two
workers re-indexing at once lose each other's work, and a crash mid-write
truncates the file.

Vectors are stored as raw ``float64`` blobs, not JSON text. A 1536-dimension
embedding is 12 KB of JSON and 12 KB of float32 — but only the blob round-trips
*bit-identically*, which is what makes "cache hit returns the same vector"
checkable rather than hopeful.
"""

from __future__ import annotations

import sqlite3
import time
from array import array
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    model      TEXT    NOT NULL,
    key        TEXT    NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB    NOT NULL,
    chars      INTEGER NOT NULL,
    created_at REAL    NOT NULL,
    PRIMARY KEY (model, key)
);
CREATE INDEX IF NOT EXISTS idx_model_created ON embeddings (model, created_at);
"""


class StoreError(Exception):
    """The store exists but cannot be used."""


class SqliteStore:
    """A persistent map from (model, content key) to a vector."""

    def __init__(self, path: str = ":memory:", *, timeout: float = 10.0) -> None:
        self.path = path
        if path != ":memory:":
            parent = Path(path).expanduser().resolve().parent
            if not parent.exists():
                raise StoreError(f"directory does not exist: {parent}")
        try:
            self._db = sqlite3.connect(str(path), timeout=timeout)
            # WAL lets one writer and many readers coexist; without it a second
            # indexing process fails on the first insert.
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.executescript(SCHEMA)
            self._db.commit()
        except sqlite3.DatabaseError as exc:
            raise StoreError(f"cannot open {path}: {exc}") from None

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> "SqliteStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- reads --------------------------------------------------------------

    def get_many(self, model: str, keys: Sequence[str]) -> Dict[str, Tuple[int, List[float]]]:
        if not keys:
            return {}
        placeholders = ",".join("?" * len(keys))
        rows = self._db.execute(
            f"SELECT key, dim, vector FROM embeddings "
            f"WHERE model = ? AND key IN ({placeholders})",
            (model, *keys),
        ).fetchall()
        found: Dict[str, Tuple[int, List[float]]] = {}
        for key, dim, blob in rows:
            vector = array("d")
            vector.frombytes(blob)
            found[key] = (dim, vector.tolist())
        return found

    def count(self, model: Optional[str] = None) -> int:
        if model is None:
            row = self._db.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        else:
            row = self._db.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model = ?", (model,)
            ).fetchone()
        return int(row[0])

    def characters(self, model: Optional[str] = None) -> int:
        if model is None:
            row = self._db.execute("SELECT COALESCE(SUM(chars), 0) FROM embeddings").fetchone()
        else:
            row = self._db.execute(
                "SELECT COALESCE(SUM(chars), 0) FROM embeddings WHERE model = ?",
                (model,),
            ).fetchone()
        return int(row[0])

    def models(self) -> List[str]:
        rows = self._db.execute(
            "SELECT DISTINCT model FROM embeddings ORDER BY model"
        ).fetchall()
        return [row[0] for row in rows]

    # -- writes -------------------------------------------------------------

    def put_many(
        self,
        model: str,
        entries: Iterable[Tuple[str, Sequence[float], int]],
    ) -> int:
        """Insert ``(key, vector, chars)`` triples in one transaction.

        Existing rows are left alone: a content key is immutable, so re-writing
        it would only rewrite identical bytes.
        """
        rows = [
            (model, key, len(vector), array("d", vector).tobytes(), chars, time.time())
            for key, vector, chars in entries
        ]
        if not rows:
            return 0
        with self._db:
            self._db.executemany(
                "INSERT OR IGNORE INTO embeddings "
                "(model, key, dim, vector, chars, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def purge(self, model: Optional[str] = None) -> int:
        with self._db:
            if model is None:
                cursor = self._db.execute("DELETE FROM embeddings")
            else:
                cursor = self._db.execute(
                    "DELETE FROM embeddings WHERE model = ?", (model,)
                )
        return cursor.rowcount
