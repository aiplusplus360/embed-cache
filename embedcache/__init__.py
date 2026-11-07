"""Cache embeddings, keyed by model and content hash."""

from .cache import EmbeddingCache, EmbeddingError, Embedder, Stats
from .demo import hashing_embedder
from .normalize import key_for, normalize
from .store import SqliteStore, StoreError

__version__ = "0.1.0"

__all__ = [
    "EmbeddingCache",
    "EmbeddingError",
    "Embedder",
    "Stats",
    "SqliteStore",
    "StoreError",
    "hashing_embedder",
    "key_for",
    "normalize",
    "__version__",
]
