"""A content-addressed embedding cache.

The cost model this exists for: re-indexing a corpus re-embeds almost every
chunk, for the same vectors, at full price. Content addressing turns that into
a lookup — and because a hash of the text *is* the identity, an edited document
only re-embeds the chunks that actually changed.

Two invariants, both enforced rather than assumed:

**A vector never crosses models.** The key is (model, content hash), so the same
text embedded by two models is two rows and can never be confused. A dimension
mismatch inside one model is a hard error, not a silent truncation.

**A hit returns the bytes that were stored.** Vectors round-trip through a
``float64`` blob, so a cached result is bit-identical to the original — a test
can assert equality instead of closeness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from .normalize import key_for, normalize
from .store import SqliteStore

# An embedder takes unique texts and returns one vector per text, in order.
Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]

# Used only to report roughly how many prompt tokens were avoided.
_CHARS_PER_TOKEN = 4.0


class EmbeddingError(Exception):
    """The embedder returned something the cache cannot trust."""


@dataclass
class Stats:
    requests: int = 0
    texts: int = 0
    hits: int = 0
    misses: int = 0
    embedded: int = 0
    saved_chars: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.texts if self.texts else 0.0

    @property
    def saved_tokens_estimate(self) -> int:
        return int(self.saved_chars / _CHARS_PER_TOKEN)

    def as_dict(self) -> Dict[str, float]:
        return {
            "requests": self.requests,
            "texts": self.texts,
            "hits": self.hits,
            "misses": self.misses,
            "embedded": self.embedded,
            "hit_rate": round(self.hit_rate, 4),
            "saved_chars": self.saved_chars,
            "saved_tokens_estimate": self.saved_tokens_estimate,
        }


class EmbeddingCache:
    """Embed texts through a store, calling the embedder only for misses."""

    def __init__(
        self,
        store: SqliteStore,
        model: str,
        embedder: Embedder,
        *,
        dim: Optional[int] = None,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        self.store = store
        self.model = model
        self.embedder = embedder
        self.dim = dim
        self.stats = Stats()

    # -- public API ---------------------------------------------------------

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed ``texts``, reusing anything already cached.

        Every text is checked before the embedder is called, so a batch of 500
        chunks with 480 already indexed costs one call for 20 texts.
        """
        if isinstance(texts, str):
            raise TypeError("embed() takes a list of texts, not a single string")

        cleaned = self._validate(texts)
        self.stats.requests += 1
        self.stats.texts += len(cleaned)
        if not cleaned:
            return []

        keys = [key_for(text) for text in cleaned]
        cached = self.store.get_many(self.model, list(dict.fromkeys(keys)))

        resolved: Dict[str, List[float]] = {}
        pending: List[str] = []
        queued = set()

        # "Hit" means "served without calling the embedder". That includes a
        # repeat inside the same batch: the second copy costs nothing, and
        # counting it as a miss would overstate the spend the cache prevented.
        hits = 0
        saved_chars = 0
        for text, key in zip(cleaned, keys):
            if key in cached:
                if text not in resolved:
                    dim, vector = cached[key]
                    self._check_dim(dim, key)
                    resolved[text] = vector
                hits += 1
                saved_chars += len(normalize(text))
            elif key in queued:
                hits += 1
                saved_chars += len(normalize(text))
            else:
                # A batch often repeats itself; embed each distinct text once.
                queued.add(key)
                pending.append(text)

        self.stats.hits += hits
        self.stats.saved_chars += saved_chars
        self.stats.misses += len(pending)

        if pending:
            for text, vector in self._embed_uncached(pending).items():
                resolved[text] = vector

        return [list(resolved[text]) for text in cleaned]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "EmbeddingCache":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _validate(texts: Sequence[str]) -> List[str]:
        cleaned: List[str] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"texts[{index}] is {type(text).__name__}, not str")
            if not normalize(text):
                # An empty embedding is not a smaller embedding; it is a bug in
                # the caller that would otherwise poison the index with a
                # zero-information vector.
                raise ValueError(f"texts[{index}] is empty after normalisation")
            cleaned.append(text)
        return cleaned

    def _check_dim(self, found: int, key: str) -> None:
        if self.dim is None:
            self.dim = found
        elif found != self.dim:
            raise EmbeddingError(
                f"model {self.model!r} returned {found} dimensions for {key[:12]}… "
                f"but earlier vectors had {self.dim}; refusing to mix them"
            )

    def _embed_uncached(self, texts: List[str]) -> Dict[str, List[float]]:
        vectors = self.embedder(texts)
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"embedder returned {len(vectors)} vectors for {len(texts)} texts"
            )

        results: Dict[str, List[float]] = {}
        rows = []
        for text, vector in zip(texts, vectors):
            values = [float(value) for value in vector]
            if not values:
                raise EmbeddingError(f"embedder returned an empty vector for {text[:40]!r}")
            self._check_dim(len(values), key_for(text))
            results[text] = values
            rows.append((key_for(text), values, len(normalize(text))))

        # One transaction: a crash or a bad batch leaves the cache unchanged,
        # never half-written with a dimension nobody checked.
        self.store.put_many(self.model, rows)
        self.stats.embedded += len(rows)
        return results
