"""A deterministic toy embedder.

Not a model. It exists so the cache can be demonstrated, benchmarked and tested
without a network call or an API key — and because a deterministic vector makes
"the hit returned exactly the bytes that were stored" an assertion rather than
a hope.

It also happens to be a decent lexical similarity stand-in: hashing word
trigrams into fixed buckets gives documents that share phrasing similar
vectors.
"""

from __future__ import annotations

import hashlib
from typing import Callable, List, Sequence


def hashing_embedder(dim: int = 64) -> Callable[[Sequence[str]], List[List[float]]]:
    if dim <= 0:
        raise ValueError("dim must be positive")

    def embed(texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            buckets = [0.0] * dim
            words = text.lower().split()
            grams = [" ".join(words[i:i + 3]) for i in range(max(len(words) - 2, 1))]
            for gram in grams:
                digest = hashlib.sha256(gram.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % dim
                buckets[index] += 1.0
            total = sum(buckets) or 1.0
            vectors.append([value / total for value in buckets])
        return vectors

    return embed
