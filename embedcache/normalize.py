"""Turning text into a cache key.

The temptation is to normalise aggressively — lowercase, strip punctuation, drop
stop words — so more lookups hit. That is a correctness bug wearing an
optimisation's clothes: ``"Hello"`` and ``"hello"`` are different strings to the
model, and returning one's vector for the other silently corrupts every
downstream result while raising the hit rate.

So normalisation here is limited to changes that cannot alter meaning:

* Unicode NFC, so the same grapheme is the same bytes;
* leading/trailing whitespace, which no tokenizer preserves;
* runs of whitespace collapsed to one space, which tokenizers treat identically.

Everything else — case, punctuation, diacritics — is part of the text and part
of the key.
"""

from __future__ import annotations

import hashlib
import unicodedata


def normalize(text: str) -> str:
    """Whitespace-and-encoding normalisation that cannot change meaning."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def key_for(text: str) -> str:
    """Stable content address for ``text``."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()
