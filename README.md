# embed-cache

**Re-index a corpus without paying to embed it twice.**

[![CI](https://github.com/aiplusplus360/embed-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/aiplusplus360/embed-cache/actions/workflows/ci.yml)

```
first index  -> embedded 200 texts
re-index     -> embedded 200 texts total, hit rate 0.50
after 5 edits-> embedded 205 texts total, hit rate 0.66
```

Two hundred chunks indexed, then re-indexed for **zero** additional embedding
calls. Then a document was edited — five chunks changed — and the re-index cost
**five** calls, not two hundred.

Zero dependencies. SQLite is in the standard library.

---

## The problem

A re-index is usually triggered by one thing changing: a document, a chunking
strategy, an embedding batch size. The naive pipeline re-embeds the entire
corpus to pick up that one change, and it does it at full price.

The obvious fix — cache on the chunk id — does not survive re-chunking, because
chunk ids are positional. Cache on a *content hash* and the identity follows the
text: an unchanged chunk is the same chunk, no matter where it moved to.

---

## Install

```bash
pip install embed-cache        # or: pip install -e .
```

## Use it

```python
from embedcache import EmbeddingCache, SqliteStore

with SqliteStore("embeddings.db") as store:
    cache = EmbeddingCache(store, model="text-embedding-3-small", embedder=my_embedder)

    vectors = cache.embed(chunks)          # one call for the misses only
    print(cache.stats.as_dict())
    # {'texts': 200, 'hits': 0, 'misses': 200, 'embedded': 200, ...}
```

`my_embedder` is any callable taking a list of strings and returning one vector
per string, in order. The cache owns the batching, deduplication and persistence.

### CLI

```bash
embedcache --db cache.db embed --model demo "some text"
embedcache --db cache.db stats
embedcache --db cache.db purge --model demo
```

Ships with a deterministic hashing embedder so the whole thing runs offline with
no API key — useful for tests, benchmarks, and demos.

---

## The four things that make a cache safe

A fast cache that returns wrong vectors is worse than no cache, because nothing
looks broken. These are enforced in code, not documented as advice.

### 1. A vector never crosses models

The primary key is `(model, content hash)`. The same text embedded by two models
is two rows, and a lookup can only ever see its own model's row. Mixing two
models' vectors still produces plausible-looking distances — which is exactly
why it is the failure nobody catches.

### 2. Normalisation cannot change meaning

Only these are normalised, because a tokenizer treats them as identical:

* Unicode NFC (`café` composed == `café` decomposed);
* leading/trailing whitespace;
* runs of whitespace collapsed to one space.

**Case is not normalised.** `"Hello"` and `"hello"` are different strings to the
model; merging them raises the hit rate and corrupts every result downstream.
The same goes for punctuation and diacritics. A test pins the hash of a known
string, so a future "optimisation" that starts lowercasing fails the build.

### 3. A hit returns the exact bytes that were stored

Vectors are stored as raw `float64` blobs, not JSON text. They round-trip
bit-identically, so `cache.embed(["x"]) == cache.embed(["x"])` is an equality
assertion rather than a tolerance check — and a cache bug shows up as a failing
test rather than as slightly worse retrieval six weeks later.

### 4. A batch is all-or-nothing

The embedder is called once per unique miss set, and results are written in a
single transaction *after* the count and dimensions have been checked. If the
provider returns 19 vectors for 20 texts, or one vector of the wrong width, the
cache raises and stores nothing. A half-written batch is a cache that disagrees
with itself.

---

## What counts as a hit

"Hit" means **served without calling the embedder**, and that includes a repeat
within the same batch:

```python
cache.embed(["same", "same", "same"])   # 1 embedder call, 2 hits
```

Counting those as misses would overstate the spend the cache prevented, which is
the number people actually look at. `saved_chars` follows the same rule.

---

## Design notes

- **SQLite, not a JSON file.** Atomic writes and concurrent readers for free.
  Two workers re-indexing at once do not lose each other's work; a crash
  mid-write does not truncate the cache. WAL mode is enabled on open.
- **`INSERT OR IGNORE`, not upsert.** A content key is immutable — re-writing it
  would only rewrite identical bytes. If a vector *changed* for the same text,
  the model changed, and that belongs in a new `model` value.
- **Text is validated before any work is done.** Empty-after-normalisation is a
  `ValueError` naming the index, not a stored zero vector. A zero vector is a
  smaller lie than a wrong one, but it is still a lie.
- **`embed("abc")` is a `TypeError`.** Otherwise it iterates the string and
  embeds three one-letter texts, which looks like it worked.

### What this deliberately does not do

- **No HTTP client.** Bring your own embedder; this does not pick a provider.
- **No vector search.** This is the cache in front of the index, not the index.
  Pair it with your existing vector store.
- **No eviction.** Content-addressed entries stay valid until the model changes.
  `purge --model` is the invalidation mechanism, and it is explicit.

---

## Development

```bash
python -m unittest discover -s tests -v
```

48 tests, no third-party runtime dependencies.

```
embedcache/
  normalize.py   conservative text normalisation + content keys
  store.py       SQLite persistence, float64 blobs, transactions
  cache.py       batching, dedupe, stats, invariants
  demo.py        deterministic hashing embedder (offline)
  cli.py         embed / stats / purge
```

## License

MIT
