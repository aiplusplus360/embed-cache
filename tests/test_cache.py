"""Tests for normalisation, storage, and the cache.

The dangerous failures in an embedding cache are not "slow". They are:

* returning a vector from the *wrong model*, which nobody notices until quality
  quietly degrades;
* returning the vector of *different text*, because normalisation got creative;
* writing half a batch, so the cache disagrees with itself;
* reporting a hit rate that is really a pile of silent mismatches.

Every test below is about one of those.
"""

import json
import tempfile
import unittest
from pathlib import Path

from embedcache import (
    EmbeddingCache,
    EmbeddingError,
    SqliteStore,
    StoreError,
    hashing_embedder,
    key_for,
    normalize,
)
from embedcache.cli import main


class CountingEmbedder:
    """Records what it was asked to embed, so the cache's claims are checkable."""

    def __init__(self, dim=8):
        self.dim = dim
        self.calls = []
        self._inner = hashing_embedder(dim)

    def __call__(self, texts):
        self.calls.append(list(texts))
        return self._inner(texts)

    @property
    def embedded_texts(self):
        return [text for call in self.calls for text in call]


def make_cache(tmp, embedder=None, model="test-model", dim=None):
    store = SqliteStore(str(Path(tmp) / "cache.db"))
    return EmbeddingCache(store, model, embedder or CountingEmbedder(), dim=dim)


class TestNormalize(unittest.TestCase):
    def test_whitespace_does_not_change_the_key(self):
        self.assertEqual(key_for("hello  world"), key_for("hello world"))

    def test_leading_and_trailing_space_does_not_change_the_key(self):
        self.assertEqual(key_for("  hello\n"), key_for("hello"))

    def test_case_is_part_of_the_identity(self):
        # "Hello" and "hello" are different strings to the model. Merging them
        # would raise the hit rate and corrupt the results.
        self.assertNotEqual(key_for("Hello"), key_for("hello"))

    def test_punctuation_is_part_of_the_identity(self):
        self.assertNotEqual(key_for("stop."), key_for("stop"))

    def test_unicode_is_the_same_in_nfc_and_nfd(self):
        composed = "caf\u00e9"
        decomposed = "cafe\u0301"
        self.assertEqual(key_for(composed), key_for(decomposed))

    def test_normalize_collapses_runs(self):
        self.assertEqual(normalize(" a \t\n b "), "a b")

    def test_key_is_sha256_of_the_normalised_text(self):
        # Pinned to a literal: if normalisation ever starts stripping case or
        # punctuation, this fails instead of silently merging distinct content.
        self.assertEqual(
            key_for("stable"),
            "f379ccb92b9116442dc65bdc35648a85d3786b34779db7f704a901fa07b00cb6",
        )


class TestStore(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self._tmp.name) / "store.db")

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip_returns_identical_floats(self):
        with SqliteStore(self.path) as store:
            vector = [0.1, -2.5, 1e-17, 3.14159265358979]
            store.put_many("m", [("k", vector, 10)])
            found = store.get_many("m", ["k"])
        self.assertEqual(found["k"][1], vector)

    def test_missing_key_is_absent(self):
        with SqliteStore(self.path) as store:
            self.assertEqual(store.get_many("m", ["nope"]), {})

    def test_models_are_isolated(self):
        with SqliteStore(self.path) as store:
            store.put_many("a", [("k", [1.0], 1)])
            self.assertEqual(store.get_many("b", ["k"]), {})
            self.assertEqual(store.count("a"), 1)
            self.assertEqual(store.count("b"), 0)

    def test_put_is_idempotent(self):
        with SqliteStore(self.path) as store:
            store.put_many("m", [("k", [1.0], 1)])
            store.put_many("m", [("k", [9.0], 1)])
            self.assertEqual(store.count("m"), 1)
            self.assertEqual(store.get_many("m", ["k"])["k"][1], [1.0])

    def test_purge_one_model_leaves_the_other(self):
        with SqliteStore(self.path) as store:
            store.put_many("a", [("k", [1.0], 1)])
            store.put_many("b", [("k", [1.0], 1)])
            removed = store.purge("a")
            self.assertEqual(removed, 1)
            self.assertEqual(store.count("a"), 0)
            self.assertEqual(store.count("b"), 1)

    def test_purge_everything(self):
        with SqliteStore(self.path) as store:
            store.put_many("a", [("k", [1.0], 1)])
            store.put_many("b", [("k", [1.0], 1)])
            self.assertEqual(store.purge(), 2)
            self.assertEqual(store.count(), 0)

    def test_character_total(self):
        with SqliteStore(self.path) as store:
            store.put_many("m", [("a", [1.0], 5), ("b", [1.0], 7)])
            self.assertEqual(store.characters("m"), 12)

    def test_models_listed(self):
        with SqliteStore(self.path) as store:
            store.put_many("b", [("k", [1.0], 1)])
            store.put_many("a", [("k", [1.0], 1)])
            self.assertEqual(store.models(), ["a", "b"])

    def test_missing_directory_is_reported(self):
        with self.assertRaises(StoreError):
            SqliteStore(str(Path(self._tmp.name) / "nope" / "cache.db"))

    def test_data_survives_reopening(self):
        with SqliteStore(self.path) as store:
            store.put_many("m", [("k", [1.5], 1)])
        with SqliteStore(self.path) as store:
            self.assertEqual(store.get_many("m", ["k"])["k"][1], [1.5])


class TestCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_second_call_does_not_embed_again(self):
        embedder = CountingEmbedder()
        with make_cache(self._tmp.name, embedder) as cache:
            first = cache.embed(["alpha"])
            second = cache.embed(["alpha"])
        self.assertEqual(embedder.embedded_texts, ["alpha"])
        self.assertEqual(first, second)

    def test_hit_is_bit_identical_to_the_original(self):
        embedder = CountingEmbedder()
        with make_cache(self._tmp.name, embedder) as cache:
            first = cache.embed(["alpha"])[0]
            second = cache.embed(["alpha"])[0]
        self.assertEqual(first, second)

    def test_duplicates_within_one_batch_are_embedded_once(self):
        embedder = CountingEmbedder()
        with make_cache(self._tmp.name, embedder) as cache:
            vectors = cache.embed(["same", "same", "same"])
        self.assertEqual(embedder.embedded_texts, ["same"])
        self.assertEqual(vectors[0], vectors[1])
        self.assertEqual(vectors[1], vectors[2])

    def test_whitespace_variant_is_a_cache_hit(self):
        embedder = CountingEmbedder()
        with make_cache(self._tmp.name, embedder) as cache:
            cache.embed(["hello world"])
            cache.embed(["  hello   world "])
        self.assertEqual(embedder.embedded_texts, ["hello world"])

    def test_different_text_is_not_a_hit(self):
        embedder = CountingEmbedder()
        with make_cache(self._tmp.name, embedder) as cache:
            cache.embed(["hello"])
            cache.embed(["hello!"])
        self.assertEqual(sorted(embedder.embedded_texts), ["hello", "hello!"])

    def test_order_is_preserved(self):
        with make_cache(self._tmp.name) as cache:
            texts = ["one", "two", "three"]
            vectors = cache.embed(texts)
            # Re-request in a different order; each must follow its own text.
            reversed_vectors = cache.embed(texts[::-1])
        self.assertEqual(vectors, reversed_vectors[::-1])

    def test_model_isolation(self):
        embedder = CountingEmbedder()
        with make_cache(self._tmp.name, embedder, model="model-a") as cache_a:
            a = cache_a.embed(["text"])[0]
        with make_cache(self._tmp.name, embedder, model="model-b") as cache_b:
            b = cache_b.embed(["text"])[0]
        # Different models must not share a row even for identical text.
        self.assertEqual(len(embedder.embedded_texts), 2)
        self.assertEqual(a, b)  # the same toy embedder, different cache rows
        with SqliteStore(str(Path(self._tmp.name) / "cache.db")) as store:
            self.assertEqual(store.count(), 2)

    def test_dimension_conflict_is_loud(self):
        vectors = iter([[1.0, 2.0], [1.0, 2.0, 3.0]])

        def bad_embedder(texts):
            return [next(vectors) for _ in texts]

        with self.assertRaises(EmbeddingError) as caught:
            with make_cache(self._tmp.name, bad_embedder) as cache:
                cache.embed(["a", "b"])
        self.assertIn("dimension", str(caught.exception))

    def test_wrong_vector_count_is_loud(self):
        def short_embedder(texts):
            return [[1.0]] * (len(texts) - 1)

        with self.assertRaises(EmbeddingError):
            with make_cache(self._tmp.name, short_embedder) as cache:
                cache.embed(["a", "b"])

    def test_empty_vector_is_rejected(self):
        def empty_embedder(texts):
            return [[] for _ in texts]

        with self.assertRaises(EmbeddingError):
            with make_cache(self._tmp.name, empty_embedder) as cache:
                cache.embed(["a"])

    def test_failed_batch_writes_nothing(self):
        def exploding_embedder(texts):
            raise RuntimeError("provider is down")

        with self.assertRaises(RuntimeError):
            with make_cache(self._tmp.name, exploding_embedder) as cache:
                cache.embed(["a"])
        with SqliteStore(str(Path(self._tmp.name) / "cache.db")) as store:
            self.assertEqual(store.count(), 0)

    def test_empty_text_is_rejected_not_stored(self):
        with make_cache(self._tmp.name) as cache:
            with self.assertRaises(ValueError):
                cache.embed(["   "])
        with SqliteStore(str(Path(self._tmp.name) / "cache.db")) as store:
            self.assertEqual(store.count(), 0)

    def test_non_string_is_rejected(self):
        with make_cache(self._tmp.name) as cache:
            with self.assertRaises(TypeError):
                cache.embed([None])

    def test_a_bare_string_is_a_type_error(self):
        # `cache.embed("abc")` would otherwise iterate characters and embed 3
        # one-letter texts.
        with make_cache(self._tmp.name) as cache:
            with self.assertRaises(TypeError):
                cache.embed("abc")

    def test_empty_batch(self):
        embedder = CountingEmbedder()
        with make_cache(self._tmp.name, embedder) as cache:
            self.assertEqual(cache.embed([]), [])
        self.assertEqual(embedder.calls, [])

    def test_returned_lists_are_independent(self):
        with make_cache(self._tmp.name) as cache:
            first = cache.embed(["x"])[0]
            first[0] = 999.0
            second = cache.embed(["x"])[0]
        self.assertNotEqual(second[0], 999.0)

    def test_persists_across_instances(self):
        embedder = CountingEmbedder()
        with make_cache(self._tmp.name, embedder) as cache:
            cache.embed(["persisted"])
        with make_cache(self._tmp.name, embedder) as cache:
            cache.embed(["persisted"])
        self.assertEqual(embedder.embedded_texts, ["persisted"])

    def test_embed_one(self):
        with make_cache(self._tmp.name) as cache:
            self.assertEqual(len(cache.embed_one("one")), 8)


class TestStats(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def test_counts_hits_and_misses(self):
        with make_cache(self._tmp.name) as cache:
            cache.embed(["a", "b"])
            cache.embed(["a", "c"])
            stats = cache.stats
        self.assertEqual(stats.texts, 4)
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.misses, 3)
        self.assertEqual(stats.embedded, 3)

    def test_duplicate_hits_count_per_occurrence(self):
        with make_cache(self._tmp.name) as cache:
            cache.embed(["x", "x", "x"])
            stats = cache.stats
        self.assertEqual(stats.hits, 2)
        self.assertEqual(stats.misses, 1)

    def test_saved_characters_reflect_avoided_work_only(self):
        with make_cache(self._tmp.name) as cache:
            cache.embed(["abcdefgh"])
            cache.embed(["abcdefgh"])
            stats = cache.stats
        self.assertEqual(stats.saved_chars, 8)

    def test_hit_rate(self):
        with make_cache(self._tmp.name) as cache:
            cache.embed(["a", "b"])
            cache.embed(["a", "b"])
            stats = cache.stats
        self.assertAlmostEqual(stats.hit_rate, 0.5)

    def test_hit_rate_of_empty_cache_is_zero(self):
        with make_cache(self._tmp.name) as cache:
            self.assertEqual(cache.stats.hit_rate, 0.0)

    def test_saved_tokens_is_estimated_from_characters(self):
        with make_cache(self._tmp.name) as cache:
            cache.embed(["a" * 400])
            cache.embed(["a" * 400])
            stats = cache.stats
        self.assertEqual(stats.saved_tokens_estimate, 100)


class TestCli(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self._tmp.name) / "cli.db")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, argv):
        from contextlib import redirect_stderr, redirect_stdout
        import io

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_embed_then_stats(self):
        code, out, _ = self._run(["--db", self.db, "embed", "--model", "m", "hello"])
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(out)["vectors"]), 1)

        code, out, _ = self._run(["--db", self.db, "stats"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out)["entries"], 1)

    def test_second_embed_is_a_hit(self):
        self._run(["--db", self.db, "embed", "--model", "m", "hello"])
        _, out, _ = self._run(["--db", self.db, "embed", "--model", "m", "hello"])
        self.assertEqual(json.loads(out)["stats"]["hits"], 1)

    def test_stats_restricted_to_one_model(self):
        self._run(["--db", self.db, "embed", "--model", "a", "x"])
        self._run(["--db", self.db, "embed", "--model", "b", "y"])
        _, out, _ = self._run(["--db", self.db, "stats", "--model", "a"])
        self.assertEqual(json.loads(out)["entries"], 1)
        self.assertEqual(json.loads(out)["models"], ["a", "b"])

    def test_purge_one_model(self):
        self._run(["--db", self.db, "embed", "--model", "a", "x"])
        self._run(["--db", self.db, "embed", "--model", "b", "y"])
        _, out, _ = self._run(["--db", self.db, "purge", "--model", "a"])
        self.assertEqual(json.loads(out)["removed"], 1)
        _, out, _ = self._run(["--db", self.db, "stats"])
        self.assertEqual(json.loads(out)["entries"], 1)

    def test_purge_without_model_removes_everything(self):
        self._run(["--db", self.db, "embed", "--model", "a", "x"])
        self._run(["--db", self.db, "embed", "--model", "b", "y"])
        _, out, _ = self._run(["--db", self.db, "purge"])
        self.assertEqual(json.loads(out)["removed"], 2)

    def test_bad_db_path_exits_2(self):
        code, _, err = self._run(["--db", str(Path(self._tmp.name) / "nope" / "x.db"), "stats"])
        self.assertEqual(code, 2)
        self.assertIn("directory does not exist", err)

    def test_empty_text_exits_2(self):
        code, _, err = self._run(["--db", self.db, "embed", "   "])
        self.assertEqual(code, 2)
        self.assertIn("empty", err)


if __name__ == "__main__":
    unittest.main()
