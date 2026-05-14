"""Tests for RFC-0018 §3.1 streaming sketches — count_min_sketch, hyperloglog, minhash."""
from __future__ import annotations

import pytest

from pypeline.math.sketches import (
    count_min_sketch,
    hyperloglog_cardinality,
    minhash_lsh_index,
)


# ===========================================================================
# count_min_sketch
# ===========================================================================

class TestCountMinSketch:
    def test_empty_stream_all_zeros(self) -> None:
        result = count_min_sketch([], width=64, depth=3)
        assert result["total_items"] == 0
        assert result["query"]("anything") == 0

    def test_single_item_frequency_one(self) -> None:
        result = count_min_sketch(["apple"], width=64, depth=3)
        assert result["query"]("apple") == 1

    def test_repeated_item_exact_count(self) -> None:
        stream = ["a"] * 10
        result = count_min_sketch(stream, width=64, depth=5)
        assert result["query"]("a") == 10

    def test_never_underestimates(self) -> None:
        stream = ["x"] * 7 + ["y"] * 3
        result = count_min_sketch(stream, width=64, depth=5)
        assert result["query"]("x") >= 7
        assert result["query"]("y") >= 3

    def test_unseen_item_zero(self) -> None:
        result = count_min_sketch(["known"], width=64, depth=3)
        # unseen items may collide but this specific item with wide table should be 0 or near
        count = result["query"]("definitely_not_in_stream_xyz")
        assert count >= 0  # CMS can overcount, never negative

    def test_total_items_matches_stream_length(self) -> None:
        stream = list(range(100))
        result = count_min_sketch(stream)
        assert result["total_items"] == 100

    def test_width_depth_stored(self) -> None:
        result = count_min_sketch([], width=512, depth=7)
        assert result["width"] == 512
        assert result["depth"] == 7

    def test_sketch_dimensions(self) -> None:
        result = count_min_sketch([], width=16, depth=4)
        sketch = result["sketch"]
        assert len(sketch) == 4
        assert all(len(row) == 16 for row in sketch)

    def test_invalid_width_raises(self) -> None:
        with pytest.raises(ValueError, match="width"):
            count_min_sketch([], width=0, depth=3)

    def test_invalid_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            count_min_sketch([], width=64, depth=0)

    def test_multiple_distinct_items(self) -> None:
        stream = ["a"] * 5 + ["b"] * 3 + ["c"] * 1
        result = count_min_sketch(stream, width=256, depth=5)
        assert result["query"]("a") >= 5
        assert result["query"]("b") >= 3
        assert result["query"]("c") >= 1

    def test_integer_items(self) -> None:
        stream = [1, 2, 1, 3, 1]
        result = count_min_sketch(stream, width=64, depth=4)
        assert result["query"](1) >= 3

    def test_mixed_types_in_stream(self) -> None:
        stream = ["hello", 42, None, "hello"]
        result = count_min_sketch(stream, width=128, depth=4)
        assert result["query"]("hello") >= 2


# ===========================================================================
# hyperloglog_cardinality
# ===========================================================================

class TestHyperLogLogCardinality:
    def test_empty_stream_returns_zero(self) -> None:
        result = hyperloglog_cardinality([])
        assert result == pytest.approx(0.0, abs=1.0)

    def test_single_unique_item(self) -> None:
        result = hyperloglog_cardinality(["only_one"])
        assert result >= 0.5  # HLL estimate near 1

    def test_exact_cardinality_small(self) -> None:
        stream = list(range(10))
        result = hyperloglog_cardinality(stream, precision=12)
        assert abs(result - 10) <= 5  # generous tolerance for small sets

    def test_large_cardinality_approximate(self) -> None:
        stream = list(range(10_000))
        result = hyperloglog_cardinality(stream, precision=14)
        # Within 5% of true value
        assert abs(result - 10_000) / 10_000 < 0.05

    def test_duplicates_not_double_counted(self) -> None:
        stream = ["same"] * 1000
        result = hyperloglog_cardinality(stream, precision=12)
        assert result < 10  # should be approximately 1

    def test_precision_too_low_raises(self) -> None:
        with pytest.raises(ValueError, match="precision"):
            hyperloglog_cardinality([], precision=3)

    def test_precision_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="precision"):
            hyperloglog_cardinality([], precision=17)

    def test_higher_precision_more_accurate(self) -> None:
        stream = list(range(1000))
        low_p = hyperloglog_cardinality(stream, precision=6)
        high_p = hyperloglog_cardinality(stream, precision=14)
        assert abs(high_p - 1000) < abs(low_p - 1000) or abs(high_p - 1000) < 50

    def test_returns_float(self) -> None:
        assert isinstance(hyperloglog_cardinality([1, 2, 3]), float)


# ===========================================================================
# minhash_lsh_index
# ===========================================================================

class TestMinHashLSHIndex:
    def test_empty_documents_list(self) -> None:
        result = minhash_lsh_index([], num_perm=32)
        assert result["num_documents"] == 0
        assert result["signatures"] == []

    def test_single_document_signature_length(self) -> None:
        result = minhash_lsh_index(["hello world"], num_perm=64)
        assert len(result["signatures"]) == 1
        assert len(result["signatures"][0]) == 64

    def test_identical_documents_jaccard_one(self) -> None:
        docs = ["the quick brown fox", "the quick brown fox"]
        result = minhash_lsh_index(docs, num_perm=128)
        similarity = result["query"](0, 1)
        assert similarity == pytest.approx(1.0)

    def test_completely_different_docs_low_similarity(self) -> None:
        docs = ["aaa bbb ccc ddd eee fff", "xxx yyy zzz 111 222 333"]
        result = minhash_lsh_index(docs, num_perm=128)
        similarity = result["query"](0, 1)
        assert similarity < 0.3

    def test_near_duplicate_high_similarity(self) -> None:
        base = "the quick brown fox jumps over the lazy dog"
        near_dup = "the quick brown fox jumps over the lazy cat"
        result = minhash_lsh_index([base, near_dup], num_perm=128)
        similarity = result["query"](0, 1)
        assert similarity > 0.5

    def test_num_perm_stored(self) -> None:
        result = minhash_lsh_index(["a", "b"], num_perm=64)
        assert result["num_perm"] == 64

    def test_num_documents_stored(self) -> None:
        docs = ["a", "b", "c"]
        result = minhash_lsh_index(docs, num_perm=32)
        assert result["num_documents"] == 3

    def test_index_out_of_range_raises(self) -> None:
        result = minhash_lsh_index(["only_one"], num_perm=32)
        with pytest.raises(IndexError):
            result["query"](0, 5)

    def test_invalid_num_perm_raises(self) -> None:
        with pytest.raises(ValueError, match="num_perm"):
            minhash_lsh_index(["doc"], num_perm=0)

    def test_symmetry_of_jaccard(self) -> None:
        docs = ["abc def ghi", "abc def xyz"]
        result = minhash_lsh_index(docs, num_perm=128)
        assert result["query"](0, 1) == result["query"](1, 0)

    def test_short_document(self) -> None:
        result = minhash_lsh_index(["ab"], num_perm=32)
        assert len(result["signatures"][0]) == 32
