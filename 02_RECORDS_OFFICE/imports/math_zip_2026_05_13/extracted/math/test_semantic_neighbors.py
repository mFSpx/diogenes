"""Tests for pypeline.math.semantic_neighbors — RFC-0018 §3.4 WO-BM-010."""
from __future__ import annotations

import pytest
from pypeline.math.semantic_neighbors import (
    NeighborResult,
    register_document,
    clear_enclave,
    semantic_neighbors,
)


ENCLAVE = "test_enc_001"


@pytest.fixture(autouse=True)
def clean_enclave():
    clear_enclave(ENCLAVE)
    yield
    clear_enclave(ENCLAVE)


def _register(n: int = 5) -> None:
    docs = [
        ("doc_bcfsa", "BCFSA regulatory complaint financial fraud mortgage broker"),
        ("doc_civil", "Civil claim plaintiff defendant breach of contract damages"),
        ("doc_tax", "CRA tax evasion income reporting offshore funds transfer"),
        ("doc_land", "Land title transfer ownership chain beneficial owner nominee"),
        ("doc_fintrac", "FINTRAC anti-money laundering suspicious transaction report"),
    ]
    for i, (doc_id, text) in enumerate(docs[:n]):
        register_document(ENCLAVE, doc_id, text)


class TestRegisterAndRetrieve:
    def test_empty_enclave_returns_empty(self):
        results = semantic_neighbors("fraud", ENCLAVE, k=5, backend="memory")
        assert results == []

    def test_returns_list_of_neighbor_results(self):
        _register()
        results = semantic_neighbors("fraud", ENCLAVE, k=3, backend="memory")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, NeighborResult)

    def test_respects_k_limit(self):
        _register(5)
        results = semantic_neighbors("fraud", ENCLAVE, k=2, backend="memory")
        assert len(results) <= 2

    def test_distance_in_unit_interval(self):
        _register()
        results = semantic_neighbors("fraud", ENCLAVE, k=5, backend="memory")
        for r in results:
            assert 0.0 <= r.distance <= 1.0

    def test_relevant_doc_ranks_higher(self):
        _register()
        results = semantic_neighbors("BCFSA complaint financial fraud", ENCLAVE, k=5, backend="memory")
        assert len(results) > 0
        # doc_bcfsa should be near the top since it has highest overlap
        top_ids = [r.neighbor_id for r in results[:2]]
        assert "doc_bcfsa" in top_ids

    def test_retrieval_receipt_id_emitted(self):
        _register()
        results = semantic_neighbors("fraud", ENCLAVE, k=3, backend="memory")
        for r in results:
            assert r.retrieval_receipt_id.startswith("receipt_")

    def test_all_results_share_receipt_id_per_call(self):
        _register()
        results = semantic_neighbors("fraud", ENCLAVE, k=5, backend="memory")
        receipts = {r.retrieval_receipt_id for r in results}
        assert len(receipts) == 1  # one receipt per call

    def test_two_calls_different_receipt_ids(self):
        _register()
        r1 = semantic_neighbors("fraud", ENCLAVE, k=3, backend="memory")
        r2 = semantic_neighbors("fraud", ENCLAVE, k=3, backend="memory")
        if r1 and r2:
            assert r1[0].retrieval_receipt_id != r2[0].retrieval_receipt_id

    def test_truth_band_filter_applied(self):
        register_document(ENCLAVE, "obs_doc", "fraud evidence observed", truth_band="OBSERVED")
        register_document(ENCLAVE, "inf_doc", "fraud inferred pattern", truth_band="INFERRED")
        results = semantic_neighbors("fraud", ENCLAVE, k=10, backend="memory",
                                     truth_band_filter=["OBSERVED"])
        ids = [r.neighbor_id for r in results]
        assert "obs_doc" in ids
        assert "inf_doc" not in ids

    def test_neighbor_id_matches_registered_doc_id(self):
        register_document(ENCLAVE, "unique_doc_xyz", "specific unique text")
        results = semantic_neighbors("specific unique text", ENCLAVE, k=5, backend="memory")
        assert any(r.neighbor_id == "unique_doc_xyz" for r in results)

    def test_k_larger_than_corpus_returns_all_docs(self):
        _register(3)
        results = semantic_neighbors("fraud", ENCLAVE, k=100, backend="memory")
        assert len(results) == 3


class TestBackendErrors:
    def test_chroma_backend_raises_import_or_not_implemented(self):
        with pytest.raises((ImportError, NotImplementedError)):
            semantic_neighbors("test", ENCLAVE, backend="chroma")

    def test_faiss_backend_raises_import_or_not_implemented(self):
        with pytest.raises((ImportError, NotImplementedError)):
            semantic_neighbors("test", ENCLAVE, backend="faiss")

    def test_qdrant_backend_raises_import_error(self):
        with pytest.raises(ImportError):
            semantic_neighbors("test", ENCLAVE, backend="qdrant")

    def test_pgvector_backend_raises_import_error(self):
        with pytest.raises(ImportError):
            semantic_neighbors("test", ENCLAVE, backend="pgvector")

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            semantic_neighbors("test", ENCLAVE, backend="unknown_xyz")
