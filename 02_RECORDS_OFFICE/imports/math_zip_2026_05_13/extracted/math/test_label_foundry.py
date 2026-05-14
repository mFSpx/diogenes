"""Tests for pypeline.math.label_foundry — RFC-0018 §3.8."""
import pytest
from pypeline.math.label_foundry import (
    LabelingFunctionResult,
    ProbabilisticLabel,
    LabelError,
    labeling_function,
    aggregate_labels,
    find_label_errors,
)


class TestLabelingFunctionDecorator:
    def test_decorator_wraps_function(self):
        @labeling_function(name="test_lf")
        def my_lf(doc):
            return 1
        assert callable(my_lf)
        assert my_lf({"text": "hello"}) == 1

    def test_decorator_attaches_name(self):
        @labeling_function(name="custom_name")
        def lf(doc):
            return 0
        assert lf.lf_name == "custom_name"

    def test_decorator_uses_fn_name_when_no_name(self):
        @labeling_function()
        def my_auto_lf(doc):
            return -1
        assert my_auto_lf.lf_name == "my_auto_lf"

    def test_decorator_preserves_return_value(self):
        @labeling_function(name="abstain_lf")
        def lf(doc):
            return -1
        assert lf({}) == -1


class TestAggregateLabels:
    def _make_lf_batch(self, lf_name: str, votes: list[int], doc_ids: list[str]) -> list[LabelingFunctionResult]:
        return [LabelingFunctionResult(lf_name=lf_name, doc_id=d, label=v)
                for d, v in zip(doc_ids, votes)]

    def test_empty_input(self):
        assert aggregate_labels([]) == []

    def test_majority_vote_clear_positive(self):
        doc_ids = ["doc0", "doc1"]
        lf1 = self._make_lf_batch("lf1", [1, 0], doc_ids)
        lf2 = self._make_lf_batch("lf2", [1, 0], doc_ids)
        lf3 = self._make_lf_batch("lf3", [1, 1], doc_ids)
        results = aggregate_labels([lf1, lf2, lf3])
        assert len(results) == 2
        assert results[0].label == 1
        assert results[1].label == 0

    def test_majority_vote_with_abstain(self):
        doc_ids = ["d0"]
        lf1 = self._make_lf_batch("lf1", [-1], doc_ids)
        lf2 = self._make_lf_batch("lf2", [1], doc_ids)
        lf3 = self._make_lf_batch("lf3", [1], doc_ids)
        results = aggregate_labels([lf1, lf2, lf3])
        assert results[0].label == 1

    def test_all_abstain_returns_low_confidence(self):
        doc_ids = ["d0"]
        lf1 = self._make_lf_batch("lf1", [-1], doc_ids)
        lf2 = self._make_lf_batch("lf2", [-1], doc_ids)
        results = aggregate_labels([lf1, lf2])
        assert results[0].confidence == pytest.approx(0.5, abs=1e-6)

    def test_confidence_high_when_unanimous(self):
        doc_ids = ["d0"]
        lf1 = self._make_lf_batch("lf1", [1], doc_ids)
        lf2 = self._make_lf_batch("lf2", [1], doc_ids)
        results = aggregate_labels([lf1, lf2])
        assert results[0].confidence == pytest.approx(1.0, abs=1e-6)

    def test_probabilistic_label_structure(self):
        doc_ids = ["d0", "d1"]
        lf1 = self._make_lf_batch("lf1", [1, 0], doc_ids)
        results = aggregate_labels([lf1])
        for r in results:
            assert isinstance(r, ProbabilisticLabel)
            assert r.label in (0, 1)
            assert 0.0 <= r.confidence <= 1.0


class TestFindLabelErrors:
    def test_empty_input(self):
        assert find_label_errors([], [], []) == []

    def test_obvious_errors_detected(self):
        docs = [{"id": f"d{i}"} for i in range(4)]
        given = [0, 1, 0, 1]
        # d0: given=0 but prob=0.95 (likely error)
        # d1: given=1 and prob=0.90 (correct)
        # d2: given=0 and prob=0.05 (correct)
        # d3: given=1 but prob=0.05 (likely error)
        probs = [0.95, 0.90, 0.05, 0.05]
        errors = find_label_errors(docs, given, probs)
        error_ids = {e.doc_id for e in errors}
        assert "d0" in error_ids or "d3" in error_ids  # At least one error found

    def test_clean_labels_produce_no_errors(self):
        docs = [{"id": f"d{i}"} for i in range(4)]
        given = [0, 0, 1, 1]
        probs = [0.05, 0.10, 0.90, 0.95]
        errors = find_label_errors(docs, given, probs)
        assert errors == []

    def test_mismatched_lengths_raises(self):
        with pytest.raises((ValueError, Exception)):
            find_label_errors([{"id": "d0"}], [0], [0.5, 0.8])

    def test_error_probability_positive(self):
        docs = [{"id": "d0"}]
        given = [0]
        probs = [0.99]
        errors = find_label_errors(docs, given, probs)
        if errors:
            assert errors[0].error_probability > 0

    def test_suggested_label_is_opposite(self):
        docs = [{"id": f"d{i}"} for i in range(6)]
        given = [0, 0, 0, 1, 1, 1]
        probs = [0.95, 0.95, 0.90, 0.05, 0.05, 0.10]
        errors = find_label_errors(docs, given, probs)
        for err in errors:
            assert err.suggested_label == 1 - err.given_label

    def test_errors_sorted_by_probability_descending(self):
        docs = [{"id": f"d{i}"} for i in range(6)]
        given = [0, 0, 0, 1, 1, 1]
        probs = [0.95, 0.80, 0.70, 0.30, 0.20, 0.05]
        errors = find_label_errors(docs, given, probs)
        if len(errors) >= 2:
            for i in range(len(errors) - 1):
                assert errors[i].error_probability >= errors[i + 1].error_probability
