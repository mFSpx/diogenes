"""Tests for pypeline.math.privacy — RFC-0018 §3.15 WO-BM-035/036."""
from __future__ import annotations

import hashlib
import math

import pytest

from pypeline.math.privacy import (
    AnonResult,
    DPResult,
    PrivacyBudget,
    anonymize_for_indexing,
    dp_aggregate,
    reconstruction_risk_score,
)


# ---------------------------------------------------------------------------
# anonymize_for_indexing
# ---------------------------------------------------------------------------

class TestAnonymizeForIndexing:

    def test_returns_anon_result(self):
        result = anonymize_for_indexing("hello world")
        assert isinstance(result, AnonResult)

    def test_original_hash_is_sha256(self):
        text = "contact me at user@example.com"
        result = anonymize_for_indexing(text)
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert result.original_hash == expected

    def test_operator_applied_recorded(self):
        for op in ("redact", "replace", "hash", "encrypt"):
            r = anonymize_for_indexing("test", operator=op)
            assert r.operator_applied == op

    def test_email_detected_redact(self):
        result = anonymize_for_indexing("email me at foo@bar.com today", operator="redact")
        assert "EMAIL" in result.entities_found
        assert "foo@bar.com" not in result.anonymized_text
        assert "[EMAIL]" in result.anonymized_text

    def test_email_detected_replace(self):
        result = anonymize_for_indexing("reach out at alice@example.org", operator="replace")
        assert "EMAIL" in result.entities_found
        assert "alice@example.org" not in result.anonymized_text
        assert "<EMAIL>" in result.anonymized_text

    def test_email_detected_hash(self):
        result = anonymize_for_indexing("test@domain.io is the address", operator="hash")
        assert "EMAIL" in result.entities_found
        assert "test@domain.io" not in result.anonymized_text
        # hash replacement is 12-char hex
        assert len([c for c in result.anonymized_text if c in "0123456789abcdef"]) >= 12

    def test_email_detected_encrypt(self):
        import base64
        result = anonymize_for_indexing("send to x@y.com please", operator="encrypt")
        assert "EMAIL" in result.entities_found
        assert "x@y.com" not in result.anonymized_text
        # base64 encoded value should appear
        expected_b64 = base64.b64encode(b"x@y.com").decode()
        assert expected_b64 in result.anonymized_text

    def test_phone_detected(self):
        result = anonymize_for_indexing("call 604-555-1234 for details")
        assert "PHONE" in result.entities_found
        assert "604-555-1234" not in result.anonymized_text

    def test_ssn_detected(self):
        result = anonymize_for_indexing("SSN is 123-45-6789")
        assert "SSN" in result.entities_found
        assert "123-45-6789" not in result.anonymized_text

    def test_no_pii_clean_text(self):
        result = anonymize_for_indexing("the quick brown fox")
        assert result.entities_found == ()
        assert result.anonymized_text == "the quick brown fox"

    def test_empty_string(self):
        result = anonymize_for_indexing("")
        assert result.anonymized_text == ""
        assert result.entities_found == ()

    def test_multiple_pii_types(self):
        text = "contact foo@bar.com or 604-555-9999"
        result = anonymize_for_indexing(text)
        assert "EMAIL" in result.entities_found
        assert "PHONE" in result.entities_found
        assert "foo@bar.com" not in result.anonymized_text
        assert "604-555-9999" not in result.anonymized_text

    def test_url_detected(self):
        result = anonymize_for_indexing("visit https://example.com/path?q=1")
        assert "URL" in result.entities_found

    def test_ip_detected(self):
        result = anonymize_for_indexing("connect to 192.168.1.100")
        assert "IP_ADDRESS" in result.entities_found


# ---------------------------------------------------------------------------
# PrivacyBudget
# ---------------------------------------------------------------------------

class TestPrivacyBudget:

    def test_initial_budget(self):
        b = PrivacyBudget(total_epsilon=5.0, budget_remaining=5.0)
        assert b.budget_remaining == 5.0

    def test_default_budget(self):
        b = PrivacyBudget()
        assert b.total_epsilon == 10.0
        assert b.budget_remaining == 10.0

    def test_consume_reduces_budget(self):
        b = PrivacyBudget(total_epsilon=10.0, budget_remaining=10.0)
        b.consume(3.0)
        assert abs(b.budget_remaining - 7.0) < 1e-10

    def test_consume_exhaustion_raises(self):
        b = PrivacyBudget(total_epsilon=2.0, budget_remaining=2.0)
        with pytest.raises(ValueError, match="DP budget exhausted"):
            b.consume(3.0)

    def test_consume_exact_remaining_ok(self):
        b = PrivacyBudget(total_epsilon=1.0, budget_remaining=1.0)
        b.consume(1.0)
        assert b.budget_remaining == 0.0


# ---------------------------------------------------------------------------
# dp_aggregate
# ---------------------------------------------------------------------------

class TestDPAggregate:

    def test_returns_dp_result(self):
        r = dp_aggregate("mean", [1.0, 2.0, 3.0])
        assert isinstance(r, DPResult)

    def test_query_id_is_string(self):
        r = dp_aggregate("count", [1.0, 2.0])
        assert isinstance(r.query_id, str) and len(r.query_id) == 12

    def test_epsilon_consumed_recorded(self):
        r = dp_aggregate("sum", [1.0, 2.0, 3.0], epsilon=0.5)
        assert r.epsilon_consumed == 0.5

    def test_noise_scale_positive(self):
        r = dp_aggregate("mean", [0.0, 10.0], epsilon=1.0)
        assert r.noise_scale > 0.0

    def test_reconstruction_risk_in_range(self):
        r = dp_aggregate("mean", [1.0, 2.0, 3.0])
        assert 0.0 <= r.reconstruction_risk_score <= 1.0

    def test_high_epsilon_high_risk(self):
        r = dp_aggregate("mean", [1.0, 2.0], epsilon=10.0)
        assert r.reconstruction_risk_score == 1.0

    def test_low_epsilon_low_risk(self):
        r = dp_aggregate("mean", [1.0, 2.0], epsilon=0.1)
        assert r.reconstruction_risk_score < 0.1

    def test_budget_consumed(self):
        budget = PrivacyBudget(total_epsilon=5.0, budget_remaining=5.0)
        dp_aggregate("mean", [1.0, 2.0, 3.0], epsilon=2.0, budget=budget)
        assert abs(budget.budget_remaining - 3.0) < 1e-10

    def test_budget_reflected_in_result(self):
        budget = PrivacyBudget(total_epsilon=5.0, budget_remaining=5.0)
        r = dp_aggregate("sum", [1.0], epsilon=1.0, budget=budget)
        assert abs(r.budget_remaining - 4.0) < 1e-10

    def test_budget_exhausted_raises(self):
        budget = PrivacyBudget(total_epsilon=1.0, budget_remaining=0.5)
        with pytest.raises(ValueError, match="DP budget exhausted"):
            dp_aggregate("mean", [1.0], epsilon=1.0, budget=budget)

    def test_no_budget_infinite_remaining(self):
        r = dp_aggregate("mean", [1.0, 2.0])
        assert r.budget_remaining == float("inf")

    def test_query_types_all_run(self):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        for qt in ("mean", "sum", "count", "median", "variance"):
            r = dp_aggregate(qt, data, epsilon=1.0)
            assert isinstance(r.result, float)

    def test_count_noise_scale_is_one_per_epsilon(self):
        r = dp_aggregate("count", [1.0, 2.0, 3.0], epsilon=2.0)
        # sensitivity=1.0 → noise_scale = 1/epsilon = 0.5
        assert abs(r.noise_scale - 0.5) < 1e-10

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            dp_aggregate("mean", [])

    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError, match="positive"):
            dp_aggregate("mean", [1.0, 2.0], epsilon=-1.0)

    def test_zero_epsilon_raises(self):
        with pytest.raises(ValueError, match="positive"):
            dp_aggregate("mean", [1.0], epsilon=0.0)

    def test_unknown_query_type_raises(self):
        with pytest.raises(ValueError, match="unknown query_type"):
            dp_aggregate("mode", [1.0, 2.0])

    def test_result_finite(self):
        r = dp_aggregate("mean", [0.0, 1.0], epsilon=1.0)
        assert math.isfinite(r.result)

    def test_single_element_count(self):
        r = dp_aggregate("count", [42.0], epsilon=1.0)
        # true count = 1, result should be near 1 with noise
        assert isinstance(r.result, float)


# ---------------------------------------------------------------------------
# reconstruction_risk_score
# ---------------------------------------------------------------------------

class TestReconstructionRiskScore:

    def test_empty_plan_zero_risk(self):
        assert reconstruction_risk_score([]) == 0.0

    def test_single_low_epsilon_plan(self):
        score = reconstruction_risk_score([{"epsilon": 0.1, "query_type": "mean"}])
        assert 0.0 <= score <= 1.0
        assert score < 0.5

    def test_high_epsilon_high_risk(self):
        plan = [{"epsilon": 10.0, "query_type": "sum"}]
        score = reconstruction_risk_score(plan)
        assert score >= 0.5

    def test_many_queries_increase_risk(self):
        plan = [{"epsilon": 0.5, "query_type": "mean"} for _ in range(20)]
        score = reconstruction_risk_score(plan)
        assert score > reconstruction_risk_score([{"epsilon": 0.5, "query_type": "mean"}])

    def test_result_bounded_zero_one(self):
        plan = [{"epsilon": 100.0}] * 100
        score = reconstruction_risk_score(plan)
        assert 0.0 <= score <= 1.0

    def test_missing_epsilon_defaults_to_one(self):
        score = reconstruction_risk_score([{"query_type": "mean"}])
        # epsilon defaults to 1.0
        assert score == reconstruction_risk_score([{"epsilon": 1.0, "query_type": "mean"}])

    def test_monotone_in_epsilon(self):
        low = reconstruction_risk_score([{"epsilon": 1.0}])
        high = reconstruction_risk_score([{"epsilon": 9.0}])
        assert high > low

    def test_monotone_in_query_count(self):
        few = reconstruction_risk_score([{"epsilon": 1.0}] * 2)
        many = reconstruction_risk_score([{"epsilon": 1.0}] * 10)
        assert many > few
