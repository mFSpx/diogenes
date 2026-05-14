"""Tests for pypeline.math.cockpit_metrics — RFC-0019 §17 cockpit number wrappers."""
from __future__ import annotations

import pytest

from pypeline.math.cockpit_metrics import anti_slop_ratio, audit_debt, cockpit_honesty


class TestAntiSlopRatio:
    def test_returns_float(self) -> None:
        result = anti_slop_ratio(claims_with_evidence=8, total_claims_emitted=10)
        assert isinstance(result, float)

    def test_all_cited(self) -> None:
        assert anti_slop_ratio(claims_with_evidence=10, total_claims_emitted=10) == pytest.approx(1.0)

    def test_none_cited(self) -> None:
        assert anti_slop_ratio(claims_with_evidence=0, total_claims_emitted=10) == pytest.approx(0.0)

    def test_partial_citation(self) -> None:
        assert anti_slop_ratio(claims_with_evidence=8, total_claims_emitted=10) == pytest.approx(0.8)

    def test_zero_claims_is_perfect(self) -> None:
        assert anti_slop_ratio(claims_with_evidence=0, total_claims_emitted=0) == pytest.approx(1.0)

    def test_score_in_unit_interval(self) -> None:
        for cited, total in [(0, 5), (3, 5), (5, 5), (0, 1), (1, 1)]:
            score = anti_slop_ratio(claims_with_evidence=cited, total_claims_emitted=total)
            assert 0.0 <= score <= 1.0

    def test_single_claim_cited(self) -> None:
        assert anti_slop_ratio(claims_with_evidence=1, total_claims_emitted=1) == pytest.approx(1.0)

    def test_single_claim_uncited(self) -> None:
        assert anti_slop_ratio(claims_with_evidence=0, total_claims_emitted=1) == pytest.approx(0.0)


class TestCockpitHonesty:
    def test_returns_float(self) -> None:
        result = cockpit_honesty(displayed_ok=10, unknown_displayed_as_ok=0)
        assert isinstance(result, float)

    def test_all_genuine_ok(self) -> None:
        assert cockpit_honesty(displayed_ok=10, unknown_displayed_as_ok=0) == pytest.approx(1.0)

    def test_all_fake_green(self) -> None:
        assert cockpit_honesty(displayed_ok=0, unknown_displayed_as_ok=10) == pytest.approx(0.0)

    def test_mixed(self) -> None:
        assert cockpit_honesty(displayed_ok=9, unknown_displayed_as_ok=1) == pytest.approx(0.9)

    def test_both_zero_is_perfect(self) -> None:
        assert cockpit_honesty(displayed_ok=0, unknown_displayed_as_ok=0) == pytest.approx(1.0)

    def test_score_in_unit_interval(self) -> None:
        for ok, fake in [(0, 0), (5, 5), (10, 0), (0, 10), (7, 3)]:
            score = cockpit_honesty(displayed_ok=ok, unknown_displayed_as_ok=fake)
            assert 0.0 <= score <= 1.0

    def test_single_genuine(self) -> None:
        assert cockpit_honesty(displayed_ok=1, unknown_displayed_as_ok=0) == pytest.approx(1.0)

    def test_single_fake(self) -> None:
        assert cockpit_honesty(displayed_ok=0, unknown_displayed_as_ok=1) == pytest.approx(0.0)


class TestAuditDebt:
    def test_returns_float(self) -> None:
        result = audit_debt(exports_missing_audit_step=0)
        assert isinstance(result, float)

    def test_no_debt(self) -> None:
        assert audit_debt(exports_missing_audit_step=0) == pytest.approx(0.0)

    def test_some_debt(self) -> None:
        assert audit_debt(exports_missing_audit_step=3) == pytest.approx(3.0)

    def test_large_debt(self) -> None:
        assert audit_debt(exports_missing_audit_step=100) == pytest.approx(100.0)

    def test_single_missing(self) -> None:
        assert audit_debt(exports_missing_audit_step=1) == pytest.approx(1.0)

    def test_score_is_count_not_ratio(self) -> None:
        assert audit_debt(exports_missing_audit_step=7) == pytest.approx(7.0)
