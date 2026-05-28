"""Tests for pypeline.math.validators — RFC-0019 WO-EQ-001."""
from __future__ import annotations

import pytest
from pypeline.math.validators import ValidationResult, validate, register


class TestValidationResult:
    def test_passed_margin_positive(self) -> None:
        vr = ValidationResult(passed=True, score=0.9, threshold=0.85, equation_id="EQ-015")
        assert vr.margin == pytest.approx(0.05)

    def test_failed_margin_negative(self) -> None:
        vr = ValidationResult(passed=False, score=0.7, threshold=0.85, equation_id="EQ-015")
        assert vr.margin == pytest.approx(-0.15)

    def test_str_pass(self) -> None:
        vr = ValidationResult(passed=True, score=1.0, threshold=1.0, equation_id="EQ-089")
        assert "PASS" in str(vr)
        assert "EQ-089" in str(vr)

    def test_str_fail(self) -> None:
        vr = ValidationResult(passed=False, score=0.5, threshold=1.0, equation_id="EQ-089")
        assert "FAIL" in str(vr)

    def test_frozen(self) -> None:
        vr = ValidationResult(passed=True, score=1.0, threshold=1.0, equation_id="EQ-089")
        with pytest.raises(Exception):
            vr.passed = False  # type: ignore[misc]

    def test_ledger_eligible_default_true(self) -> None:
        vr = ValidationResult(passed=True, score=1.0, threshold=1.0, equation_id="EQ-001")
        assert vr.ledger_eligible is True


class TestValidateDispatch:
    def test_unknown_equation_raises(self) -> None:
        with pytest.raises(KeyError, match="No validator registered"):
            validate("EQ-999-NONEXISTENT")

    def test_dispatch_by_eq_id(self) -> None:
        result = validate("EQ-003", tests_passing_now=3000)
        assert result.equation_id == "EQ-003"

    def test_dispatch_by_name(self) -> None:
        result = validate("TestFloorPreservation", tests_passing_now=3000)
        assert result.equation_id == "EQ-003"


class TestEQ001NamespaceBleedCost:
    def test_zero_imports(self) -> None:
        r = validate("EQ-001", cross_project_imports=0, total_imports=0)
        assert r.score == 0.0
        assert r.passed

    def test_no_cross_imports(self) -> None:
        r = validate("EQ-001", cross_project_imports=0, total_imports=100)
        assert r.score == 0.0
        assert r.passed

    def test_cross_imports_fail(self) -> None:
        r = validate("EQ-001", cross_project_imports=5, total_imports=100)
        assert r.score == pytest.approx(0.5)  # 10 * 0.05 * 1.0
        assert not r.passed

    def test_custom_threshold(self) -> None:
        r = validate("EQ-001", cross_project_imports=1, total_imports=100, threshold=1.0)
        assert r.passed  # 0.1 <= 1.0


class TestEQ003TestFloor:
    def test_above_floor_passes(self) -> None:
        r = validate("EQ-003", tests_passing_now=3000)
        assert r.passed
        assert r.score == 3000.0

    def test_at_floor_passes(self) -> None:
        r = validate("EQ-003", tests_passing_now=2778)
        assert r.passed

    def test_below_floor_fails(self) -> None:
        r = validate("EQ-003", tests_passing_now=2777)
        assert not r.passed

    def test_custom_floor(self) -> None:
        r = validate("EQ-003", tests_passing_now=10, floor=5)
        assert r.passed


class TestEQ015RAMBudget:
    def test_under_threshold_passes(self) -> None:
        r = validate("EQ-015", resident_models_mb=3000.0, indexes_mb=1000.0, os_overhead_mb=500.0)
        util = (3000 + 1000 + 500) / 8028.16
        assert r.score == pytest.approx(util)
        assert r.passed

    def test_over_threshold_fails(self) -> None:
        r = validate("EQ-015", resident_models_mb=5000.0, indexes_mb=2000.0, os_overhead_mb=1000.0)
        assert not r.passed

    def test_exact_ceiling(self) -> None:
        r = validate("EQ-015", resident_models_mb=8028.16, indexes_mb=0.0, os_overhead_mb=0.0)
        assert r.score == pytest.approx(1.0)
        assert not r.passed  # 1.0 > 0.85


class TestEQ087AuditDebt:
    def test_zero_debt_passes(self) -> None:
        r = validate("EQ-087", exports_missing_audit_step=0)
        assert r.passed
        assert r.score == 0.0

    def test_any_debt_fails(self) -> None:
        r = validate("EQ-087", exports_missing_audit_step=1)
        assert not r.passed


class TestEQ089AntiSlopRatio:
    def test_no_claims_is_perfect(self) -> None:
        r = validate("EQ-089", claims_with_evidence=0, total_claims_emitted=0)
        assert r.score == 1.0
        assert r.passed

    def test_all_cited_passes(self) -> None:
        r = validate("EQ-089", claims_with_evidence=10, total_claims_emitted=10)
        assert r.passed
        assert r.score == pytest.approx(1.0)

    def test_partial_fails(self) -> None:
        r = validate("EQ-089", claims_with_evidence=7, total_claims_emitted=10)
        assert not r.passed
        assert r.score == pytest.approx(0.7)


class TestEQ027CockpitHonesty:
    def test_all_genuine_passes(self) -> None:
        r = validate("EQ-027", displayed_ok=5, unknown_displayed_as_ok=0)
        assert r.passed
        assert r.score == pytest.approx(1.0)

    def test_any_fake_green_fails(self) -> None:
        r = validate("EQ-027", displayed_ok=4, unknown_displayed_as_ok=1)
        assert not r.passed

    def test_no_displays(self) -> None:
        r = validate("EQ-027", displayed_ok=0, unknown_displayed_as_ok=0)
        assert r.score == 1.0
        assert r.passed


class TestEQ070EVI:
    def test_positive_evi_passes(self) -> None:
        r = validate("EQ-070", expected_value_with_answer=10.0, value_without_answer=5.0, query_cost=2.0)
        assert r.score == pytest.approx(3.0)
        assert r.passed

    def test_negative_evi_fails(self) -> None:
        r = validate("EQ-070", expected_value_with_answer=5.0, value_without_answer=5.0, query_cost=1.0)
        assert not r.passed


class TestEQ071RegretAgainstNoOp:
    def test_better_than_noop_passes(self) -> None:
        r = validate("EQ-071", strategy_ev=10.0, noop_ev=5.0)
        assert r.score == pytest.approx(5.0)
        assert r.passed

    def test_worse_than_noop_fails(self) -> None:
        r = validate("EQ-071", strategy_ev=3.0, noop_ev=5.0)
        assert not r.passed


class TestCockpitMetrics:
    def test_anti_slop_ratio_import(self) -> None:
        from pypeline.math import anti_slop_ratio
        score = anti_slop_ratio(claims_with_evidence=5, total_claims_emitted=5)
        assert score == pytest.approx(1.0)

    def test_cockpit_honesty_import(self) -> None:
        from pypeline.math import cockpit_honesty
        score = cockpit_honesty(displayed_ok=3, unknown_displayed_as_ok=0)
        assert score == pytest.approx(1.0)

    def test_audit_debt_import(self) -> None:
        from pypeline.math import audit_debt
        score = audit_debt(exports_missing_audit_step=2)
        assert score == pytest.approx(2.0)

    def test_math_package_imports_clean(self) -> None:
        import pypeline.math  # noqa: F401 — smoke test for ImportError
