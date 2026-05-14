"""Tests for pypeline.math.validators_extended — RFC-0019 EQ-002..EQ-100."""
from __future__ import annotations

import math
import pytest
from pypeline.math.validators import validate, ValidationResult
import pypeline.math.validators_extended  # noqa: F401 — ensures registration


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------

class TestRegistryWiring:
    def test_all_eq_ids_registered(self) -> None:
        expected = [
            "EQ-002", "EQ-004", "EQ-005", "EQ-006", "EQ-007", "EQ-008", "EQ-009", "EQ-010",
            "EQ-011", "EQ-012", "EQ-013", "EQ-014", "EQ-016", "EQ-017", "EQ-018", "EQ-019",
            "EQ-020", "EQ-021", "EQ-022", "EQ-023", "EQ-024", "EQ-025", "EQ-026", "EQ-028",
            "EQ-029", "EQ-030", "EQ-031", "EQ-032", "EQ-033", "EQ-034", "EQ-035", "EQ-036",
            "EQ-037", "EQ-038", "EQ-039", "EQ-040", "EQ-041", "EQ-042", "EQ-043", "EQ-044",
            "EQ-045", "EQ-046", "EQ-047", "EQ-048", "EQ-049", "EQ-050", "EQ-051", "EQ-052",
            "EQ-053", "EQ-054", "EQ-055", "EQ-056", "EQ-057", "EQ-058", "EQ-059", "EQ-060",
            "EQ-061", "EQ-062", "EQ-063", "EQ-064", "EQ-065", "EQ-066", "EQ-067", "EQ-068",
            "EQ-069", "EQ-072", "EQ-073", "EQ-074", "EQ-075", "EQ-076", "EQ-077", "EQ-078",
            "EQ-079", "EQ-080", "EQ-081", "EQ-082", "EQ-083", "EQ-084", "EQ-085", "EQ-086",
            "EQ-088", "EQ-090", "EQ-091", "EQ-092", "EQ-093", "EQ-094", "EQ-095", "EQ-096",
            "EQ-097", "EQ-098", "EQ-099", "EQ-100",
        ]
        for eq_id in expected:
            r = validate.__module__  # just ensure module available
            # actual check: dispatch must not raise KeyError
            try:
                validate(eq_id)
            except TypeError:
                pass  # missing kwargs — that's fine, it's registered

    def test_named_aliases_registered(self) -> None:
        aliases = [
            "LedgerIntegrity", "SingleWriterDiscipline", "LaneDisciplineViolation",
            "ConfidenceCalibrationError", "SelfConsistencyConfidence",
            "TruthInfra", "Slop", "SystemHealth", "CaseShipReadiness",
            "RefusalsByVibes", "CatchAllMoralismDetector",
        ]
        for alias in aliases:
            try:
                validate(alias)
            except TypeError:
                pass  # wrong kwargs — name IS registered

    def test_unknown_eq_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            validate("EQ-999")


# ---------------------------------------------------------------------------
# §1. KERNEL
# ---------------------------------------------------------------------------

class TestKernelEquations:
    def test_eq002_ledger_integrity_pass(self) -> None:
        r = validate("EQ-002", chain_valid=True)
        assert r.passed and r.score == 1.0

    def test_eq002_ledger_integrity_fail(self) -> None:
        r = validate("EQ-002", chain_valid=False)
        assert not r.passed and r.score == 0.0

    def test_eq004_single_writer_pass(self) -> None:
        r = validate("EQ-004", actor_set=["cli_operator"])
        assert r.passed

    def test_eq004_single_writer_fail(self) -> None:
        r = validate("EQ-004", actor_set=["cli_operator", "rogue_writer"])
        assert not r.passed


# ---------------------------------------------------------------------------
# §2. COGNITION
# ---------------------------------------------------------------------------

class TestCognitionEquations:
    def test_eq005_no_violations(self) -> None:
        r = validate("EQ-005", violations=[])
        assert r.passed and r.score == 0.0

    def test_eq005_violations_fail(self) -> None:
        r = validate("EQ-005", violations=[1.0, 2.0])
        assert not r.passed and r.score == pytest.approx(3.0)

    def test_eq006_ece_perfect_calibration(self) -> None:
        bins = [{"accuracy": 0.9, "confidence": 0.9, "count": 100}]
        r = validate("EQ-006", bins=bins, total_samples=100)
        assert r.passed and r.score == pytest.approx(0.0)

    def test_eq007_self_consistency_pass(self) -> None:
        r = validate("EQ-007", answer_counts={"yes": 8, "no": 2}, k=10)
        assert r.passed and r.score == pytest.approx(0.8)

    def test_eq007_self_consistency_fail(self) -> None:
        r = validate("EQ-007", answer_counts={"yes": 5, "no": 5}, k=10)
        assert not r.passed and r.score == pytest.approx(0.5)

    def test_eq008_adversary_pass(self) -> None:
        r = validate("EQ-008", steel_man_evidence_cites=3, steel_man_quality=0.9, passes_self_consistency=True)
        assert r.passed

    def test_eq009_hidden_info_coverage_pass(self) -> None:
        r = validate("EQ-009", enumerated_unknowns=8, suspected_unknown_unknowns=2)
        assert r.passed and r.score == pytest.approx(0.8)

    def test_eq010_provider_health_sigmoid(self) -> None:
        r = validate("EQ-010", uptime_ratio=0.99, error_rate=0.01, latency_p99=100.0,
                     latency_budget=500.0, cost_overrun_ratio=0.0)
        assert r.passed
        assert 0 < r.score < 1


# ---------------------------------------------------------------------------
# §3. ROUTING
# ---------------------------------------------------------------------------

class TestRoutingEquations:
    def test_eq011_lora_promotion_all_pass(self) -> None:
        r = validate("EQ-011", f1_new=0.95, f1_old=0.90, ece_new=0.05, ece_old=0.08,
                     latency_p99_new=200.0, latency_budget=500.0,
                     adv_f1_new=0.87, clean_f1_new=0.95,
                     ram_new_mb=2000.0, pool_budget_mb=4000.0)
        assert r.passed and r.score == 1.0

    def test_eq011_lora_promotion_f1_fail(self) -> None:
        r = validate("EQ-011", f1_new=0.91, f1_old=0.90, ece_new=0.05, ece_old=0.08,
                     latency_p99_new=200.0, latency_budget=500.0,
                     adv_f1_new=0.87, clean_f1_new=0.95,
                     ram_new_mb=2000.0, pool_budget_mb=4000.0)
        assert not r.passed

    def test_eq012_drift_score_pass(self) -> None:
        r = validate("EQ-012", kl_divergence=0.10)
        assert r.passed

    def test_eq012_drift_score_fail(self) -> None:
        r = validate("EQ-012", kl_divergence=0.20)
        assert not r.passed

    def test_eq013_calibration_gap_pass(self) -> None:
        r = validate("EQ-013", confidence_at_threshold=0.80, empirical_accuracy_at_threshold=0.82)
        assert r.passed

    def test_eq014_cost_ratio_pass(self) -> None:
        r = validate("EQ-014", tokens=1000, cost_per_token=0.001, evi=2.0)
        assert r.passed and r.score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# §4. ONTOLOGY
# ---------------------------------------------------------------------------

class TestOntologyEquations:
    def test_eq016_concept_coverage_full(self) -> None:
        r = validate("EQ-016", concepts_used={"entity", "claim"}, global_vocab={"entity", "claim", "evidence"})
        assert r.passed and r.score == 1.0

    def test_eq016_concept_coverage_partial(self) -> None:
        r = validate("EQ-016", concepts_used={"entity", "unknown_concept"}, global_vocab={"entity"})
        assert not r.passed and r.score == pytest.approx(0.5)

    def test_eq017_domain_pack_purity(self) -> None:
        r = validate("EQ-017", cross_domain_imports=2, total_imports=100)
        assert r.passed and r.score == pytest.approx(0.98)

    def test_eq018_ontology_drift_no_rfc(self) -> None:
        r = validate("EQ-018", delta_concept_yamls=3, companion_rfc_ratified=False)
        assert not r.passed

    def test_eq018_ontology_drift_with_rfc(self) -> None:
        r = validate("EQ-018", delta_concept_yamls=3, companion_rfc_ratified=True)
        assert r.passed

    def test_eq019_seven_tuple_all_present(self) -> None:
        r = validate("EQ-019", entity_present=True, channel_present=True, asset_present=True,
                     task_present=True, route_present=True, action_present=True, evidence_present=True)
        assert r.passed and r.score == 1.0

    def test_eq019_seven_tuple_missing(self) -> None:
        r = validate("EQ-019", entity_present=True, channel_present=True, asset_present=False,
                     task_present=True, route_present=True, action_present=True, evidence_present=True)
        assert not r.passed


# ---------------------------------------------------------------------------
# §5. SOVEREIGN DAEMON
# ---------------------------------------------------------------------------

class TestSovereignDaemonEquations:
    def test_eq020_soul_law_isolation_pass(self) -> None:
        r = validate("EQ-020", law_functions_import_soul_field=False)
        assert r.passed and r.score == 1.0

    def test_eq020_soul_law_isolation_fail(self) -> None:
        r = validate("EQ-020", law_functions_import_soul_field=True)
        assert not r.passed and r.score == 0.0

    def test_eq021_memory_staleness_fresh(self) -> None:
        from datetime import datetime, timezone, timedelta
        recent = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        r = validate("EQ-021", last_verified_iso=recent)
        assert r.passed

    def test_eq021_memory_staleness_stale(self) -> None:
        r = validate("EQ-021", last_verified_iso="2020-01-01T00:00:00+00:00")
        assert not r.passed

    def test_eq022_death_handoff_complete(self) -> None:
        r = validate("EQ-022", required_fields=["task_state", "ledger_op"], provided_fields=["task_state", "ledger_op"])
        assert r.passed and r.score == 1.0

    def test_eq022_death_handoff_incomplete(self) -> None:
        r = validate("EQ-022", required_fields=["task_state", "ledger_op"], provided_fields=["task_state"])
        assert not r.passed and r.score == pytest.approx(0.5)

    def test_eq023_burn_rate_ok(self) -> None:
        r = validate("EQ-023", tokens_used=1000, tokens_budgeted=1000)
        assert r.passed and r.score == pytest.approx(1.0)

    def test_eq023_burn_rate_over(self) -> None:
        r = validate("EQ-023", tokens_used=1500, tokens_budgeted=1000)
        assert not r.passed

    def test_eq024_manifest_integrity_pass(self) -> None:
        r = validate("EQ-024", signature_valid=True)
        assert r.passed

    def test_eq024_manifest_integrity_fail(self) -> None:
        r = validate("EQ-024", signature_valid=False)
        assert not r.passed


# ---------------------------------------------------------------------------
# §6. IMMUNE SYSTEM
# ---------------------------------------------------------------------------

class TestImmuneSystemEquations:
    def test_eq025_eval_suite_pass(self) -> None:
        r = validate("EQ-025", pass_rate=0.95, suite_threshold=0.90,
                     adversarial_pass_rate=0.88, clean_pass_rate=0.95)
        assert r.passed

    def test_eq025_eval_suite_fail_pass_rate(self) -> None:
        r = validate("EQ-025", pass_rate=0.85, suite_threshold=0.90,
                     adversarial_pass_rate=0.88, clean_pass_rate=0.95)
        assert not r.passed

    def test_eq026_quarantine_no_triggers(self) -> None:
        r = validate("EQ-026")
        assert r.passed and r.score == 0.0

    def test_eq026_quarantine_injection_trigger(self) -> None:
        r = validate("EQ-026", injection_signature_hit=True)
        assert not r.passed and r.score == 1.0

    def test_eq026_quarantine_operator_stop(self) -> None:
        r = validate("EQ-026", operator_stop_kill=True)
        assert not r.passed

    def test_eq028_ledger_op_density_pass(self) -> None:
        r = validate("EQ-028", ledger_ops_emitted=10, disk_mutations=10)
        assert r.passed and r.score == 1.0

    def test_eq028_ledger_op_density_fail(self) -> None:
        r = validate("EQ-028", ledger_ops_emitted=5, disk_mutations=10)
        assert not r.passed and r.score == pytest.approx(0.5)

    def test_eq029_adversarial_robustness_pass(self) -> None:
        r = validate("EQ-029", f1_adversarial=0.87, f1_clean=0.95)
        assert r.passed

    def test_eq030_incident_escalation_below(self) -> None:
        r = validate("EQ-030", signature_hits_last_24h=2)
        assert r.passed

    def test_eq030_incident_escalation_at_threshold(self) -> None:
        r = validate("EQ-030", signature_hits_last_24h=3)
        assert not r.passed


# ---------------------------------------------------------------------------
# §7. PROCESS CONTROL
# ---------------------------------------------------------------------------

class TestProcessControlEquations:
    def test_eq031_transition_legal(self) -> None:
        r = validate("EQ-031", from_state="PENDING", to_state="ACTIVE",
                     allowed_transitions=[["PENDING", "ACTIVE"], ["ACTIVE", "DONE"]])
        assert r.passed

    def test_eq031_transition_illegal(self) -> None:
        r = validate("EQ-031", from_state="DONE", to_state="PENDING",
                     allowed_transitions=[["PENDING", "ACTIVE"]])
        assert not r.passed

    def test_eq032_evidence_completeness_full(self) -> None:
        r = validate("EQ-032", required_evidence=["receipt", "id"], attached_evidence=["receipt", "id"])
        assert r.passed

    def test_eq035_replayability_full(self) -> None:
        r = validate("EQ-035", steps_with_decision_trace=10, total_steps=10)
        assert r.passed and r.score == 1.0

    def test_eq035_replayability_partial(self) -> None:
        r = validate("EQ-035", steps_with_decision_trace=7, total_steps=10)
        assert not r.passed

    def test_eq036_consent_fresh(self) -> None:
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        r = validate("EQ-036", consent_timestamp_iso=now_iso, max_age_seconds=300)
        assert r.passed

    def test_eq036_consent_stale(self) -> None:
        r = validate("EQ-036", consent_timestamp_iso="2020-01-01T00:00:00+00:00", max_age_seconds=300)
        assert not r.passed


# ---------------------------------------------------------------------------
# §8. INVESTIGATIVE WORKBENCH
# ---------------------------------------------------------------------------

class TestInvestigativeWorkbenchEquations:
    def test_eq037_packet_strength_all_observed(self) -> None:
        r = validate("EQ-037", observed_claims=10, inferred_claims=0, not_proven_claims=0)
        assert r.passed and r.score == pytest.approx(1.0)

    def test_eq037_packet_strength_mixed(self) -> None:
        r = validate("EQ-037", observed_claims=5, inferred_claims=5, not_proven_claims=0)
        assert r.score == pytest.approx((5 + 5 * 0.3) / 10)

    def test_eq039_truth_band_laundering_pass(self) -> None:
        r = validate("EQ-039", observed_ratio_now=0.8, observed_ratio_baseline=0.75)
        assert r.passed

    def test_eq039_truth_band_laundering_fail(self) -> None:
        r = validate("EQ-039", observed_ratio_now=0.3, observed_ratio_baseline=0.8)
        assert not r.passed

    def test_eq040_lead_actionability_pass(self) -> None:
        r = validate("EQ-040", confidence=0.9, evidence_strength=0.8, case_relevance=0.9,
                     cost=1.0, dismissed_pattern_match=0.0)
        assert r.passed

    def test_eq041_dismissal_valid(self) -> None:
        r = validate("EQ-041", dismissal_carries_cited_disproof=True, falsification_criteria_met=True)
        assert r.passed

    def test_eq041_dismissal_invalid_no_disproof(self) -> None:
        r = validate("EQ-041", dismissal_carries_cited_disproof=False, falsification_criteria_met=True)
        assert not r.passed

    def test_eq043_contradiction_sharpness_pass(self) -> None:
        r = validate("EQ-043", source_independence_a=0.9, source_independence_b=0.9,
                     structural_incompatibility=0.8)
        assert r.passed

    def test_eq044_tier1_share_pass(self) -> None:
        r = validate("EQ-044", deterministic_contradictions=8, slm_inferential_contradictions=2)
        assert r.passed and r.score == pytest.approx(0.8)

    def test_eq044_tier1_share_fail(self) -> None:
        r = validate("EQ-044", deterministic_contradictions=2, slm_inferential_contradictions=8)
        assert not r.passed

    def test_eq045_mode_bleed_zero(self) -> None:
        r = validate("EQ-045", claims_outside_allowed_sources=0)
        assert r.passed

    def test_eq046_citation_density_full(self) -> None:
        r = validate("EQ-046", cited_claims=10, total_claims=10)
        assert r.passed and r.score == 1.0

    def test_eq047_steel_man_valid(self) -> None:
        r = validate("EQ-047", word_count=150, counter_evidence_cites=3, passes_sc_vote=True)
        assert r.passed

    def test_eq047_steel_man_too_short(self) -> None:
        r = validate("EQ-047", word_count=50, counter_evidence_cites=3, passes_sc_vote=True)
        assert not r.passed


# ---------------------------------------------------------------------------
# §9. PYPER
# ---------------------------------------------------------------------------

class TestPyperEquations:
    def test_eq048_firewall_clean(self) -> None:
        r = validate("EQ-048", outbound_contains_banned_identifier=False)
        assert r.passed and r.score == 1.0

    def test_eq048_firewall_blocked(self) -> None:
        r = validate("EQ-048", outbound_contains_banned_identifier=True)
        assert not r.passed and r.score == 0.0

    def test_eq050_revenue_reconciliation_pass(self) -> None:
        r = validate("EQ-050", ledger_revenue=1000.0, platform_payout=950.0, processor_fees=50.0)
        assert r.passed and r.score == pytest.approx(0.0)

    def test_eq051_chargeback_risk_low(self) -> None:
        r = validate("EQ-051", refund_phrase_score=0.1, dispute_signature=0.1,
                     fan_history_anomaly=0.1, payment_window_anomaly=0.1)
        assert r.passed

    def test_eq052_consent_coverage_full(self) -> None:
        r = validate("EQ-052", asset_scope_within_consent=True, consent_active=True, consent_not_expired=True)
        assert r.passed and r.score == 1.0

    def test_eq052_consent_coverage_expired(self) -> None:
        r = validate("EQ-052", asset_scope_within_consent=True, consent_active=True, consent_not_expired=False)
        assert not r.passed


# ---------------------------------------------------------------------------
# §10. STORAGE MESH
# ---------------------------------------------------------------------------

class TestStorageMeshEquations:
    def test_eq053_enclave_integrity_all_brokered(self) -> None:
        r = validate("EQ-053", cross_enclave_reads_with_broker=10, total_cross_enclave_reads=10)
        assert r.passed and r.score == 1.0

    def test_eq054_dp_budget_within(self) -> None:
        r = validate("EQ-054", epsilon_used=0.5, epsilon_total_budget=1.0)
        assert r.passed and r.score == pytest.approx(0.5)

    def test_eq054_dp_budget_exhausted(self) -> None:
        r = validate("EQ-054", epsilon_used=1.1, epsilon_total_budget=1.0)
        assert not r.passed

    def test_eq056_entity_merge_confidence_pass(self) -> None:
        r = validate("EQ-056", name_match=0.95, dob_match=0.90, address_match=0.85,
                     source_independence=0.80, public_record_corroboration=0.85)
        assert r.passed

    def test_eq058_corpus_freshness_pass(self) -> None:
        r = validate("EQ-058", files_with_current_sha256=97, total_files=100)
        assert r.passed and r.score == pytest.approx(0.97)


# ---------------------------------------------------------------------------
# §11. STRATEGIC INTELLIGENCE
# ---------------------------------------------------------------------------

class TestStrategicIntelligenceEquations:
    def test_eq059_influence_leverage_pass(self) -> None:
        r = validate("EQ-059", influence_score=2.0, uncertainty_reduction=0.8,
                     case_relevance=0.9, opsec_self_risk=0.1)
        assert r.passed

    def test_eq060_pressure_cascade_no_legal_basis(self) -> None:
        r = validate("EQ-060", actor_ppr_weights=[0.5, 0.3], downstream_dependencies=[0.8, 0.6],
                     all_have_legal_basis=False)
        assert not r.passed and r.score == 0.0

    def test_eq060_pressure_cascade_with_legal_basis(self) -> None:
        r = validate("EQ-060", actor_ppr_weights=[0.5, 0.3], downstream_dependencies=[0.8, 0.6],
                     all_have_legal_basis=True)
        assert r.passed

    def test_eq061_swarm_coordination_low(self) -> None:
        r = validate("EQ-061", signal_strengths=[0.1, 0.1, 0.1])
        assert r.passed

    def test_eq061_swarm_coordination_high(self) -> None:
        r = validate("EQ-061", signal_strengths=[0.9, 0.9, 0.9])
        assert not r.passed

    def test_eq062_abba3_movement_pass(self) -> None:
        r = validate("EQ-062", prior_before_cycle_1=0.3, posterior_after_cycle_3=0.7)
        assert r.passed and r.score == pytest.approx(0.4)

    def test_eq062_abba3_movement_fail(self) -> None:
        r = validate("EQ-062", prior_before_cycle_1=0.5, posterior_after_cycle_3=0.51)
        assert not r.passed

    def test_eq064_claim_density_suspicious(self) -> None:
        r = validate("EQ-064", claims_per_kb_text=8.0, observed_ratio=0.1)
        assert not r.passed and r.score == 1.0

    def test_eq064_claim_density_ok(self) -> None:
        r = validate("EQ-064", claims_per_kb_text=3.0, observed_ratio=0.8)
        assert r.passed


# ---------------------------------------------------------------------------
# §12. BRUTAL MATH
# ---------------------------------------------------------------------------

class TestBrutalMathEquations:
    def test_eq065_truth_infra_all_positive(self) -> None:
        r = validate("EQ-065", evidence_score=0.8, provenance_score=0.9,
                     countersearch_score=0.7, audit_score=0.95)
        assert r.passed
        assert r.score == pytest.approx(0.8 * 0.9 * 0.7 * 0.95)

    def test_eq065_truth_infra_zero_kills(self) -> None:
        r = validate("EQ-065", evidence_score=0.8, provenance_score=0.0,
                     countersearch_score=0.7, audit_score=0.95)
        assert not r.passed and r.score == 0.0

    def test_eq066_strategy_pass(self) -> None:
        r = validate("EQ-066", expected_value=5.0, evidence_strength=0.9, option_value=0.8,
                     risk=1.0, uncertainty=0.5, audit_debt=1.0)
        assert r.passed

    def test_eq067_pressure_pass(self) -> None:
        r = validate("EQ-067", lawful_vector=0.9, evidence_basis=0.8, human_approval=1.0, collateral_risk=0.1)
        assert r.passed and r.score == pytest.approx(0.9 * 0.8 * 1.0 - 0.1)

    def test_eq068_intelligence_all_positive(self) -> None:
        r = validate("EQ-068", hypothesis_update=0.3, contradiction_reduction=0.2, decision_improvement=0.1)
        assert r.passed and r.score == pytest.approx(0.6)

    def test_eq069_slop_zero(self) -> None:
        r = validate("EQ-069", claims_emitted=5, claims_with_evidence=5)
        assert r.passed and r.score == 0.0

    def test_eq069_slop_nonzero(self) -> None:
        r = validate("EQ-069", claims_emitted=10, claims_with_evidence=7)
        assert not r.passed and r.score == pytest.approx(3.0)

    def test_eq072_alpha_evolve_survives(self) -> None:
        r = validate("EQ-072", expected_value=5.0, risk=1.0, uncertainty=0.5, audit_debt=0.5, parent_fitness=2.0)
        assert r.passed and r.score == pytest.approx(3.0)

    def test_eq072_alpha_evolve_dies(self) -> None:
        r = validate("EQ-072", expected_value=1.0, risk=1.0, uncertainty=0.5, audit_debt=0.5, parent_fitness=2.0)
        assert not r.passed

    def test_eq073_sketch_accuracy_pass(self) -> None:
        r = validate("EQ-073", sketch_estimate=1010.0, true_value=1000.0)
        assert r.passed and r.score == pytest.approx(0.01)

    def test_eq074_near_dup_purity_pass(self) -> None:
        r = validate("EQ-074", same_source_count=9, cluster_size=10)
        assert r.passed and r.score == pytest.approx(0.9)

    def test_eq075_conformal_abstain_rate_pass(self) -> None:
        r = validate("EQ-075", abstained_calls=3, total_calls=10)
        assert r.passed and r.score == pytest.approx(0.3)

    def test_eq075_conformal_abstain_rate_fail(self) -> None:
        r = validate("EQ-075", abstained_calls=5, total_calls=10)
        assert not r.passed

    def test_eq078_evidence_reliability_sigmoid(self) -> None:
        r = validate("EQ-078", source_score=1.0, corroboration_score=1.0,
                     provenance_score=1.0, timestamp_score=1.0)
        assert r.passed
        assert 0 < r.score < 1  # sigmoid output

    def test_eq079_claim_score_positive(self) -> None:
        r = validate("EQ-079", r_evidence=0.9, corroboration=0.8, source_independence=0.7,
                     temporal_fit=0.8, incentive_fit=0.7,
                     contradiction_weight=0.0, provenance_gap=0.0, synthetic_risk=0.0)
        assert r.passed and r.score > 0

    def test_eq080_contradiction_penalty_low_count(self) -> None:
        r = validate("EQ-080", unresolved_weighted_count=0.0)
        assert r.passed and r.score == pytest.approx(1.0)

    def test_eq080_contradiction_penalty_high_count(self) -> None:
        r = validate("EQ-080", unresolved_weighted_count=5.0)
        assert not r.passed
        assert r.score == pytest.approx(math.exp(-0.5 * 5.0))

    def test_eq081_min_max_counterplay(self) -> None:
        r = validate("EQ-081", action_utilities={"file": [3.0, 2.0, 1.0], "wait": [5.0, 0.5, 4.0]})
        assert r.passed
        assert r.score == pytest.approx(1.0)  # file: min=1.0, wait: min=0.5 → best=file=1.0


# ---------------------------------------------------------------------------
# §13. OPERATOR LAWS
# ---------------------------------------------------------------------------

class TestOperatorLawEquations:
    def test_eq082_face_match_actionable_pass(self) -> None:
        r = validate("EQ-082", match_confidence=0.9, case_basis_strength=0.8, subject_class_allowed=True)
        assert r.passed

    def test_eq082_face_match_class_blocked(self) -> None:
        r = validate("EQ-082", match_confidence=0.9, case_basis_strength=0.8, subject_class_allowed=False)
        assert not r.passed and r.score == 0.0

    def test_eq083_osint_permitted(self) -> None:
        r = validate("EQ-083", subject_class_in_allowed=True, source_tier_permitted=True, case_id_present=True)
        assert r.passed

    def test_eq083_osint_blocked_no_case_id(self) -> None:
        r = validate("EQ-083", subject_class_in_allowed=True, source_tier_permitted=True, case_id_present=False)
        assert not r.passed

    def test_eq084_abductive_execute_kind(self) -> None:
        r = validate("EQ-084", resolution_kind="execute")
        assert r.passed

    def test_eq084_abductive_invalid_kind(self) -> None:
        r = validate("EQ-084", resolution_kind="vibes_decision")
        assert not r.passed

    def test_eq084_abductive_disprove_requires_cites(self) -> None:
        r = validate("EQ-084", resolution_kind="disprove_with_evidence",
                     cited_evidence_count=0, falsification_criteria_met=True)
        assert not r.passed

    def test_eq084_abductive_disprove_valid(self) -> None:
        r = validate("EQ-084", resolution_kind="disprove_with_evidence",
                     cited_evidence_count=3, falsification_criteria_met=True)
        assert r.passed

    def test_eq085_moralism_refusal_detected(self) -> None:
        r = validate("EQ-085", refusal_matches_vibes_pattern=True, cite_count=0, refusal_kind="PERSONAL_PREFERENCE")
        assert not r.passed and r.score == 1.0

    def test_eq085_legitimate_safety_refusal(self) -> None:
        r = validate("EQ-085", refusal_matches_vibes_pattern=True, cite_count=0, refusal_kind="SAFETY_GATE")
        assert r.passed

    def test_eq086_operator_trust_erosion_low(self) -> None:
        r = validate("EQ-086", directives_disproved_delta=0.1)
        assert r.passed

    def test_eq086_operator_trust_erosion_high(self) -> None:
        r = validate("EQ-086", directives_disproved_delta=0.3)
        assert not r.passed


# ---------------------------------------------------------------------------
# §14. META
# ---------------------------------------------------------------------------

class TestMetaEquations:
    def test_eq088_phase_gate_progress_pass(self) -> None:
        r = validate("EQ-088", closed_wos=8, total_wos=10)
        assert r.passed and r.score == pytest.approx(0.8)

    def test_eq088_phase_gate_progress_fail(self) -> None:
        r = validate("EQ-088", closed_wos=5, total_wos=10)
        assert not r.passed

    def test_eq090_failure_signature_velocity_pass(self) -> None:
        r = validate("EQ-090", delta_signature_observations=4.0, delta_time_hours=4.0)
        assert r.passed and r.score == pytest.approx(1.0)

    def test_eq090_failure_signature_velocity_fail(self) -> None:
        r = validate("EQ-090", delta_signature_observations=10.0, delta_time_hours=2.0)
        assert not r.passed

    def test_eq091_router_drift_cost_pass(self) -> None:
        r = validate("EQ-091", drift_scores=[0.02, 0.01], call_volume_shares=[0.6, 0.4])
        assert r.passed

    def test_eq091_mismatched_lengths_raise(self) -> None:
        with pytest.raises(ValueError):
            validate("EQ-091", drift_scores=[0.02], call_volume_shares=[0.6, 0.4])

    def test_eq092_ledger_chain_stability_perfect(self) -> None:
        r = validate("EQ-092", fork_attempts_detected=0, total_appends=1000)
        assert r.passed and r.score == 1.0

    def test_eq092_ledger_chain_stability_fail(self) -> None:
        r = validate("EQ-092", fork_attempts_detected=5, total_appends=1000)
        assert not r.passed

    def test_eq097_slop_volume_zero(self) -> None:
        r = validate("EQ-097", unsupported_claims=0, refusals_without_cite=0, unknown_displayed_as_ok=0)
        assert r.passed

    def test_eq097_slop_volume_nonzero(self) -> None:
        r = validate("EQ-097", unsupported_claims=2, refusals_without_cite=1, unknown_displayed_as_ok=0)
        assert not r.passed

    def test_eq098_silent_failure_zero(self) -> None:
        r = validate("EQ-098", failed_without_ledger_op=0)
        assert r.passed

    def test_eq099_false_green_zero(self) -> None:
        r = validate("EQ-099", unknown_displayed_as_ok_panels=0)
        assert r.passed

    def test_eq100_vibes_refusals_zero(self) -> None:
        r = validate("EQ-100", vibes_refusal_count=0)
        assert r.passed

    def test_eq100_vibes_refusals_nonzero(self) -> None:
        r = validate("EQ-100", vibes_refusal_count=3)
        assert not r.passed and r.score == 3.0


# ---------------------------------------------------------------------------
# §15. COMPOSITE KILLERS
# ---------------------------------------------------------------------------

class TestCompositeEquations:
    def _all_pass_kwargs(self) -> dict:
        return dict(
            packet_strength=0.8,
            case_complete_factors_min=0.95,
            retrieval_receipt_coverage=1.0,
            contradictions_resolved_ratio=0.98,
            human_approval_receipt_present=True,
            bleed_guard_pass=True,
            crypto_agile_provenance_valid=True,
        )

    def test_eq093_case_ship_ready_all_pass(self) -> None:
        r = validate("EQ-093", **self._all_pass_kwargs())
        assert r.passed and r.score == 1.0

    def test_eq093_case_ship_ready_one_fail(self) -> None:
        kwargs = self._all_pass_kwargs()
        kwargs["human_approval_receipt_present"] = False
        r = validate("EQ-093", **kwargs)
        assert not r.passed and r.score == 0.0

    def test_eq094_system_health_green(self) -> None:
        r = validate("EQ-094", ledger_integrity=True, chain_stable=True, quarantine_active=False,
                     test_floor_pass=True, burn_rate_ok=True, cockpit_honesty_ok=True,
                     no_evidence_backed_refusal_violation=True, bleed_guard_pass=True, consent_coverage_ok=True)
        assert r.passed and r.score == 1.0

    def test_eq094_system_health_quarantine_red(self) -> None:
        r = validate("EQ-094", ledger_integrity=True, chain_stable=True, quarantine_active=True,
                     test_floor_pass=True, burn_rate_ok=True, cockpit_honesty_ok=True,
                     no_evidence_backed_refusal_violation=True, bleed_guard_pass=True, consent_coverage_ok=True)
        assert not r.passed

    def test_eq095_export_approval_all_pass(self) -> None:
        r = validate("EQ-095", strategy_score_positive=True, pressure_not_flagged=True,
                     case_ship_ready=True, human_task_approved=True, adv_strength_passed=True,
                     hidden_info_coverage=0.8, audit_debt_zero=True)
        assert r.passed

    def test_eq095_export_blocked_no_human_approval(self) -> None:
        r = validate("EQ-095", strategy_score_positive=True, pressure_not_flagged=True,
                     case_ship_ready=True, human_task_approved=False, adv_strength_passed=True,
                     hidden_info_coverage=0.8, audit_debt_zero=True)
        assert not r.passed

    def test_eq096_directive_execute_path(self) -> None:
        r = validate("EQ-096", executed_and_ledgered=True)
        assert r.passed

    def test_eq096_directive_disprove_path(self) -> None:
        r = validate("EQ-096", disproved_with_evidence=True, disproved_cited_count=2,
                     disproved_falsification_met=True)
        assert r.passed

    def test_eq096_directive_unresolved(self) -> None:
        r = validate("EQ-096")
        assert not r.passed
