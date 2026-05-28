from scripts import goal_scenario_batch


def test_goal_scenario_batch_builds_go25_receipt_and_rules():
    report = goal_scenario_batch.build_report(
        objective="Build the next version of the system by running scenario batches.",
        target="groq|cohere|local",
        scenario_count=12,
        batch_size=4,
        packet={"schema": "lucidota.worker_order.v1", "target": "local", "intent": "scenario_batch"},
    )

    assert report["schema"] == "lucidota.goal_scenario_batch.v1"
    assert report["ontology_mode"] == "GO25_STRICT"
    assert report["model_calls_performed"] is False
    assert report["canonical_graph_writes_performed"] is False
    assert report["scenario_count"] == 12
    assert len(report["scenario_batches"]) == 3
    assert len(report["decision_pairs"]) == 12
    assert report["decision_rule_candidates"]
    assert set(report["ontology_terms"]) == {"OBJECT", "EVENT", "EDGE"}
    assert all("receipt_path" not in batch or batch["scenario_count"] > 0 for batch in report["scenario_batches"])


def test_goal_scenario_batch_uses_current_handoff_when_objective_missing():
    report = goal_scenario_batch.build_report(
        objective="",
        target="local",
        scenario_count=6,
        batch_size=3,
        packet={"schema": "lucidota.worker_order.v1", "target": "local"},
    )

    assert report["objective"]
    assert report["family_counts"]
    assert report["evidence_refs"]


def test_goal_scenario_batch_supports_evidence_ingest_and_queue_integrity_families():
    evidence_text = goal_scenario_batch.scenario_text(
        "evidence_ingest",
        "Verify byte-perfect evidence ingress across three queues.",
        1,
        {"schema": "lucidota.worker_order.v1"},
    )
    queue_text = goal_scenario_batch.scenario_text(
        "queue_integrity",
        "Verify byte-perfect evidence ingress across three queues.",
        2,
        {"schema": "lucidota.worker_order.v1"},
    )

    assert "evidence" in evidence_text.lower()
    assert "embedding" in evidence_text.lower()
    assert "queue" in evidence_text.lower()
    assert "queue" in queue_text.lower()
    assert "lossless" in queue_text.lower() or "integrity" in queue_text.lower()


def test_goal_scenario_batch_can_prioritize_focus_from_compare_report():
    compare_report = {
        "scenario_focus": ["evidence_ingest", "queue_integrity", "ops"],
        "next_seed": ["OBJECT", "EVENT", "EDGE"],
    }

    families = goal_scenario_batch.focused_families_from_compare(compare_report)

    assert families[:2] == ["evidence_ingest", "queue_integrity"]
    assert "normal" in families
