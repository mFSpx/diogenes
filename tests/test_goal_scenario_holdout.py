from scripts import goal_scenario_batch
import sys


def test_build_holdout_report_scores_holdout_and_promotes_stable_rules():
    report = goal_scenario_batch.build_holdout_report(
        objective="Build the next version of the system by running cheap deterministic scenario passes.",
        target="groq|cohere|local",
        scenario_count=12,
        batch_size=4,
        holdout_stride=3,
        packet={"schema": "lucidota.worker_order.v1", "target": "local", "intent": "scenario_batch"},
    )

    assert report["schema"] == "lucidota.goal_scenario_holdout.v1"
    assert report["ontology_mode"] == "GO25_STRICT"
    assert report["training_count"] == 8
    assert report["holdout_count"] == 4
    assert report["holdout_accuracy"] >= 0.0
    assert report["decision_tree_candidates"]
    assert report["promoted_rules"]
    assert report["split_decision"]["reason"]


def test_cli_can_emit_holdout_report(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "goal_scenario_batch.py",
            "--scenario-count",
            "6",
            "--batch-size",
            "3",
            "--holdout-stride",
            "2",
            "--json",
        ],
    )

    rc = goal_scenario_batch.main()
    out = capsys.readouterr().out

    assert rc == 0
    assert "GOAL_SCENARIO_HOLDOUT=PASS" in out
    assert "lucidota.goal_scenario_holdout.v1" in out
