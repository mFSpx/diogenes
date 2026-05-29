import json
from pathlib import Path

from scripts.dev_journey_decision_points import (
    compile_decision_points,
    extract_decision_points_from_text,
    label_decision_point,
    train_tree_artifacts,
)


def test_imagining_maps_to_explore_workflow_pressure():
    text = "If you hear me imagining, parse that as pushing for different ways to do it; try new workflows."
    label = label_decision_point(text)
    assert label["route_class"] == "EXPLORE_WORKFLOW"
    assert label["workflow_exploration_pressure"] == 1.0
    assert label["truth_status"] == "deterministic_label_candidate_only"


def test_decision_points_get_stickers_and_board_effects(tmp_path):
    source = tmp_path / "GOAL_LOG.md"
    source.write_text(
        "Build the graph with receipts.\n\nImagine a different workflow and throw spaghetti at invisible walls.\n",
        encoding="utf-8",
    )
    points = compile_decision_points([source], max_points=10)
    assert len(points) == 2
    assert all("sticker_features" in point for point in points)
    assert any(point["labels"]["route_class"] == "EXPLORE_WORKFLOW" for point in points)
    assert any("graph" in point["board_effects"] for point in points)


def test_extract_decision_points_is_bounded_and_stable():
    rows = extract_decision_points_from_text("A.\n\nB.\n\nC.", source_path="x.md", max_points=2)
    assert [row["ordinal"] for row in rows] == [1, 2]
    assert rows[0]["source_path"] == "x.md"


def test_train_tree_artifacts_writes_xgboost_and_treelite(tmp_path):
    source = tmp_path / "journey.md"
    source.write_text(
        "Imagine new workflow.\n\nAudit slop with receipts and graph staging.\n\nBuild fixed code and run tests.\n\nGroq Cohere local models route.\n",
        encoding="utf-8",
    )
    points = compile_decision_points([source], max_points=10)
    out = train_tree_artifacts(points, out_dir=tmp_path)
    assert out["training_performed"] is True
    assert Path(out["xgboost_model_path"]).exists()
    assert Path(out["treelite_model_path"]).exists()
    assert out["row_count"] == len(points)
    assert out["smoke_prediction_count"] == len(points)


def test_receipt_shape_from_cli_dry_run(tmp_path):
    source = tmp_path / "status.md"
    source.write_text("Graph receipts.\n\nImagine new flow.", encoding="utf-8")
    points = compile_decision_points([source], max_points=10)
    assert all(point["canonical_graph_writes_performed"] is False for point in points)
    assert all(point["model_authority"] == "advisory_training_candidate" for point in points)


def test_compile_decision_points_round_robins_sources(tmp_path):
    a = tmp_path / "GOAL_LOG.md"
    b = tmp_path / "STATUS_LEDGER.md"
    a.write_text("A1 graph receipts.\n\nA2 graph receipts.\n\nA3 graph receipts.", encoding="utf-8")
    b.write_text("B1 hunch audit.\n\nB2 hunch audit.\n\nB3 hunch audit.", encoding="utf-8")
    points = compile_decision_points([a, b], max_points=4)
    assert [Path(p["source_path"]).name for p in points] == ["GOAL_LOG.md", "STATUS_LEDGER.md", "GOAL_LOG.md", "STATUS_LEDGER.md"]
    assert {p["source_kind"] for p in points} == {"goal_log", "status_ledger"}


def test_default_sources_include_high_signal_non_goal_sources():
    from scripts.dev_journey_decision_points import default_sources

    names = [str(p) for p in default_sources()]
    assert any("SCRIPT_AUDIT_MANIFEST.jsonl" in name for name in names)
    assert any("krampuschewing_normalized_index" in name for name in names)
    assert any("STATUS_LEDGER.md" in name for name in names)
