from scripts import graph_symbol_dispatch


def test_graph_symbol_dispatch_builds_go25_packets():
    compare = {
        "report_path": "05_OUTPUTS/goals/graph_symbol_compare_test.json",
        "evidence_refs": ["r1", "r2"],
        "next_seed": ["OBJECT", "EVENT", "EDGE"],
        "stable_pairs": [{"pair": ["OBJECT", "EVENT"], "confidence_bps": 8000}],
    }
    report = graph_symbol_dispatch.build_report(compare, ["groq", "local"])

    assert report["schema"] == "lucidota.graph_symbol_dispatch.v1"
    assert report["ontology_mode"] == "GO25_STRICT"
    assert len(report["packets"]) == 2
    assert report["packets"][0]["schema"] == "lucidota.worker_order.v1"
    assert report["packets"][0]["ontology_terms"] == ["OBJECT", "EVENT", "EDGE"]
    assert report["packets"][0]["required_output"] == ["status", "result", "next_action", "receipt_path"]
    assert report["packets"][0]["output_contract"]["decision_pairs_min"] == 2
    assert report["packets"][0]["output_contract"]["no_prose"] is True


def test_graph_symbol_dispatch_write_report_adds_receipt_path(tmp_path):
    compare = {"report_path": "05_OUTPUTS/goals/graph_symbol_compare_test.json", "evidence_refs": []}
    report = graph_symbol_dispatch.write_report(graph_symbol_dispatch.build_report(compare, ["groq"]))
    assert report["receipt_path"].startswith("05_OUTPUTS/goals/")
