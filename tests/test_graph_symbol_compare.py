import json
from pathlib import Path

from scripts import graph_symbol_compare


def _write_report(path: Path, claims: list[dict]):
    payload = {
        "schema": "lucidota.graph_symbol_condensation.v1",
        "report_path": str(path),
        "receipt_path": str(path),
        "claims": claims,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def test_graph_symbol_compare_emits_next_seed_and_deltas(tmp_path):
    baseline = _write_report(
        tmp_path / "baseline.json",
        [
            {"pair": ["OBJECT", "EVENT"], "confidence_bps": 5000, "status": "candidate", "evidence_refs": ["a"]},
            {"pair": ["EVENT", "EDGE"], "confidence_bps": 7000, "status": "backed", "evidence_refs": ["b"]},
        ],
    )
    current = _write_report(
        tmp_path / "current.json",
        [
            {"pair": ["OBJECT", "EVENT"], "confidence_bps": 8200, "status": "backed", "evidence_refs": ["a"]},
            {"pair": ["EDGE", "CLAIM"], "confidence_bps": 6200, "status": "candidate", "evidence_refs": ["c"]},
        ],
    )

    report = graph_symbol_compare.compare_reports(current, baseline)

    assert report["schema"] == "lucidota.graph_symbol_compare.v1"
    assert report["ontology_mode"] == "GO25_STRICT"
    assert report["comparison_summary"]["improved"] == 1
    assert report["comparison_summary"]["lost"] == 1
    assert report["next_seed"]
    assert report["next_seed"][0] in {"OBJECT", "EVENT", "EDGE"}


def test_graph_symbol_compare_write_report_adds_receipt_path(tmp_path):
    baseline = _write_report(tmp_path / "baseline.json", [])
    current = _write_report(tmp_path / "current.json", [])
    report = graph_symbol_compare.write_report(graph_symbol_compare.compare_reports(current, baseline))
    assert report["receipt_path"].startswith("05_OUTPUTS/goals/")
