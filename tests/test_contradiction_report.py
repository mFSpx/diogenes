import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from contradiction_report import detect_contradictions


def _claim(claim_id: str, text: str) -> dict:
    return {"claim_id": claim_id, "claim_text": text, "source_ref": {"id": claim_id}, "entity_candidates": []}


def test_detect_contradictions_keeps_exact_negation_semantics():
    rows = [
        _claim("a", "The notice was served."),
        _claim("b", "The notice was not served."),
        _claim("c", "Different fact."),
    ]

    out = detect_contradictions(rows)

    assert len(out) == 1
    assert out[0]["claim_ids"] == ["a", "b"]
    assert out[0]["resolution_status"] == "OPEN"


def test_detect_contradictions_is_bounded_for_large_duplicate_groups():
    rows = [_claim(f"p{i}", "The notice was served.") for i in range(20)]
    rows += [_claim(f"n{i}", "The notice was not served.") for i in range(20)]

    out = detect_contradictions(rows, max_reports=7, max_pairs_per_base=50)

    assert len(out) == 7
