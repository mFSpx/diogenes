from __future__ import annotations

from scripts import luci_response_composer as composer


def test_compose_response_orders_fast_template_map_math_quotes_improve_slow_review():
    context = {
        "text": "make my system work and get my shit ingested",
        "intent": "ops",
        "lane": "FASTLANE",
        "text_chars": 43,
        "word_count": 9,
        "ontology_terms": ["TIME", "EVENT", "TOOL", "MODE"],
        "work_order_id": "wo:test-1",
        "attempt_id": "att:test-1",
        "receipt_path": "05_OUTPUTS/luci/test.json",
        "language_rendered": "INTENT=ops TERMS=['TIME', 'EVENT'] LANE=FASTLANE TASK=ops: make my system work and get my shit ingested",
        "route_reason": ["default_fast_metadata_gate"],
        "indy_corpus_units": [
            {"label": "identity", "text": "Indy_READs is the co-operator surface.", "source": "a.md#identity"},
            {"label": "doctrine", "text": "Evidence first, receipts always.", "source": "a.md#doctrine"},
            {"label": "authority", "text": "System should own composition.", "source": "a.md#authority"},
        ],
        "recent_self_receipts": [
            {"label": "attempt", "status": "PASS", "note": "attempt engine wrote the ledger cleanly."},
            {"label": "learning", "status": "PASS", "note": "learning loop promoted a reusable candidate."},
        ],
        "external_review": {
            "groq": {"status": "PASS", "findings": ["keep it bounded"], "next_steps": ["compose lanes"], "blockers": []},
            "vibes": {"status": "PASS", "findings": ["sequence fast first"], "next_steps": ["deeper later"], "blockers": []},
        },
        "adhd_slow_lane": {
            "synthesis": "Resource management, learning, precognition, and hyperplexing belong in the slow lane.",
            "survivors": [{"frame": "evidence", "text": "Evidence first; then deeper synthesis."}],
        },
    }

    out = composer.compose_response(context)

    assert out["schema"] == "lucidota.luci.response_composer.v1"
    lanes = [segment["lane"] for segment in out["segments"]]
    assert lanes[:7] == ["fast", "template", "map", "math", "quotes", "improve", "slow"]
    assert lanes[-1] == "review"
    assert out["visible_response"]["summary"].startswith("Indy_READs:")
    assert "map:" in out["visible_response"]["summary"].lower()
    assert "math:" in out["visible_response"]["summary"].lower()
    assert "quote:" in out["visible_response"]["summary"].lower()
    assert "improve:" in out["visible_response"]["summary"].lower()
    assert "review:" in out["visible_response"]["summary"].lower()


def test_compose_response_uses_receipt_safe_json_and_next_hint():
    out = composer.compose_response({
        "text": "route this through the composer",
        "intent": "ops",
        "lane": "FASTLANE",
        "text_chars": 31,
        "word_count": 5,
        "ontology_terms": [],
        "work_order_id": "wo:test-2",
        "attempt_id": "att:test-2",
        "receipt_path": "05_OUTPUTS/luci/test.json",
        "language_rendered": "INTENT=ops",
    })

    assert out["visible_response"]["next"] in {"fast route completed", "receipt written; slow work queued"}
    assert out["composition"]["json_safe"] is True
    assert out["composition"]["lane_count"] >= 5
    assert out["composition"]["has_improve_lane"] is True
    assert out["composition"]["has_map_lane"] is True
