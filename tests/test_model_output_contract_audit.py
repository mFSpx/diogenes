from scripts import model_output_contract_audit


def test_normalize_receipt_maps_decisions_alias_into_decision_pairs():
    receipt = {
        "status": "SUCCESS",
        "result": "OK",
        "next_action": "keep going",
        "receipt_path": "05_OUTPUTS/model_invocations/local.json",
        "evidence_refs": ["a", "b"],
        "decisions": [
            {"feature": "prompt_shape", "condition": "boilerplate", "expected": "short", "observed": "short"},
        ],
    }

    normalized = model_output_contract_audit.normalize_receipt(receipt)

    assert normalized["decision_pairs"] == receipt["decisions"]
    assert normalized["field_aliases"] == {"decisions": "decision_pairs"}
    assert normalized["missing_required_fields"] == []


def test_build_report_summarizes_alias_drift_across_receipts(tmp_path):
    groq = tmp_path / "groq.json"
    local = tmp_path / "local.json"
    groq.write_text(
        '{"status":"success","result":"delta_report","next_action":"review","receipt_path":"groq.r","evidence_refs":["a"],"decision_pairs":[{"feature":"x"}]}'
    )
    local.write_text(
        '{"status":"SUCCESS","result":"delta_report","next_action":"review","receipt_path":"local.r","evidence_refs":["b"],"decisions":[{"feature":"y"}]}'
    )

    report = model_output_contract_audit.build_report([str(groq), str(local)])

    assert report["receipt_count"] == 2
    assert report["aliases"] == ["decisions->decision_pairs"]
    assert report["missing_required_fields"] == []
    assert report["normalized_receipts"][1]["decision_pairs"] == [{"feature": "y"}]


def test_normalize_receipt_parses_embedded_json_text():
    receipt = {
        "text": "```json\n{\"status\":\"success\",\"result\":\"done\",\"next_action\":\"stop\",\"receipt_path\":\"r.json\",\"evidence_refs\":[\"x\"],\"decisions\":[{\"feature\":\"prompt\"}]}\n```"
    }

    normalized = model_output_contract_audit.normalize_receipt(receipt)

    assert normalized["status"] == "success"
    assert normalized["result"] == "done"
    assert normalized["decision_pairs"] == [{"feature": "prompt"}]


def test_normalize_receipt_recovers_scalar_fields_from_truncated_json_text():
    receipt = {
        "text": "```json\n{\"status\":\"success\",\"result\":\"delta_report\",\"next_action\":\"review_minimal_changes\",\"receipt_path\":\"05_OUTPUTS/model_invocations/x.json\",\"evidence_refs\":[\"a\"],\"decision_pairs\":[{\"feature\":\"x\"}]"
    }

    normalized = model_output_contract_audit.normalize_receipt(receipt)

    assert normalized["status"] == "success"
    assert normalized["result"] == "delta_report"
    assert normalized["next_action"] == "review_minimal_changes"
    assert normalized["receipt_path"] == "05_OUTPUTS/model_invocations/x.json"


def test_normalize_receipt_flattens_nested_result_object():
    receipt = {
        "text": "```json\n{\"status\":\"success\",\"result\":{\"next_action\":\"review\",\"decision_pairs\":[{\"feature\":\"x\"}],\"receipt_path\":\"r.json\",\"evidence_refs\":[\"e1\"]},\"receipt_path\":\"outer.json\"}\n```"
    }

    normalized = model_output_contract_audit.normalize_receipt(receipt)

    assert normalized["status"] == "success"
    assert normalized["next_action"] == "review"
    assert normalized["receipt_path"] == "r.json"
    assert normalized["evidence_refs"] == ["e1"]
    assert normalized["decision_pairs"] == [{"feature": "x"}]
