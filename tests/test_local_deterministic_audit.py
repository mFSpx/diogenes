import json


def test_build_local_deterministic_audit_receipt_is_signed_and_non_model():
    from scripts.local_deterministic_audit import build_receipt

    receipt = build_receipt(
        block={"block_id": "task_block_0001", "block_signature": "a" * 64, "task_receipts": ["r1.json"]},
        verdict="PARTIAL",
        findings=["bounded deterministic audit"],
    )
    assert receipt["schema"] == "lucidota.model_invocation_audit.local_deterministic.v1"
    assert receipt["provider"] == "local"
    assert receipt["deterministic_audit_receipt"] is True
    assert receipt["model_calls_performed"] is False
    assert receipt["real_inference_performed"] is False
    parsed = json.loads(receipt["text"])
    assert parsed["block_id"] == "task_block_0001"
    assert parsed["block_signature"] == "a" * 64
    assert parsed["auditor_provider"] == "local"
    assert parsed["verdict"] == "PARTIAL"
