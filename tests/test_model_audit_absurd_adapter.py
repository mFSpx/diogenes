def test_model_audit_adapter_separates_deterministic_audit_receipts():
    from scripts.model_audit_absurd_adapter import build_rows

    rows = build_rows({
        "by_provider": {"local": 1},
        "invocations": [
            {"provider": "local", "model": "deepseek", "status": "PASS", "receipt_path": "model.json"},
            {"provider": "local", "model": "deterministic-local-audit-v1", "status": "PASS", "receipt_path": "det.json", "deterministic_audit_receipt": True},
        ],
        "five_task_audit_blocks": [],
    }, __import__("pathlib").Path("audit.json"))
    assert len(rows["model_invocation"]) == 1
    assert rows["model_invocation"][0]["receipt_ref"] == "model.json"
    assert rows["deterministic_audit_receipt"][0]["receipt_ref"] == "det.json"
