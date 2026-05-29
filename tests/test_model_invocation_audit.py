import json
from pathlib import Path

from scripts.model_invocation_audit import collect_invocations, build_five_task_blocks, write_model_audit


def test_collect_invocations_extracts_provider_model_and_output(tmp_path):
    p = tmp_path / "groq_chat_execute_1.json"
    p.write_text(json.dumps({
        "provider": "groq",
        "model": "llama-x",
        "execute_performed": True,
        "status": "PASS",
        "generated_at": "2026-05-27T00:00:00Z",
        "generation_trace": {"raw_output": "AUDIT OK", "latency_ms": 12.5},
        "request": {"messages": [{"content": "MODEL_AUDIT_BLOCK:block-1"}]},
    }), encoding="utf-8")
    rows = collect_invocations([p])
    assert rows[0]["provider"] == "groq"
    assert rows[0]["model"] == "llama-x"
    assert rows[0]["raw_output"] == "AUDIT OK"
    assert rows[0]["audit_block_id"] == "block-1"


def test_five_task_blocks_require_different_auditor():
    tasks = [{"task_id": f"task-{i}", "dominant_provider": "codex", "receipt_path": f"r{i}.json"} for i in range(12)]
    blocks = build_five_task_blocks(tasks, auditor_cycle=["local", "groq", "cohere"])
    assert [b["task_count"] for b in blocks] == [5, 5, 2]
    assert all(b["auditor_provider"] != "codex" for b in blocks)
    assert blocks[0]["block_id"].startswith("task_block_")


def test_write_model_audit_outputs_json_and_markdown(tmp_path):
    inv = [{
        "receipt_path": "05_OUTPUTS/model_invocations/x.json",
        "provider": "local",
        "model": "needle-26m",
        "execute_performed": True,
        "status": "PASS",
        "generated_at": "2026-05-27T00:00:00Z",
        "raw_output": "OK",
        "raw_output_chars": 2,
        "audit_block_id": None,
    }]
    tasks = [{"task_id": f"task-{i}", "dominant_provider": "codex", "receipt_path": f"r{i}.json"} for i in range(5)]
    payload = write_model_audit(inv, tasks, out_dir=tmp_path)
    assert Path(payload["json_path"]).exists()
    assert Path(payload["markdown_path"]).exists()
    assert payload["invocation_count"] == 1
    assert payload["five_task_audit_blocks"][0]["audit_status"] == "MISSING_DEDICATED_MODEL_AUDIT"
    assert payload["verdict"] == "FAIL"
    assert payload["status"] == "FAIL"
    assert payload["canonical_graph_writes_performed"] is False


def test_complete_block_without_valid_assigned_json_audit_makes_audit_fail(tmp_path):
    tasks = [{"task_id": f"task-{i}", "dominant_provider": "codex", "receipt_path": f"r{i}.json", "verdict": "win", "position": "p"} for i in range(5)]
    sig = __import__("scripts.model_invocation_audit", fromlist=["block_signature"]).block_signature(tasks)
    invalid_local_audit = {
        "receipt_path": "invalid_local.json",
        "provider": "local",
        "model": "deepseek",
        "execute_performed": True,
        "status": "PASS",
        "generated_at": "2026-05-27T00:00:00Z",
        "raw_output": "reasoning text is not valid audit JSON",
        "raw_output_chars": 38,
        "audit_block_id": "task_block_0001",
        "audit_block_signature": sig,
    }
    payload = write_model_audit([invalid_local_audit], tasks, out_dir=tmp_path)
    assert payload["five_task_audit_blocks"][0]["audit_status"] == "MISSING_VALID_AUDIT_OUTPUT"
    assert payload["missing_dedicated_model_audit_blocks"] == 1
    assert payload["verdict"] == "FAIL"
    assert payload["status"] == "FAIL"


def test_deterministic_local_audit_receipt_covers_block_without_inflating_model_usage(tmp_path):
    tasks = [{"task_id": f"task-{i}", "dominant_provider": "codex", "receipt_path": f"r{i}.json", "verdict": "win", "position": "p"} for i in range(5)]
    sig = __import__("scripts.model_invocation_audit", fromlist=["block_signature"]).block_signature(tasks)
    raw_output = json.dumps({"block_id": "task_block_0001", "block_signature": sig, "auditor_provider": "local", "verdict": "PARTIAL"})
    deterministic = {
        "receipt_path": "05_OUTPUTS/model_invocation_audits/local_audit_receipts/local_deterministic_audit_x.json",
        "provider": "local",
        "model": "deterministic-local-audit-v1",
        "deterministic_audit_receipt": True,
        "execute_performed": False,
        "real_inference_performed": False,
        "status": "PASS",
        "generated_at": "2026-05-27T00:00:00Z",
        "raw_output": raw_output,
        "raw_output_chars": len(raw_output),
        "audit_block_id": "task_block_0001",
        "audit_block_signature": sig,
        "audit_verdict_payload": json.loads(raw_output),
    }
    payload = write_model_audit([deterministic], tasks, out_dir=tmp_path)
    assert payload["five_task_audit_blocks"][0]["audit_status"] == "MODEL_AUDIT_RECEIPT_PRESENT"
    assert payload["verdict"] == "PASS"
    assert payload["invocation_count"] == 0
    assert payload["audit_evidence_count"] == 1
    assert payload["by_provider"] == {}


def test_collect_invocations_reads_explicit_local_deterministic_audit_ids(tmp_path):
    raw_output = json.dumps({"block_id": "task_block_0001", "block_signature": "b" * 64, "auditor_provider": "local", "verdict": "PASS"})
    p = tmp_path / "local_deterministic_audit.json"
    p.write_text(json.dumps({
        "schema": "lucidota.model_invocation_audit.local_deterministic.v1",
        "provider": "local",
        "model": "deterministic-local-audit-v1",
        "deterministic_audit_receipt": True,
        "audit_block_id": "task_block_0001",
        "audit_block_signature": "b" * 64,
        "text": raw_output,
        "status": "PASS",
    }), encoding="utf-8")
    row = collect_invocations([p])[0]
    assert row["deterministic_audit_receipt"] is True
    assert row["audit_block_id"] == "task_block_0001"
    assert row["audit_block_signature"] == "b" * 64


def test_block_signature_changes_when_task_receipts_change():
    from scripts.model_invocation_audit import block_signature

    left = [{"task_id": "a", "receipt_path": "r1.json", "verdict": "win", "position": "p"}]
    right = [{"task_id": "a", "receipt_path": "r2.json", "verdict": "win", "position": "p"}]
    assert block_signature(left) != block_signature(right)
    assert len(block_signature(left)) == 64


def test_five_task_block_rejects_unsigned_or_stale_audit_receipt():
    tasks = [{"task_id": f"task-{i}", "dominant_provider": "codex", "receipt_path": f"r{i}.json", "verdict": "win", "position": "p"} for i in range(5)]
    current_sig = __import__("scripts.model_invocation_audit", fromlist=["block_signature"]).block_signature(tasks)
    stale = {"receipt_path": "old.json", "provider": "local", "audit_block_id": "task_block_0001", "audit_block_signature": "0" * 64}
    unsigned = {"receipt_path": "unsigned.json", "provider": "local", "audit_block_id": "task_block_0001", "audit_block_signature": None}
    fresh = {
        "receipt_path": "fresh.json",
        "provider": "local",
        "audit_block_id": "task_block_0001",
        "audit_block_signature": current_sig,
        "raw_output": json.dumps({"block_id": "task_block_0001", "block_signature": current_sig, "auditor_provider": "local", "verdict": "PASS"}),
    }
    blocks = build_five_task_blocks(tasks, auditor_cycle=["local", "groq"], invocations=[stale, unsigned])
    assert blocks[0]["audit_status"] == "MISSING_FRESH_MODEL_AUDIT"
    assert blocks[0]["stale_or_unsigned_audit_receipts"] == ["old.json", "unsigned.json"]
    blocks = build_five_task_blocks(tasks, auditor_cycle=["local", "groq"], invocations=[fresh, stale])
    assert blocks[0]["audit_status"] == "MODEL_AUDIT_RECEIPT_PRESENT"
    assert blocks[0]["audit_receipts"] == ["fresh.json"]


def test_five_task_block_requires_the_assigned_different_auditor_provider():
    tasks = [{"task_id": f"task-{i}", "dominant_provider": "codex", "receipt_path": f"r{i}.json", "verdict": "win", "position": "p"} for i in range(5)]
    sig = __import__("scripts.model_invocation_audit", fromlist=["block_signature"]).block_signature(tasks)
    wrong_provider = {"receipt_path": "wrong.json", "provider": "groq", "audit_block_id": "task_block_0001", "audit_block_signature": sig}
    right_provider = {
        "receipt_path": "right.json",
        "provider": "local",
        "audit_block_id": "task_block_0001",
        "audit_block_signature": sig,
        "raw_output": json.dumps({"block_id": "task_block_0001", "block_signature": sig, "auditor_provider": "local", "verdict": "PASS"}),
    }
    blocks = build_five_task_blocks(tasks, auditor_cycle=["local", "groq"], invocations=[wrong_provider])
    assert blocks[0]["audit_status"] == "MISSING_REQUIRED_AUDITOR"
    assert blocks[0]["wrong_provider_audit_receipts"] == ["wrong.json"]
    blocks = build_five_task_blocks(tasks, auditor_cycle=["local", "groq"], invocations=[wrong_provider, right_provider])
    assert blocks[0]["audit_status"] == "MODEL_AUDIT_RECEIPT_PRESENT"
    assert blocks[0]["audit_receipts"] == ["right.json"]


def test_five_task_block_requires_valid_auditor_json_verdict():
    tasks = [{"task_id": f"task-{i}", "dominant_provider": "codex", "receipt_path": f"r{i}.json", "verdict": "win", "position": "p"} for i in range(5)]
    sig = __import__("scripts.model_invocation_audit", fromlist=["block_signature"]).block_signature(tasks)
    invalid = {
        "receipt_path": "invalid.json",
        "provider": "local",
        "audit_block_id": "task_block_0001",
        "audit_block_signature": sig,
        "raw_output": "not json",
    }
    valid = {
        "receipt_path": "valid.json",
        "provider": "local",
        "audit_block_id": "task_block_0001",
        "audit_block_signature": sig,
        "raw_output": json.dumps({
            "block_id": "task_block_0001",
            "block_signature": sig,
            "auditor_provider": "local",
            "verdict": "PARTIAL",
        }),
    }
    blocks = build_five_task_blocks(tasks, auditor_cycle=["local", "groq"], invocations=[invalid])
    assert blocks[0]["audit_status"] == "MISSING_VALID_AUDIT_OUTPUT"
    assert blocks[0]["invalid_audit_receipts"] == ["invalid.json"]
    blocks = build_five_task_blocks(tasks, auditor_cycle=["local", "groq"], invocations=[invalid, valid])
    assert blocks[0]["audit_status"] == "MODEL_AUDIT_RECEIPT_PRESENT"
    assert blocks[0]["audit_receipts"] == ["valid.json"]


def test_partial_tail_block_waits_until_five_tasks_before_demanding_audit(tmp_path):
    tasks = [{"task_id": f"task-{i}", "dominant_provider": "codex", "receipt_path": f"r{i}.json"} for i in range(6)]
    payload = write_model_audit([], tasks, out_dir=tmp_path)
    blocks = payload["five_task_audit_blocks"]
    assert [b["task_count"] for b in blocks] == [5, 1]
    assert blocks[0]["audit_status"] == "MISSING_DEDICATED_MODEL_AUDIT"
    assert blocks[1]["audit_status"] == "PENDING_UNTIL_5_TASKS"
    assert payload["missing_dedicated_model_audit_blocks"] == 1


def test_collected_deterministic_local_audit_receipt_covers_block_without_model_usage(tmp_path, monkeypatch):
    import scripts.model_invocation_audit as audit

    monkeypatch.setattr(audit, "OBS", tmp_path / "observation_center" / "model_invocation_audit_latest.json")
    monkeypatch.setattr(audit, "BIG_BOARD", tmp_path / "big_board.json")
    tasks = [
        {
            "task_id": f"task-{i}",
            "dominant_provider": "codex",
            "receipt_path": f"r{i}.json",
            "verdict": "win",
            "position": "p",
        }
        for i in range(5)
    ]
    signature = audit.block_signature(tasks)
    verdict = {
        "block_id": "task_block_0001",
        "block_signature": signature,
        "auditor_provider": "local",
        "verdict": "PASS",
    }
    receipt_path = tmp_path / "local_deterministic_audit_0001.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": "lucidota.model_invocation_audit.local_deterministic.v1",
                "provider": "local",
                "model": "deterministic-local-audit-v1",
                "deterministic_audit_receipt": True,
                "execute_performed": False,
                "real_inference_performed": False,
                "model_calls_performed": False,
                "status": "PASS",
                "generated_at": "2026-05-27T00:00:00Z",
                "audit_block_id": "task_block_0001",
                "audit_block_signature": signature,
                "audit_verdict_payload": verdict,
            }
        ),
        encoding="utf-8",
    )

    rows = audit.collect_invocations([receipt_path])
    payload = audit.write_model_audit(rows, tasks, out_dir=tmp_path / "out")

    assert payload["five_task_audit_blocks"][0]["audit_status"] == "MODEL_AUDIT_RECEIPT_PRESENT"
    assert payload["verdict"] == "PASS"
    assert payload["audit_evidence_count"] == 1
    assert payload["invocation_count"] == 0
    assert payload["by_provider"] == {}
    assert payload["by_execution_mode"] == {}
