#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_verified_case_runs_through_bitloops_bytewax_river_momentary_loop():
    from bitloops_automation_loop import orchestrate_loop, sha256_obj

    case = {
        "case_id": "krampuschewing-receipted-001",
        "source_ref": "scripts/krampuschewing_river_rows.py",
        "legacy_etl": "krampuschewing",
        "input": {"text": "Receipted KrampusChewing row should become River training evidence."},
    }
    case["receipt"] = {
        "input_sha256": sha256_obj(case["input"]),
        "output_sha256": "a" * 64,
        "transition_trace": [{"node": "legacy_etl", "to": "bitloops_context"}],
    }

    result = orchestrate_loop({"cases": [case], "loop_id": "test-loop"})

    assert result["status"] == "PASS"
    assert result["accepted_case_ids"] == ["krampuschewing-receipted-001"]
    assert result["quarantine"] == []
    assert result["canonical_graph_writes_performed"] is False
    assert result["state_collapsed"] is True
    assert result["model_lane_plan"]["policy"] == "local_first_cloud_optional_dry_by_default"
    assert result["model_lane_plan"]["lanes"][0]["lane_group"] == "local"

    flow = result["momentary_results"][0]
    assert flow["outcome"] == "succeeded"
    assert flow["state_collapsed"] is True
    channels = {example["channel"] for example in flow["training_examples"]}
    assert {"bitloops_context", "bytewax_hint", "river_mre", "ternary_truth"} <= channels
    assert result["river_training_lane_count"] == 4


def test_unreceipted_mappable_case_is_replayed_not_purged_or_deleted():
    from bitloops_automation_loop import orchestrate_loop

    result = orchestrate_loop(
        {
            "loop_id": "test-loop",
            "cases": [
                {
                    "case_id": "rickshaw-robbery-broken-hash",
                    "legacy_etl": "rickshaw_robbery",
                    "input": {"text": "Important case with broken trace must survive and be replayed."},
                    "receipt": {"input_sha256": "not-a-real-hash"},
                }
            ],
        }
    )

    assert result["status"] == "PASS"
    assert result["accepted_case_ids"] == ["rickshaw-robbery-broken-hash"]
    assert result["recovered_case_ids"] == ["rickshaw-robbery-broken-hash"]
    assert result["quarantine"] == []
    assert result["quarantine_failed"] == []
    assert result["canonical_graph_writes_performed"] is False
    assert result["graph_mutations"][0]["case_id"] == "rickshaw-robbery-broken-hash"
    assert result["graph_mutations"][0]["mutation_state"] == "candidate_requires_graph_promotion_gate"
    recovery = result["recovery_receipts"][0]
    assert recovery["receipt_origin"] == "deterministic_replay_current_logic"
    assert len(recovery["output_sha256"]) == 64
    assert recovery["transition_trace"][0]["node"] == "legacy_input"
    assert result["destroyed_case_count"] == 0
    assert result["purged_case_count"] == 0


def test_irrecoverable_schema_mismatch_is_reported_without_graph_write():
    from bitloops_automation_loop import orchestrate_loop

    result = orchestrate_loop({"loop_id": "bad-loop", "cases": [{"case_id": "bad-raw", "legacy_etl": "krampus_korpus"}]})

    assert result["status"] == "QUARANTINE_FAILED"
    assert result["accepted_case_ids"] == []
    assert result["graph_mutations"] == []
    assert result["quarantine_failed"] == [
        {
            "status": "QUARANTINE_FAILED",
            "reason": "irrecoverable_schema_mismatch",
            "raw_id": "bad-raw",
            "legacy_etl": "krampus_korpus",
        }
    ]
    assert result["canonical_graph_writes_performed"] is False


def test_cli_writes_receipt_for_dry_loop(tmp_path):
    case = {
        "case_id": "indy-reads-no-trace-yet",
        "legacy_etl": "indy_reads",
        "input": {"text": "No trace yet means quarantine, not graph truth."},
    }
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps({"loop_id": "cli-test", "cases": [case]}), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "scripts/bitloops_automation_loop.py", "--cases-file", str(cases_file), "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    receipt_line = next(line for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH="))
    receipt = ROOT / receipt_line.split("=", 1)[1]
    assert receipt.exists()
    data = json.loads(receipt.read_text(encoding="utf-8"))
    assert data["schema"] == "lucidota.bitloops.automation_loop.v1"
    assert data["status"] == "PASS"
    assert data["recovered_case_ids"] == ["indy-reads-no-trace-yet"]
    assert data["recovery_receipts"][0]["legacy_etl"] == "indy_reads"
    assert data["canonical_graph_writes_performed"] is False


def test_legacy_jsonl_rows_can_feed_recovery_cli(tmp_path):
    rows = [
        {
            "schema": "lucidota.krampuschewing.river_training_candidate.v1",
            "row_id": "krr_test_001",
            "source_path": "KRAMPUSCHEWING/example.txt",
            "source_sha256": "b" * 64,
            "lane": "FILE_ORGANIZATION",
            "truth_status": "training_candidate_only",
        }
    ]
    jsonl = tmp_path / "legacy.jsonl"
    jsonl.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/bitloops_automation_loop.py",
            "--legacy-jsonl",
            str(jsonl),
            "--legacy-etl",
            "krampuschewing",
            "--limit",
            "1",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    receipt_line = next(line for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH="))
    data = json.loads((ROOT / receipt_line.split("=", 1)[1]).read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["accepted_case_ids"] == ["krr_test_001"]
    assert data["recovered_case_ids"] == ["krr_test_001"]
    assert data["graph_mutations"][0]["legacy_etl"] == "krampuschewing"


def test_chrono_snapshot_rows_feed_ouroboros_recovery_cli(tmp_path):
    snapshot = tmp_path / "CHRONO_MASTER_SNAPSHOT_CURRENT.txt"
    snapshot.write_text(
        "header\n[RESOLVED_CHRONO_TIMELINE_JSONL]\n"
        "# one JSON object per row\n"
        + json.dumps({
            "file_uuid": "019e-test",
            "artifact_sha256": "c" * 64,
            "resolved_timestamp": "2026-02-18T16:00:00-08:00",
            "epistemic_flag": "PROBABLE",
            "provenance_path": "KRAMPUSCHEWING/example.jpg",
            "dominant_evidence_source": "filename_date",
            "snapshot_row_index": 1,
        })
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [sys.executable, "scripts/bitloops_automation_loop.py", "--chrono-snapshot", str(snapshot), "--limit", "1", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    receipt_line = next(line for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH="))
    data = json.loads((ROOT / receipt_line.split("=", 1)[1]).read_text(encoding="utf-8"))
    assert data["status"] == "PASS"
    assert data["accepted_case_ids"] == ["019e-test"]
    assert data["graph_mutations"][0]["legacy_etl"] == "chrono_master_snapshot"
    assert data["model_lane_plan"]["policy"] == "local_first_cloud_optional_dry_by_default"


def test_graph_promotion_packet_file_is_written_for_recovered_candidates(tmp_path):
    from bitloops_automation_loop import orchestrate_loop

    out = tmp_path / "promotion_packet.json"
    result = orchestrate_loop(
        {
            "loop_id": "packet-test",
            "cases": [
                {
                    "case_id": "chrono-row-001",
                    "legacy_etl": "chrono_master_snapshot",
                    "source_ref": "05_OUTPUTS/CHRONO_MASTER_SNAPSHOT_CURRENT.txt",
                    "input": {"text": "timeline row should become gated graph packet candidate"},
                }
            ],
        },
        write_graph_promotion_packet=True,
        graph_promotion_packet_path=out,
    )

    assert result["status"] == "PASS"
    assert result["compact_batch"]["accepted"] == 1
    assert result["compact_batch"]["graph_mutation_candidates"] == 1
    assert result["graph_promotion_packet_path"] == str(out)
    assert result["canonical_graph_writes_performed"] is False
    assert out.exists()

    packet = json.loads(out.read_text(encoding="utf-8"))
    assert packet["schema"] == "lucidota.bitloops.graph_promotion_packet_bundle.v1"
    assert packet["source_system"] == "bitloops_automation_loop"
    assert packet["canonical_graph_writes_performed"] is False
    assert packet["claim_packet_count"] == 1
    claim = packet["claim_packets"][0]
    assert claim["claim_packet_id"] == "chrono-row-001"
    assert claim["authority_class"] == "deterministic_metric"
    assert claim["review_state"] == "candidate_requires_graph_promotion_gate"
    assert claim["evidence_refs"]
    assert claim["candidate_payload"]["lanes"] == ["bitloops_context", "bytewax_hint", "river_training_lane", "ternary_truth"]
