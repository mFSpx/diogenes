import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/luci_ingest_route_bridge.py")


def test_ingest_route_bridge_emits_sheet_first_go_packets_and_tokio_absurd_refs(tmp_path):
    chunks = tmp_path / "chunks.jsonl"
    body_dir = tmp_path / "bodies"
    receipt = tmp_path / "bridge_receipt.json"
    rows = [
        {
            "schema": "lucidota.book_reader_lora.chunk.v1",
            "book_id": "root414",
            "chunk_ref": "root414.c0001",
            "text": "Evidence says the operator wants GO-25 packet routing. Because the risk is low, sheets should route before models.",
            "text_sha256": "a" * 64,
            "token_count": 19,
        },
        {
            "schema": "lucidota.book_reader_lora.chunk.v1",
            "book_id": "root414",
            "chunk_ref": "root414.c0002",
            "text": "A contradiction and danger flag should make ABSURD inspect the packet after Tokio pubsub receives only refs.",
            "text_sha256": "b" * 64,
            "token_count": 16,
            "algo_scores": {"danger_flag": True, "contradiction_count": 2},
        },
    ]
    chunks.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--chunks",
            str(chunks),
            "--receipt",
            str(receipt),
            "--body-dir",
            str(body_dir),
            "--max-chunks",
            "2",
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["schema"] == "lucidota.ingest_route_bridge.receipt.v1"
    assert data["status"] == "PASS"
    assert data["sheet_first"]["sheet"] == "next_work_batch"
    assert data["routing_manifest"]["lanes"]["fastlane"]["executor"] == "tokio_bounded_pubsub"
    assert data["packets_written"] == 2
    assert data["pubsub_events_written"] == 2
    assert data["absurd_events_written"] == 1

    packet_rows = [json.loads(line) for line in Path(data["go_packet_jsonl"]).read_text(encoding="utf-8").splitlines()]
    assert {p["ontology"] for p in packet_rows} == {"GO25", "GCI_O_75", "O414"}
    assert all("text" not in p for p in packet_rows)
    assert all(Path(p["body_path"]).exists() for p in packet_rows)

    events = [json.loads(line) for line in Path(data["tokio_pubsub_jsonl"]).read_text(encoding="utf-8").splitlines()]
    assert all(e["executor"] == "tokio_bounded_pubsub" for e in events)
    assert all("text" not in json.dumps(e) for e in events)
    assert any(e["route"] == "PANIC" for e in events)

    absurd = [json.loads(line) for line in Path(data["absurd_slowlane_jsonl"]).read_text(encoding="utf-8").splitlines()]
    assert absurd and absurd[0]["executor"] == "ABSURD_slowlane_receipt_gate"
    assert receipt.exists()
