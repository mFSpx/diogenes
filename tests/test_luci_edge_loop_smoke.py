import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/luci_edge_loop_smoke.py")


def test_edge_loop_smoke_proves_sheet_admission_route_indy_speed(tmp_path):
    receipt = tmp_path / "edge_loop.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--receipt",
            str(receipt),
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
    assert data["schema"] == "lucidota.edge_loop_smoke_receipt.v1"
    assert data["status"] == "PASS"
    assert data["body_policy"] == "refs_not_bodies"
    assert data["steps"]["sheet_list"]["status"] == "PASS"
    assert data["steps"]["aux_admission"]["admit"] is True
    assert data["steps"]["routing_manifest"]["lanes"]["fastlane"]["executor"] == "tokio_bounded_pubsub"
    assert "ABSURD" in data["steps"]["routing_manifest"]["lanes"]["slowlane"]["escalates_to"]
    assert data["steps"]["treelite_route"]["route"] in {"FAST", "CHECK", "STREAM", "DEEP", "PANIC"}
    body_path = Path(data["steps"]["treelite_route"]["body_path"])
    assert body_path.exists()
    assert body_path.read_text(encoding="utf-8").strip()
    assert data["steps"]["indy_comms"]["send_requires_operator_approval"] is True
    assert data["steps"]["speed_probe"]["status"] == "PASS"
    assert receipt.exists()
    saved = json.loads(receipt.read_text(encoding="utf-8"))
    assert saved["output_hash"] == data["output_hash"]
