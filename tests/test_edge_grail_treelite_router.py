from scripts.edge_grail_treelite_router import route_edge_packet, treelite_inventory_summary


def test_treelite_inventory_summary_proves_tiny_loadable_gate_set():
    summary = treelite_inventory_summary()
    assert summary["treeliteish_total"] >= 301
    assert summary["tl_deserialized"] == 103
    assert summary["tl_total_mib"] < 3
    assert summary["total_mib"] < 15
    assert summary["fil_gpu_residency_proven"] is True
    assert summary["fil_gpu_checked_tl"] == 103
    assert summary["fil_gpu_passed_tl"] == 103


def test_route_edge_packet_moves_refs_not_bodies_and_can_choose_deep():
    packet = {
        "event_id": "evt-1",
        "body": "x" * 100_000,
        "body_path": "04_RUNTIME/example_body.txt",
        "input_hash": "a" * 64,
        "operator_deep": True,
        "algo_scores": {"danger_flag": False, "contradiction_count": 0},
    }
    result = route_edge_packet(packet)
    assert result["route"] == "DEEP"
    assert result["talkie_allowed"] is True
    assert "body" not in result
    assert result["input_hash"] == "a" * 64
    assert result["packet_truncated"] is True
    assert result["treelite"]["tl_deserialized"] == 103


def test_route_edge_packet_sends_danger_contradictions_to_panic():
    packet = {
        "event_id": "evt-2",
        "input_hash": "b" * 64,
        "algo_scores": {"danger_flag": True, "contradiction_count": 3},
    }
    result = route_edge_packet(packet)
    assert result["route"] == "PANIC"
    assert result["talkie_allowed"] is False
    assert result["reason"] == "danger_with_contradictions"

import json
import subprocess
import sys


def test_edge_grail_treelite_router_cli_runs_from_repo_root():
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/edge_grail_treelite_router.py",
            "--packet",
            '{"event_id":"cli","input_hash":"abc","operator_deep":true}',
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["route"] == "DEEP"
