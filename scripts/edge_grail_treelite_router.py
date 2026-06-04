#!/usr/bin/env python3
"""Bounded Edge Grail Treelite route gate.

This is the practical switchyard between Bonsai/Mamba/Needle/algo packets and
FAST/CHECK/STREAM/DEEP/PANIC lanes. It moves refs/receipts, not bodies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ALGOS.runtime_caps import bounded_payload
from scripts.polycareer_treelite_gate import route as polycareer_route

LOADABILITY = ROOT / "05_OUTPUTS" / "model_runtime" / "treelite_loadability_latest.json"
INVENTORY = ROOT / "05_OUTPUTS" / "model_runtime" / "treelite_where_are_they_latest.json"
FIL_RESIDENCY = ROOT / "05_OUTPUTS" / "model_runtime" / "treelite_fil_residency_all_tl_latest.json"

ROUTES = {"FAST", "CHECK", "STREAM", "DEEP", "PANIC"}
BODY_KEYS = {"body", "text", "raw", "raw_text", "transcript", "prompt", "response"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def treelite_inventory_summary() -> dict[str, Any]:
    load = _read_json(LOADABILITY)
    inv = _read_json(INVENTORY)
    fil = _read_json(FIL_RESIDENCY)
    sizes = load.get("size_summary_bytes", {}) if isinstance(load, dict) else {}
    total_bytes = sum(int(v or 0) for v in sizes.values())
    fil_summary = fil.get("summary", {}) if isinstance(fil, dict) else {}
    fil_truth = fil.get("truth_flags", {}) if isinstance(fil, dict) else {}
    return {
        "treeliteish_total": int(inv.get("count", 0) or 0),
        "tl_deserialized": int((load.get("tl") or {}).get("deserialized", 0) or 0),
        "tl_total_mib": round(int(sizes.get("tl", 0) or 0) / 1024 / 1024, 3),
        "so_total_mib": round(int(sizes.get("shared_object", 0) or 0) / 1024 / 1024, 3),
        "total_mib": round(total_bytes / 1024 / 1024, 3),
        "loadability_receipt": str(LOADABILITY.relative_to(ROOT)),
        "fil_gpu_residency_proven": bool(fil_truth.get("fil_gpu_residency_proven")),
        "treelite_gpu_residency_proven": bool(fil_truth.get("treelite_gpu_residency_proven")),
        "fil_gpu_checked_tl": int(fil_summary.get("tested", 0) or 0),
        "fil_gpu_passed_tl": int(fil_summary.get("passed", 0) or 0),
        "fil_residency_receipt": str(FIL_RESIDENCY.relative_to(ROOT)) if FIL_RESIDENCY.exists() else "",
    }


def _safe_ref_packet(packet: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    ref = {k: v for k, v in packet.items() if k not in BODY_KEYS}
    raw, truncated = bounded_payload(packet)
    ref["packet_bytes_capped"] = len(raw)
    return ref, truncated


def _algo_scores(packet: dict[str, Any]) -> dict[str, Any]:
    scores = packet.get("algo_scores") or {}
    return scores if isinstance(scores, dict) else {}


def _route_from_policy(packet: dict[str, Any], treelite_lane: str) -> tuple[str, str, bool]:
    scores = _algo_scores(packet)
    danger = bool(scores.get("danger_flag") or packet.get("danger_flag"))
    contradictions = int(scores.get("contradiction_count") or packet.get("contradiction_count") or 0)
    operator_deep = bool(packet.get("operator_deep") or packet.get("needs_talkie"))
    bonsai_disagree = bool(packet.get("bonsai_disagree") or scores.get("bonsai_disagree"))

    if danger and contradictions >= 2:
        return "PANIC", "danger_with_contradictions", False
    if operator_deep:
        return "DEEP", "operator_or_policy_requested_talkie", True
    if treelite_lane in {"dead_letter", "audit"}:
        return "CHECK", f"treelite_lane_{treelite_lane}", False
    if danger or bonsai_disagree:
        return "CHECK", "danger_or_bonsai_disagreement", False
    if packet.get("stream_only"):
        return "STREAM", "stream_only", False
    if treelite_lane == "external":
        return "DEEP", "treelite_external_lane", True
    if treelite_lane == "fast":
        return "FAST", "treelite_fast_lane", False
    return "CHECK", f"treelite_lane_{treelite_lane}", False


def route_edge_packet(packet: dict[str, Any], artifact_path: str | None = None) -> dict[str, Any]:
    ref_packet, truncated = _safe_ref_packet(packet)
    payload_for_gate = {
        "mutation": bool(packet.get("mutation")),
        "source_type": str(packet.get("source_type") or ""),
        "danger_flag": bool((_algo_scores(packet)).get("danger_flag") or packet.get("danger_flag")),
    }
    treelite_result = polycareer_route(artifact_path, payload_for_gate)
    route, reason, talkie_allowed = _route_from_policy(packet, treelite_result["lane"])
    summary = treelite_inventory_summary()
    return {
        "schema": "lucidota.edge_grail.treelite_route.v1",
        "event_id": str(packet.get("event_id") or ""),
        "input_hash": str(packet.get("input_hash") or packet.get("content_hash") or ""),
        "body_path": str(packet.get("body_path") or packet.get("path") or ""),
        "route": route,
        "reason": reason,
        "talkie_allowed": talkie_allowed,
        "packet_truncated": truncated,
        "packet_bytes_capped": ref_packet["packet_bytes_capped"],
        "polycareer_gate": treelite_result,
        "treelite": summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="edge-grail-treelite-router")
    ap.add_argument("--packet", default="{}", help="bounded JSON packet; bodies are not returned")
    ap.add_argument("--artifact", default=None)
    args = ap.parse_args()
    packet = json.loads(args.packet)
    print(json.dumps(route_edge_packet(packet, artifact_path=args.artifact), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
