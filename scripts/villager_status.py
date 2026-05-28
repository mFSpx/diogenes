#!/usr/bin/env python3
"""Report Percyphon/villager status from receipts without model calls or graph writes."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "village"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str, root: Path = ROOT) -> str:
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _slot_from_receipt(path: Path, data: dict[str, Any], root: Path) -> dict[str, Any] | None:
    percyphon = data.get("percyphon") if isinstance(data.get("percyphon"), dict) else {}
    slot = percyphon.get("slot") if isinstance(percyphon.get("slot"), dict) else {}
    packet = data.get("control_packet") if isinstance(data.get("control_packet"), dict) else {}
    payload = packet.get("payload") if isinstance(packet.get("payload"), dict) else {}
    packet_slot = payload.get("percyphon_slot") if isinstance(payload.get("percyphon_slot"), dict) else {}
    merged = {**packet_slot, **slot}
    if not merged:
        return None
    return {
        "name": merged.get("name"),
        "uuid": merged.get("uuid"),
        "alias": merged.get("alias"),
        "persona": merged.get("persona"),
        "slot_index": merged.get("slot_index"),
        "ternary_offset": merged.get("ternary_offset"),
        "status": data.get("status"),
        "source": data.get("source"),
        "authority_class": data.get("authority_class"),
        "authority": percyphon.get("authority"),
        "zero_vram": percyphon.get("zero_vram"),
        "generated_at": data.get("generated_at"),
        "raw_command": data.get("raw_command"),
        "blockers": data.get("blockers") or [],
        "canonical_graph_writes_performed": bool(data.get("canonical_graph_writes_performed")),
        "model_calls_performed": bool(data.get("model_calls_performed")),
        "receipt_path": rel(path, root),
    }


def load_percyphon_slots(root: Path) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for path in sorted((root / "05_OUTPUTS" / "diogenes").glob("percyphon_kernel_bridge_*.json")):
        data = read_json(path)
        if not data:
            continue
        slot = _slot_from_receipt(path, data, root)
        if slot:
            slots.append(slot)
    return slots


def _heartbeat_status(line_data: dict[str, Any], path: Path, line_no: int, root: Path) -> dict[str, Any]:
    status = line_data.get("status") if isinstance(line_data.get("status"), dict) else {}
    weights = status.get("weights") if isinstance(status.get("weights"), dict) else {}
    return {
        "path": rel(path, root),
        "line": line_no,
        "created_at": line_data.get("created_at"),
        "cycle": line_data.get("cycle"),
        "engine_channel": line_data.get("engine_channel"),
        "backend": status.get("backend"),
        "mode": status.get("mode"),
        "kernel_loaded": (status.get("kernel") or {}).get("loaded") if isinstance(status.get("kernel"), dict) else None,
        "weights_mapped": weights.get("mapped"),
        "weights_status": weights.get("status"),
        "load_errors": status.get("load_errors") or [],
    }


def load_fairyfuse_cycle_matches(root: Path, targets: list[str]) -> dict[str, list[dict[str, Any]]]:
    matches = {target: [] for target in targets}
    path = root / "04_RUNTIME" / "fairyfuse" / "ternary_router_heartbeat.jsonl"
    if not path.exists():
        return matches
    target_set = set(targets)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                data = json.loads(line)
            except Exception:
                continue
            cycle = data.get("cycle")
            if str(cycle) in target_set:
                matches[str(cycle)].append(_heartbeat_status(data, path, line_no, root))
    return matches


def villager_matches(slots: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    target_norm = target.strip()
    wanted_name = f"Villager-{target_norm.zfill(4)}" if target_norm.isdigit() else target_norm
    for slot in slots:
        reasons: list[str] = []
        name = str(slot.get("name") or "")
        uuid = str(slot.get("uuid") or "")
        if uuid == target_norm:
            reasons.append("slot_uuid_exact")
        elif target_norm and target_norm in uuid:
            reasons.append("slot_uuid_contains")
        if name == wanted_name or name == target_norm:
            reasons.append("slot_name_suffix")
        for reason in reasons:
            item = dict(slot)
            item["match_reason"] = reason
            out.append(item)
    return out


def unique_routed_slots(slots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_uuid: dict[str, dict[str, Any]] = {}
    for slot in slots:
        key = str(slot.get("uuid") or slot.get("name") or slot.get("receipt_path"))
        if slot.get("status") == "ROUTED" or key not in by_uuid:
            by_uuid[key] = slot
    return list(by_uuid.values())


def build_status(root: Path = ROOT, targets: list[str] | None = None) -> dict[str, Any]:
    targets = [str(t) for t in (targets or [])]
    slots = load_percyphon_slots(root)
    routed = [s for s in unique_routed_slots(slots) if s.get("status") == "ROUTED"]
    denied = [s for s in slots if s.get("status") == "DENIED"]
    cycles = load_fairyfuse_cycle_matches(root, targets)
    target_report: dict[str, Any] = {}
    for target in targets:
        vm = villager_matches(slots, target)
        target_report[target] = {
            "status": "VILLAGER_RECORD_FOUND" if vm else "NO_VERIFIED_VILLAGER_RECORD",
            "villager_matches": vm,
            "fairyfuse_cycle_matches": cycles.get(target, []),
            "note": "FairyFuse cycle matches are ternary-router heartbeat cycles, not villager identity records.",
        }
    return {
        "schema": "lucidota.village.villager_status.v1",
        "generated_at": now(),
        "targets": target_report,
        "village": {
            "source": "Percyphon procedural scaffold receipts",
            "receipt_count": len(slots),
            "routed_count": len(routed),
            "denied_count": len(denied),
            "slots": routed,
            "denied_slots": denied,
            "authority_note": "procedural_scaffold_candidate_not_truth; not canonical people/truth by itself",
        },
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
    }


def write_report(report: dict[str, Any], root: Path = ROOT) -> Path:
    out = root / "05_OUTPUTS" / "village"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"villager_status_{stamp()}.json"
    report["report_path"] = rel(path, root)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Show Percyphon/Village villager status from receipts.")
    ap.add_argument("uuid", nargs="*", help="Full UUID, UUID fragment, or Villager-#### suffix to inspect")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = build_status(ROOT, args.uuid)
    path = write_report(report, ROOT)
    print("REPORT_PATH=" + rel(path, ROOT))
    print("VILLAGER_STATUS=PASS")
    for target, item in report["targets"].items():
        print(f"TARGET={target} STATUS={item['status']} VILLAGER_MATCHES={len(item['villager_matches'])} FAIRYFUSE_CYCLES={len(item['fairyfuse_cycle_matches'])}")
    if args.json:
        print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
