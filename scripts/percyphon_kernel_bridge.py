#!/usr/bin/env python3
"""Route a PercyphonAI procedural scaffold through the Diogenes control-packet gate."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from ALGOS.percyphon import procedural_entity_generator  # noqa: E402
from scripts.ckdog_kernel_route_plan import route_plan_from_packet  # noqa: E402
from scripts.kernel_control_packet import make_control_packet  # noqa: E402
from scripts.spine_common import sha256_json  # noqa: E402

OUT = ROOT / "05_OUTPUTS" / "diogenes"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def write_receipt(receipt: dict[str, Any], receipt_dir: Path) -> dict[str, Any]:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    path = receipt_dir / f"percyphon_kernel_bridge_{stamp()}.json"
    receipt["receipt_path"] = rel(path)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


def build_bridge(
    *,
    raw_command: str,
    normalized_intent: str,
    authority_class: str,
    source: str,
    villagers: list[str] | None = None,
    fluid_slots: int = 8,
    queue_name: str = "percyphon",
    lane: str = "procedural_scaffold",
    ledger_path: str | Path | None = None,
    event_log: str | Path | None = None,
    receipt_dir: Path = OUT,
) -> dict[str, Any]:
    scaffold = procedural_entity_generator(villagers or [source, normalized_intent], fluid_slots=fluid_slots)
    percyphon = {
        "schema": scaffold["schema"],
        "zero_vram": scaffold["zero_vram"],
        "slot_count": scaffold["slot_count"],
        "fluid_slot_count": scaffold["fluid_slot_count"],
        "slot": scaffold["slots"][0],
        "authority": "procedural_scaffold_candidate_not_truth",
    }
    receipt: dict[str, Any] = {
        "schema": "lucidota.diogenes.percyphon_kernel_bridge.v1",
        "generated_at": now(),
        "raw_command": raw_command,
        "normalized_intent": normalized_intent,
        "authority_class": authority_class,
        "source": source,
        "percyphon": percyphon,
        "control_packet": None,
        "route_plan": None,
        "canonical_graph_writes_performed": False,
        "model_calls_performed": False,
        "blockers": [],
        "status": "PENDING",
    }
    if authority_class != "operator_authored_assertion":
        receipt["status"] = "DENIED"
        receipt["blockers"].append("authority_class_not_operator_authored_assertion")
        return write_receipt(receipt, receipt_dir)
    payload = {
        "capability": "absurd_enqueue",
        "queue_name": queue_name,
        "lane": lane,
        "source_path": "ALGOS/percyphon.py",
        "idempotency_key": "percyphon-" + sha256_json({"raw": raw_command, "intent": normalized_intent})[:16],
        "canonical_mutation_allowed": False,
        "percyphon_slot": percyphon["slot"],
        "percyphon_authority": percyphon["authority"],
    }
    packet = make_control_packet(lane=f"diogenes:{queue_name}:{lane}", action="add_mandatory", authorized_by=source, payload=payload)
    receipt["control_packet"] = packet
    plan = route_plan_from_packet(packet, purpose="percyphon_scaffold_route", ledger_path=ledger_path, event_log=event_log)
    receipt["route_plan"] = plan
    receipt["status"] = plan["status"]
    if plan["status"] != "ROUTED":
        receipt["blockers"].append(plan.get("error") or "route_denied")
    return write_receipt(receipt, receipt_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate PercyphonAI scaffold and route it through Diogenes kernel authorization.")
    ap.add_argument("--raw-command", required=True)
    ap.add_argument("--normalized-intent", required=True)
    ap.add_argument("--authority-class", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--villager", action="append", default=[])
    ap.add_argument("--fluid-slots", type=int, default=8)
    ap.add_argument("--queue-name", default="percyphon")
    ap.add_argument("--lane", default="procedural_scaffold")
    ap.add_argument("--ledger-path")
    ap.add_argument("--event-log")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    receipt = build_bridge(raw_command=args.raw_command, normalized_intent=args.normalized_intent, authority_class=args.authority_class, source=args.source, villagers=args.villager, fluid_slots=args.fluid_slots, queue_name=args.queue_name, lane=args.lane, ledger_path=args.ledger_path, event_log=args.event_log)
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    print("RECEIPT_PATH=" + receipt["receipt_path"])
    print("PERCYPHON_KERNEL_BRIDGE=" + ("PASS" if receipt["status"] == "ROUTED" else "BLOCKED"))
    return 0 if receipt["status"] == "ROUTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
