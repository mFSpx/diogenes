#!/usr/bin/env python3
"""Turn a symbol-compare receipt into compact GO-25 worker packets."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "goals"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def load_report(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding="utf-8"))


def packet_for_lane(lane: str, compare: dict[str, Any], idx: int) -> dict[str, Any]:
    next_seed = compare.get("next_seed") or ["OBJECT", "EVENT", "EDGE"]
    top_rules = compare.get("improved_pairs") or compare.get("stable_pairs") or []
    top_rule = top_rules[0] if top_rules else {}
    evidence_refs = compare.get("evidence_refs") or []
    return {
        "schema": "lucidota.worker_order.v1",
        "target": lane,
        "intent": f"Use GO-25 seed {', '.join(next_seed[:3])} to validate and compress the next evidence-backed symbol step.",
        "ontology_mode": "GO25_STRICT",
        "ontology_terms": list(next_seed[:3]),
        "evidence_refs": evidence_refs,
        "output_contract": {
            "schema": "lucidota.worker_order.v1",
            "required_output": ["status", "result", "next_action", "receipt_path"],
            "decision_pairs_min": 2,
            "evidence_required": True,
            "no_prose": True,
        },
        "constraints": [
            "small batch",
            "deterministic",
            "no canonical graph writes",
            "return only receipt-backed output",
        ],
        "required_output": ["status", "result", "next_action", "receipt_path"],
        "seed_compare_ref": compare.get("report_path") or compare.get("receipt_path") or "",
        "lane_priority": idx,
        "top_rule": top_rule,
    }


def build_report(compare: dict[str, Any], lanes: list[str]) -> dict[str, Any]:
    packets = [packet_for_lane(lane, compare, idx + 1) for idx, lane in enumerate(lanes)]
    return {
        "schema": "lucidota.graph_symbol_dispatch.v1",
        "generated_at": now(),
        "objective": "Dispatch compare output into compact GO-25 worker packets.",
        "intent": "symbol_dispatch",
        "ontology_mode": "GO25_STRICT",
        "ontology_terms": ["OBJECT", "EVENT", "EDGE"],
        "evidence_refs": [compare.get("report_path") or compare.get("receipt_path") or ""],
        "status": "PASS",
        "next_action": "Feed the packets to Groq/local workers and compare the returned receipts on the next pass.",
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
        "compare_report": rel(compare.get("report_path") or compare.get("receipt_path") or ""),
        "packets": packets,
    }


def write_report(report: dict[str, Any]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"graph_symbol_dispatch_{stamp()}.json"
    report["receipt_path"] = rel(path)
    report["report_path"] = report["receipt_path"]
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", required=True)
    ap.add_argument("--lanes", default="groq,local")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    compare = load_report(a.compare)
    lanes = [lane.strip() for lane in a.lanes.split(",") if lane.strip()]
    report = write_report(build_report(compare, lanes))
    print("REPORT_PATH=" + report["report_path"])
    print("GRAPH_SYMBOL_DISPATCH=PASS")
    print(json.dumps(report, sort_keys=True) if a.json else report["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
