#!/usr/bin/env python3
"""Compare two symbol-condensation receipts and propose the next seed."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "goals"
GO25 = ("OBJECT", "EVENT", "EDGE")


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


def claim_map(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out = {}
    for claim in report.get("claims") or []:
        pair = tuple(claim.get("pair") or [])
        if len(pair) == 2:
            out[(str(pair[0]), str(pair[1]))] = claim
    return out


def compare_reports(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    cur = claim_map(current)
    base = claim_map(baseline)
    stable, new, lost, improved, weakened = [], [], [], [], []
    for pair, claim in cur.items():
        prev = base.get(pair)
        row = {
            "pair": list(pair),
            "current_confidence_bps": claim.get("confidence_bps", 0),
            "current_status": claim.get("status"),
            "baseline_confidence_bps": prev.get("confidence_bps", 0) if prev else None,
            "baseline_status": prev.get("status") if prev else None,
            "delta_bps": int(claim.get("confidence_bps", 0)) - int(prev.get("confidence_bps", 0)) if prev else int(claim.get("confidence_bps", 0)),
            "evidence_refs": claim.get("evidence_refs") or [],
            "mutable": bool(claim.get("mutable", True)),
        }
        if prev:
            stable.append(row)
            if row["delta_bps"] >= 500:
                improved.append(row)
            elif row["delta_bps"] <= -500:
                weakened.append(row)
        else:
            new.append(row)
    for pair, claim in base.items():
        if pair not in cur:
            lost.append({
                "pair": list(pair),
                "baseline_confidence_bps": claim.get("confidence_bps", 0),
                "baseline_status": claim.get("status"),
                "delta_bps": -int(claim.get("confidence_bps", 0)),
                "evidence_refs": claim.get("evidence_refs") or [],
            })
    next_seed = sorted(
        [row["pair"][0] for row in improved[:3]] +
        [row["pair"][1] for row in improved[:3]] +
        [row["pair"][0] for row in new[:2]] +
        [row["pair"][1] for row in new[:2]]
    )
    next_seed = [s for s in next_seed if s in GO25][:3] or list(GO25)
    return {
        "schema": "lucidota.graph_symbol_compare.v1",
        "generated_at": now(),
        "objective": "Compare symbol condensation receipts over time and propose the next seed.",
        "intent": "symbol_compare",
        "ontology_mode": "GO25_STRICT",
        "ontology_terms": list(GO25),
        "evidence_refs": [rel(current.get("report_path") or current.get("receipt_path") or ""), rel(baseline.get("report_path") or baseline.get("receipt_path") or "")],
        "status": "PASS",
        "next_action": "Use the next_seed to drive the following condensation pass.",
        "model_calls_performed": False,
        "canonical_graph_writes_performed": False,
        "current_report": rel(current.get("report_path") or current.get("receipt_path") or ""),
        "baseline_report": rel(baseline.get("report_path") or baseline.get("receipt_path") or ""),
        "stable_pairs": stable,
        "new_pairs": new,
        "lost_pairs": lost,
        "improved_pairs": improved,
        "weakened_pairs": weakened,
        "comparison_summary": {
            "stable": len(stable),
            "new": len(new),
            "lost": len(lost),
            "improved": len(improved),
            "weakened": len(weakened),
            "current_backed": len([c for c in cur.values() if c.get("status") == "backed"]),
            "baseline_backed": len([c for c in base.values() if c.get("status") == "backed"]),
        },
        "next_seed": next_seed,
    }


def write_report(report: dict[str, Any]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"graph_symbol_compare_{stamp()}.json"
    report["receipt_path"] = rel(path)
    report["report_path"] = report["receipt_path"]
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    report = write_report(compare_reports(load_report(a.current), load_report(a.baseline)))
    print("REPORT_PATH=" + report["report_path"])
    print("GRAPH_SYMBOL_COMPARE=PASS")
    print(json.dumps(report, sort_keys=True) if a.json else report["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
