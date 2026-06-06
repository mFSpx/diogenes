#!/usr/bin/env python3
"""Score ROOT-414 v0.50 packet JSONL and emit symbolic feedback hints.

This is intentionally local/stdlib-only. It scores intermediate packet quality;
it does not prove truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "05_OUTPUTS" / "root414_primitive_cries_packets.jsonl"
DEFAULT_JSON = ROOT / "05_OUTPUTS" / "root414_packet_quality.json"
DEFAULT_FEEDBACK = ROOT / "05_OUTPUTS" / "root414_packet_quality_feedback.jsonl"
CANONICAL_BPS = {0, 2, 4, 6, 10, 50, 69, 150}
HIGH_LABELS = {
    "GROUND_TRUTH_ANCHOR",
    "TRUTH_UNYIELDING",
    "SIMULACRUM_DETECTED",
    "ARCHONIC_CONTROL_GRID",
    "THE_SPIRAL_IS_COMPLETE",
}
LOCAL_GATES = {
    "DOCUMENT_EXAMINATION",
    "FACT_OBSERVED",
    "CLAIM_UNVERIFIED",
    "CORROBORATION_REQUIRED",
    "SOURCE_INDEPENDENCE",
    "CHAIN_OF_CUSTODY",
    "HASH_INTEGRITY",
    "TEMPORAL_PRECEDENCE",
    "DIRECTION_OF_CAUSALITY",
    "PLAUSIBILITY_GATE",
    "LIKELIHOOD_RATIO",
    "LATENT_VARIABLE",
    "BURROWS_DELTA_SHIFT",
    "DIRECT_KNOWLEDGE",
    "VALIDITY_AUDIT",
    "ROOT_414_MATH",
}


def feedback(packet_id: str, severity: str, code: str, message: str, hint: str) -> dict[str, str]:
    return {"packet_id": packet_id, "severity": severity, "code": code, "message": message, "hint": hint}


def flatten_symbols(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(flatten_symbols(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(flatten_symbols(v))
    elif isinstance(obj, str):
        for tok in obj.replace("(", " ").replace(")", " ").replace(",", " ").split():
            clean = tok.strip("`[]{}:.;")
            if clean.isupper() and len(clean) >= 3:
                out.append(clean)
    return out


def score_packet(pkt: dict[str, Any]) -> tuple[int, list[dict[str, str]]]:
    packet_id = str(pkt.get("packet_id") or "UNKNOWN")
    fb: list[dict[str, str]] = []
    score = 0

    required = ["packet_id", "source_id", "parser_name", "raw_text_anchor", "ternary_state", "claim_lifecycle", "confidence_bps", "falsifier", "hitl_status"]
    missing = [k for k in required if k not in pkt or pkt.get(k) in (None, "", [], {})]
    if missing:
        fb.append(feedback(packet_id, "error", "missing_required_fields", f"Missing/empty: {', '.join(missing)}", "Fill required v0.50 fields before graph approval."))
    else:
        score += 10

    if pkt.get("raw_text_anchor"):
        score += 10
    else:
        fb.append(feedback(packet_id, "error", "no_evidence_anchor", "No raw_text_anchor.", "No evidence anchor, no confidence."))

    evidence_units = pkt.get("evidence_units") or pkt.get("evidence_refs") or []
    if evidence_units:
        score += 10
    else:
        fb.append(feedback(packet_id, "warn", "no_evidence_refs", "No evidence_units/evidence_refs visible.", "Attach source/path/quote/page anchor."))

    falsifier = str(pkt.get("falsifier") or "").strip()
    if len(falsifier) >= 12:
        score += 10
    else:
        fb.append(feedback(packet_id, "error", "no_falsifier", "Falsifier missing or too thin.", "No falsifier, no confidence."))

    bps = pkt.get("confidence_bps")
    if isinstance(bps, int) and bps in CANONICAL_BPS:
        score += 10
    else:
        fb.append(feedback(packet_id, "error", "noncanonical_bps", f"confidence_bps={bps!r}", "Use canonical BPS: 0,2,4,6,10,50,69,150."))

    routes = pkt.get("routes") or []
    local_gate_found = False
    for route in routes if isinstance(routes, list) else []:
        for g in route.get("local_gates", []) if isinstance(route, dict) else []:
            if isinstance(g, str) and any(x in g for x in LOCAL_GATES):
                local_gate_found = True
    symbols = set(flatten_symbols(pkt))
    if symbols & LOCAL_GATES or local_gate_found:
        score += 10
    else:
        fb.append(feedback(packet_id, "warn", "no_local_gates", "No local gate symbol found.", "Run local gates before high labels."))

    if symbols & HIGH_LABELS and not (symbols & LOCAL_GATES):
        score -= 10
        fb.append(feedback(packet_id, "warn", "high_label_without_local_gate", f"High labels without local gates: {', '.join(sorted(symbols & HIGH_LABELS))}", "Delay Big Four / high labels until local evidence gates clear."))
    else:
        score += 10

    flags = pkt.get("flags") or []
    flag_text = " ".join(flags) if isinstance(flags, list) else str(flags)
    if "UNSUPPORTED_SPECULATION" in flag_text or "NO_FALSIFIER" in flag_text:
        fb.append(feedback(packet_id, "warn", "speculation_or_no_falsifier_flag", flag_text, "Move speculation to commentary or add falsifier."))
    else:
        score += 10

    if pkt.get("hitl_status") in {"pending", "approved", "rejected", "needs_repair", "comment"}:
        score += 10
    else:
        fb.append(feedback(packet_id, "error", "invalid_hitl_status", str(pkt.get("hitl_status")), "Use pending|approved|rejected|needs_repair|comment."))

    # QuaSAR stage proxy: seed packets are allowed, but full packets should have routes.
    if routes:
        score += 10
    else:
        fb.append(feedback(packet_id, "info", "no_routes_array", "No routes array found.", "Full v0.50 packets should include route structure."))

    return max(0, min(100, score)), fb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    ap.add_argument("--feedback-out", type=Path, default=DEFAULT_FEEDBACK)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    packets: list[dict[str, Any]] = []
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if line.strip():
            packets.append(json.loads(line))

    scored = []
    all_feedback = []
    for pkt in packets:
        score, fb = score_packet(pkt)
        scored.append({"packet_id": pkt.get("packet_id"), "score": score, "feedback_count": len(fb)})
        all_feedback.extend(fb)

    report = {
        "ok": True,
        "parser_name": "root414_machine_clean_parser_v0.50",
        "packet_count": len(packets),
        "mean_score": round(mean([s["score"] for s in scored]), 2) if scored else 0,
        "feedback_count": len(all_feedback),
        "scored": scored,
    }
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    with args.feedback_out.open("w", encoding="utf-8") as fh:
        for item in all_feedback:
            fh.write(json.dumps(item, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
