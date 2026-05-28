#!/usr/bin/env python3
"""Bridge TRACER-lite epistemic labels into claim-packet dry-run output.

Dry-run only. No DB writes. No graph writes. Claim packets remain candidates.
"""
from __future__ import annotations
import argparse, json, hashlib, datetime
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]
REQUIRED_LABELS = ["quote","compression","inference","abduction","speculation","operator_prior","heuristic","contradiction","falsification_target","PFM"]

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
def stamp(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
def sha(obj: Any) -> str: return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def classify(packet: dict[str, Any]) -> dict[str, Any]:
    text = str(packet.get("claim_text", ""))
    if text.startswith("Detected "):
        label = "inference"
        rationale = "GLiNER/entity detection is a model-derived inference over a quoted span, not graph truth."
    elif packet.get("authority_class") == "operator_authored_assertion":
        label = "operator_prior"
        rationale = "Operator-authored assertion is carried as prior/doctrine pending evidence routing."
    else:
        label = "heuristic"
        rationale = "Default dry-run heuristic label; requires review before promotion."
    return {"epistemic_label": label, "rationale": rationale, "pfm_required": label in {"abduction", "speculation", "heuristic"}}

def main() -> int:
    ap = argparse.ArgumentParser(description="TRACER-lite claim-packet bridge dry-run")
    ap.add_argument("--claim-report", required=True)
    ap.add_argument("--epistemic-report")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", default=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "05_OUTPUTS" / "research_integration" / stamp()
    if not out_dir.is_absolute(): out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    claim_report = load_json(args.claim_report)
    epistemic_report = load_json(args.epistemic_report) if args.epistemic_report else None
    enriched=[]
    for packet in claim_report.get("claim_packets", []):
        ep = classify(packet)
        enriched.append({**packet, "tracer_lite": {"required_labels": REQUIRED_LABELS, **ep, "source_claim_report": args.claim_report}})
    report={
        "schema":"lucidota.research.tracer_claim_packet_bridge.v1",
        "generated_at":now(),
        "mode":"dry_run",
        "claim_report":args.claim_report,
        "epistemic_report":args.epistemic_report,
        "required_labels":REQUIRED_LABELS,
        "enriched_claim_packets":enriched,
        "enriched_count":len(enriched),
        "payload_sha256":sha(enriched),
        "db_writes_performed":False,
        "graph_writes_performed":False,
        "canonical_mutation_allowed":False,
        "hard_law":"TRACER-lite labels epistemic moves; retrieved/generated/model-detected does not mean verified.",
        "blockers":[] if enriched else ["no_claim_packets_to_label"],
    }
    out=out_dir/"tracer_claim_packet_bridge_dry_run_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    return 0 if enriched else 1
if __name__ == "__main__":
    raise SystemExit(main())
