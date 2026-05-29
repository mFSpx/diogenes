#!/usr/bin/env python3
"""Define CatchMe / Personal Context Spine consent, revocation, and sensitivity scopes.

Dry-run only. No DB writes. No graph writes.
"""
from __future__ import annotations
import argparse, json, datetime, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
def stamp(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
def sha(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

def main() -> int:
    ap=argparse.ArgumentParser(description="CatchMe scope contract dry-run")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", default=True)
    args=ap.parse_args()
    out_dir=Path(args.out_dir) if args.out_dir else ROOT/"05_OUTPUTS"/"research_integration"/stamp()
    if not out_dir.is_absolute(): out_dir=ROOT/out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    contract={
      "scope_classes":["public_project","operator_owned","private_context","sensitive_secret_adjacent","quarantined_secret","relationship_context","health_body_context","financial_legal_context"],
      "consent_states":["allowed_local_index","allowed_metadata_only","allowed_operator_review_only","blocked_quarantined","revoked"],
      "revocation_policy":{"effect":"future reads blocked; prior derived artifacts marked revoked/superseded, not silently deleted","requires":"operator command envelope or explicit local command"},
      "sensitivity_policy":{"quarantined_secret":"cannot be embedded, summarized, extracted, or graph-promoted","private_context":"requires explicit downstream purpose","operator_owned":"default local custody only"},
      "activity_event_required_fields":["source_ref","source_sha256","event_time_candidates","actor","action","object_ref","sensitivity_scope","consent_state","evidence_refs"],
      "canonical_mutation_allowed":False,
    }
    report={"schema":"lucidota.research.catchme_scope_contract.v1","generated_at":now(),"mode":"dry_run","component":"CatchMe / Personal Context Spine","contract":contract,"payload_sha256":sha(contract),"db_writes_performed":False,"graph_writes_performed":False,"canonical_mutation_allowed":False,"blockers":["operator_review_required_before_execute_path"]}
    out=out_dir/"catchme_scope_contract_dry_run_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
