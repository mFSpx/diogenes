#!/usr/bin/env python3
"""Bridge DeMem decision boundaries into surface instruction compiler constraints.

Dry-run only. No DB writes. No graph writes. Boundaries become guardrails, not mutation authority.
"""
from __future__ import annotations
import argparse, json, hashlib, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOUNDARIES = ["generated != policy-mutable", "retrieved != verified", "repeated != preferred", "surface != UI", "graph path != evidence"]

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
def stamp(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
def sha(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

def load_boundaries(path: str | None):
    if not path: return DEFAULT_BOUNDARIES
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    payload=data.get("payload", data)
    return payload.get("boundaries", DEFAULT_BOUNDARIES)

def main() -> int:
    ap=argparse.ArgumentParser(description="DeMem boundary bridge for surface instruction compiler")
    ap.add_argument("--boundary-report")
    ap.add_argument("--surface-report")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", default=True)
    args=ap.parse_args()
    out_dir=Path(args.out_dir) if args.out_dir else ROOT/"05_OUTPUTS"/"research_integration"/stamp()
    if not out_dir.is_absolute(): out_dir=ROOT/out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    boundaries=load_boundaries(args.boundary_report)
    surface=json.loads(Path(args.surface_report).read_text(encoding="utf-8")) if args.surface_report else {}
    compiler_constraints=[{"boundary":b,"surface_instruction_guard":f"Before acting, preserve boundary: {b}.","mutation_authority_granted":False} for b in boundaries]
    instruction=surface.get("plain_language_instruction", "No prior surface instruction supplied; dry-run guardrail packet only.")
    guarded_instruction=instruction + " Boundaries: " + "; ".join(boundaries) + "."
    report={"schema":"lucidota.research.demem_surface_boundary_bridge.v1","generated_at":now(),"mode":"dry_run","boundary_report":args.boundary_report,"surface_report":args.surface_report,"boundaries":boundaries,"compiler_constraints":compiler_constraints,"guarded_plain_language_instruction":guarded_instruction,"payload_sha256":sha(compiler_constraints),"db_writes_performed":False,"graph_writes_performed":False,"canonical_mutation_allowed":False,"hard_law":"Decision boundaries constrain generated instructions; they do not grant mutation authority.","blockers":[]}
    out=out_dir/"demem_surface_boundary_bridge_dry_run_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
