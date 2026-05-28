#!/usr/bin/env python3
"""Scan model custody gaps; do not run models

Research-only dry-run scaffold. No DB writes. No graph writes.
"""
from __future__ import annotations
import argparse, json, datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "05_OUTPUTS" / "research_integration"
PAYLOAD = {
  "search_roots": [
    "03_VAULT/models",
    "04_RUNTIME/lora_cartridges",
    "LoRAs",
    "MODELS"
  ],
  "artifact_kinds": [
    "GGUF",
    "LoRA",
    "adapter",
    "Hoebrain"
  ],
  "models_run": False
}

def stamp(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

def main():
    ap=argparse.ArgumentParser(description='Scan model custody gaps; do not run models')
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--input")
    ap.add_argument("--out-dir", default=None)
    args=ap.parse_args()
    out_dir=Path(args.out_dir) if args.out_dir else DEFAULT_OUT / stamp()
    if not out_dir.is_absolute(): out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report={"schema":'lucidota.research.latent_mode_registry_scan.v1',"generated_at":now(),"mode":"dry_run","purpose":'Scan model custody gaps; do not run models',"input":args.input,"payload":PAYLOAD,"db_writes_performed":False,"graph_writes_performed":False,"canonical_mutation_allowed":False,"blockers":[]}
    out=out_dir/(Path(__file__).stem + "_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
