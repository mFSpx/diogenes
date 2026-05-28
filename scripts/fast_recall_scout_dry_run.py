#!/usr/bin/env python3
"""SimpleMem-style fast recall scout; candidates only

Research-only dry-run scaffold. No DB writes. No graph writes.
"""
from __future__ import annotations
import argparse, json, datetime
from pathlib import Path
import hashlib
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "05_OUTPUTS" / "research_integration"
PAYLOAD = {
  "pass_kinds": [
    "broad",
    "lexical",
    "semantic_neighbor",
    "contradiction",
    "temporal",
    "variant",
    "weird_neighbor"
  ],
  "safe_to_answer_from_this_alone": False,
  "candidates": [
    {
      "candidate_ref": "sample",
      "authority": "low",
      "why": "lexical hit only"
    }
  ]
}

def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

def collect_sources(raw: str | None):
    if not raw:
        return []
    p = Path(raw)
    paths = []
    if p.is_dir():
        paths = sorted(p.glob("LUCIDOTA_*.md"))
    elif p.is_file() and p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        for rec in data.get("records", []):
            rel = rec.get("source_md_path")
            if rel:
                rp = ROOT / rel
                if rp.exists(): paths.append(rp)
    elif p.is_file():
        paths = [p]
    sources=[]
    for path in paths:
        try:
            text=path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        sources.append({"path": str(path.relative_to(ROOT)) if path.is_absolute() and ROOT in path.parents else str(path), "text": text, "sha256": sha_text(text)})
    return sources

def recall_candidates(sources):
    pass_kinds = PAYLOAD["pass_kinds"]
    needles = ["graph", "truth", "operator", "memory", "workflow", "evidence", "semantic", "decision", "trace", "context"]
    candidates=[]
    for src in sources:
        lower=src["text"].lower()
        for kind in pass_kinds:
            hits=[n for n in needles if n in lower]
            if kind == "weird_neighbor":
                hits += [w for w in ["ponyboy", "crucifixion", "WOOOAAHH".lower()] if w in lower]
            if hits:
                candidates.append({"pass_kind": kind, "source_ref": src["path"], "source_sha256": src["sha256"], "matched_terms": sorted(set(hits)), "authority": "candidate_low_authority", "safe_to_answer_from_this_alone": False})
    return candidates

def stamp(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

def main():
    ap=argparse.ArgumentParser(description='SimpleMem-style fast recall scout; candidates only')
    ap.add_argument("--dry-run", action="store_true", default=True)
    ap.add_argument("--input")
    ap.add_argument("--out-dir", default=None)
    args=ap.parse_args()
    out_dir=Path(args.out_dir) if args.out_dir else DEFAULT_OUT / stamp()
    if not out_dir.is_absolute(): out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = collect_sources(args.input)
    candidates = recall_candidates(sources)
    payload = dict(PAYLOAD)
    payload["scanned_source_count"] = len(sources)
    payload["candidate_count"] = len(candidates)
    payload["candidates"] = candidates
    report={"schema":'lucidota.research.fast_recall_scout.v1',"generated_at":now(),"mode":"dry_run","purpose":'SimpleMem-style fast recall scout; candidates only',"input":args.input,"payload":payload,"db_writes_performed":False,"graph_writes_performed":False,"canonical_mutation_allowed":False,"safe_to_answer_from_this_alone":False,"blockers":[] if sources else ["no_input_sources_scanned"]}
    out=out_dir/(Path(__file__).stem + "_report.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
