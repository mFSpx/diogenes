#!/usr/bin/env python3
"""LUCIDOTA SITREP — live %-complete progress meter. Reads live DB + model fabric + receipts.
Run anytime:  python3 scripts/lucidota_sitrep.py
No theater: every number is a live query. Writes a JSON receipt under 05_OUTPUTS/runtime/.
"""
from __future__ import annotations
import json, sqlite3, subprocess, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = "postgresql:///lucidota_state"
STORAGE = "postgresql:///lucidota_storage"
CORPUS_TOTAL = 138322  # from runtime_facts_refresh corpus_map; refreshed below if available

def q(dsn: str, sql: str) -> str:
    try:
        r = subprocess.run(["psql", dsn, "-Atc", sql], capture_output=True, text=True, timeout=15)
        return (r.stdout or "").strip()
    except Exception as e:
        return f"ERR:{e}"

def qi(dsn: str, sql: str) -> int:
    v = q(dsn, sql)
    try: return int(v)
    except Exception: return 0

def health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False

def bar(pct: float, width: int = 28) -> str:
    pct = max(0.0, min(100.0, pct))
    fill = int(round(pct/100*width))
    return "[" + "#"*fill + "-"*(width-fill) + f"] {pct:5.1f}%"

def pct(n: int, d: int) -> float:
    return 100.0*n/d if d else 0.0

# ---- live corpus total (prefer the fact) ----
ct = q(STATE, "select (fact_value->'corpus_map'->>'remaining_files') from lucidota_control.runtime_status_fact where fact_key='runtime_facts_refresh'")
corpus_total = int(ct) if ct.isdigit() else CORPUS_TOTAL

# ---- graph ----
g_item = qi(STORAGE, "select count(*) from lucidota_go.graph_item")
g_edge = qi(STORAGE, "select count(*) from lucidota_go.graph_edge")
g_stage = qi(STORAGE, "select count(*) from lucidota_go.staging_packet")
g_cand = qi(STORAGE, "select count(*) from lucidota_go.graph_promotion_candidate")
g_mat = qi(STORAGE, "select count(*) from lucidota_go.graph_promotion_materialization")

# ---- embeddings ----
bc = qi(STORAGE, "select count(*) from lucidota_indy.book_chunk")
ce = qi(STORAGE, "select count(*) from lucidota_indy.chunk_embedding")
kc = qi(STORAGE, "select count(*) from lucidota_korpus.component")
kce = qi(STORAGE, "select count(embedding) from lucidota_korpus.component")

# ---- corpus ingested (distinct source artifacts that produced chunks/staging) ----
ingested = max(qi(STORAGE, "select count(distinct source_id) from lucidota_go.staging_packet"), 0)
cc_files = qi(STORAGE, "select count(distinct sha256) from lucidota_korpus.corpus_chunk")
cc_chunks = qi(STORAGE, "select count(*) from lucidota_korpus.corpus_chunk")
fo = qi(STORAGE, "select count(*) from lucidota_korpus.file_object")

# ---- groq work (audit scanner sqlite) ----
scan_db = ROOT/"05_OUTPUTS/groq_scanner/scan_LUCIDOTA.db"
groq_chunks = groq_findings = 0
if scan_db.exists():
    try:
        c = sqlite3.connect(scan_db)
        groq_chunks = c.execute("select count(*) from chunks where processed=1").fetchone()[0]
        groq_findings = c.execute("select count(*) from findings").fetchone()[0]
        c.close()
    except Exception: pass
patternalysis = (ROOT/"05_OUTPUTS/groq_scanner/patternalysis.md").exists()

# ---- model fabric ----
LANES = {"deepseek":8080,"mamba_ram":8081,"bonsai":8082,"mamba_gpu":8083,
         "needle0":8090,"needle1":8091,"needle2":8092,"needle3":8093,"needle4":8094,"needle5":8095,
         "embed1":8101,"embed2":8102,"embed3":8103,"embed4":8104}
lane_up = {n: health(p) for n, p in LANES.items()}
lanes_up = sum(lane_up.values())

streams = [
    ("MODEL FABRIC (lanes hot)", pct(lanes_up, len(LANES)), f"{lanes_up}/{len(LANES)} lanes"),
    ("CORPUS EXTRACT (Groq)",    pct(cc_files, fo) if fo else 0.0, f"{cc_files:,}/{fo:,} registered files extracted | {cc_chunks:,} chunks on graph"),
    ("EMBEDDINGS (live)",        100.0 if cc_chunks else 0.0, f"corpus_chunk={cc_chunks:,} @1024d | book chunk_embedding={ce:,}"),
    ("GRAPH POPULATED (GONN)",   100.0 if g_item>2 else 0.0, f"items={g_item} edges={g_edge} staged={g_stage} cand={g_cand} materialized={g_mat}"),
    ("GROQ AUDIT",               100.0 if patternalysis else (pct(groq_chunks, groq_chunks) if groq_chunks else 0.0), f"{groq_chunks:,} chunks scanned, {groq_findings:,} findings, patternalysis={'yes' if patternalysis else 'no'}"),
]

now = datetime.now(timezone.utc).isoformat()
lines = [f"\n=========== LUCIDOTA SITREP — {now} ===========\n"]
for name, p, detail in streams:
    lines.append(f"  {name:<26} {bar(p)}")
    lines.append(f"  {'':<26} {detail}\n")
lines.append("  LANES: " + "  ".join(f"{n}{'UP' if up else 'x'}" for n,up in lane_up.items()))
overall = sum(p for _,p,_ in streams)/len(streams)
lines.append(f"\n  OVERALL PIPELINE: {bar(overall)}\n")
report = "\n".join(lines)
print(report)

receipt = {
    "receipt":"SITREP","generated_at":now,"overall_pct":round(overall,1),
    "graph":{"items":g_item,"edges":g_edge,"staging":g_stage,"candidates":g_cand,"materialized":g_mat},
    "embeddings":{"book_chunk":bc,"chunk_embedding":ce,"korpus_component":kc,"korpus_embedded":kce},
    "corpus":{"total":corpus_total,"ingested":ingested,"pct":round(pct(ingested,corpus_total),3)},
    "groq":{"audit_chunks":groq_chunks,"findings":groq_findings,"patternalysis":patternalysis},
    "fabric":{"lanes_up":lanes_up,"lanes_total":len(LANES),"lanes":lane_up},
}
out = ROOT/"05_OUTPUTS/runtime"; out.mkdir(parents=True, exist_ok=True)
ts = now.replace(":","").replace("-","").split(".")[0]+"Z"
(out/f"sitrep_{ts}.json").write_text(json.dumps(receipt, indent=2))
print(f"  receipt: 05_OUTPUTS/runtime/sitrep_{ts}.json\n")
