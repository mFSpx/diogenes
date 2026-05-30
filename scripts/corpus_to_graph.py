#!/usr/bin/env python3
"""corpus_to_graph.py — THE BRIDGE the ingest pipeline was missing.

corpus_chunk (EVIDENCE, the embed/archive crush sink) -> Groq entity extraction
(cloud hands, not Claude) -> lucidota_go.staging_packet (the graph HYPOTHESIS field,
status=pending). This is candidate_writer-legal: staging_packet is NOT canonical graph
truth. Promotion (staging_packet -> graph_promotion_packet via the gate) and
materialization (-> graph_item) stay operator/authority-gated downstream.

Doctrine: CLAIM != EVIDENCE; MODEL OUTPUT != AUTHORITY. A Groq-extracted entity is a
HYPOTHESIS staged from raw evidence (the chunk, raw_anchor=cas://<sha>#<idx>), not truth.

Usage: scripts/corpus_to_graph.py --limit 50
"""
import os, sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_groq_extractor as cge
import requests

ROOT = Path(os.environ.get('LUCIDOTA_HOME', '/home/mfspx/LUCIDOTA'))
RECEIPT = ROOT / '05_OUTPUTS/corpus_to_graph_receipt.json'
GROQ_MODEL = os.environ.get('LUCIDOTA_GROQ_EXTRACT_MODEL', 'llama-3.1-8b-instant')
# WO-2: map the generic NER labels Groq emits -> valid base terms in the 75-ontology
# (term_registry). Esoteric IO/adversarial terms (LIE/GHOST/MASK/GRIP...) are applied
# later by analysis passes, not raw NER. Unknowns default to ENTITY (always valid).
TERM_MAP = {
    "PERSON": "ENTITY", "ORG": "GROUP", "ORGANIZATION": "GROUP", "GROUP": "GROUP",
    "LOCATION": "LOCATION", "PLACE": "LOCATION", "DATE": "TIME", "TIME": "TIME",
    "MONEY": "ATTRIBUTE", "AMOUNT": "ATTRIBUTE", "CLAIM": "CLAIM", "EVENT": "EVENT",
    "DOC": "SOURCE", "DOCUMENT": "SOURCE", "SOURCE": "SOURCE", "LAW": "LAW",
    "REGULATOR": "REGULATOR", "WITNESS": "WITNESS", "THREAT": "THREAT", "ENTITY": "ENTITY",
}

def groq_entities(text):
    prompt = ("Extract up to 6 salient items from the text as a STRICT JSON array of "
              "{\"term\":<one of PERSON|ORG|LOCATION|DATE|MONEY|CLAIM|EVENT|DOC>,"
              "\"label\":<verbatim span>}. Only the JSON array, no prose.\n\nTEXT:\n" + (text or "")[:4000])
    body = {"model": GROQ_MODEL, "max_tokens": 512, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    h = {"Authorization": f"Bearer {cge.GROQ_API_KEY}", "Content-Type": "application/json",
         "User-Agent": "groq-python/0.28.0"}
    import time
    r = None
    for attempt in range(4):
        r = requests.post(cge.GROQ_API_URL, json=body, headers=h, timeout=60)
        if r.status_code == 200:
            break
        if r.status_code == 429:   # Groq rate limit -> back off
            time.sleep(2 * (attempt + 1)); continue
        return []
    if r is None or r.status_code != 200:
        return []
    c = r.json()["choices"][0]["message"]["content"].strip()
    try:
        arr = json.loads(c[c.find("["):c.rfind("]") + 1])
    except Exception:
        return []
    out = []
    for e in (arr if isinstance(arr, list) else []):
        if isinstance(e, dict) and e.get("term") and e.get("label"):
            raw = str(e["term"]).upper()[:32]
            term = TERM_MAP.get(raw, 'ENTITY')   # map generic NER -> valid 75-ontology base term
            out.append({"term": term, "label": str(e["label"])[:200]})
    return out[:6]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=25)
    a = ap.parse_args()
    conn = cge.connect_db(); cur = conn.cursor()
    # incremental: only chunks not already bridged into the hypothesis field
    cur.execute("SELECT c.sha256, c.source_path, c.chunk_index, c.content "
                "FROM lucidota_korpus.corpus_chunk c "
                "WHERE NOT EXISTS (SELECT 1 FROM lucidota_go.staging_packet s "
                "WHERE s.source_id = 'cas://'||c.sha256||'#'||c.chunk_index) "
                "ORDER BY c.created_at DESC LIMIT %s", (a.limit,))
    rows = cur.fetchall()
    chunks = cands = staged = errs = 0
    for sha, src, idx, content in rows:
        chunks += 1
        try:
            ents = groq_entities(content)
        except Exception:
            errs += 1; continue
        anchor = f"cas://{sha}#{idx}"
        for e in ents:
            cands += 1
            try:
                # dedupe: skip identical (anchor, term, claim) already staged
                cur.execute("SELECT 1 FROM lucidota_go.staging_packet WHERE source_id=%s AND proposed_term=%s AND claim=%s LIMIT 1",
                            (anchor, e["term"], e["label"][:500]))
                if cur.fetchone():
                    continue
                cur.execute(
                    "INSERT INTO lucidota_go.staging_packet "
                    "(source_id,parser_name,proposed_term,raw_anchor,claim,proposed_item,proposed_edges,status,confidence_bps) "
                    "VALUES (%s,'corpus_to_graph',%s,%s,%s,%s::jsonb,'[]'::jsonb,'pending',10)",
                    (anchor, e["term"], e["label"][:200], e["label"][:500],
                     json.dumps({"term": e["term"], "label": e["label"], "sha256": sha, "chunk_index": idx, "source_path": src})))
                staged += 1
            except Exception:
                conn.rollback(); errs += 1; continue
        conn.commit()
    rec = {"scope": "GRAPH_PROMOTION_RUNTIME", "chunks_read": chunks, "candidates": cands,
           "staging_packets_written": staged, "errors": errs}
    RECEIPT.write_text(json.dumps(rec, indent=2))
    print("corpus_to_graph " + json.dumps(rec))

if __name__ == "__main__":
    main()
