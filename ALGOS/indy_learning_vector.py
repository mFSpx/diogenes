"""INDY_READs deterministic book-learning vectors.

Pure math/packet layer. Runtime scripts own I/O and receipts. No model calls.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORD_RE = re.compile(r"\S+")
DEFAULT_TERMS = (
    "ENTITY", "ATTRIBUTE", "RELATIONSHIP", "ACTION", "EVENT", "TIME", "EVIDENCE",
    "CLAIM", "HYPOTHESIS", "SIGNAL", "PATTERN", "TOOL", "ALGORITHM", "BOOK",
    "SOURCE", "LEAD", "LOCATION", "LAW", "RULE",
)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def load_go_terms(root: Path = ROOT) -> list[str]:
    p = root / "OFFICIAL_ONTOLOGY.json"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        terms = data.get("active_terms") or []
        return [str(t).upper() for t in terms if str(t).strip()]
    except Exception:
        return list(DEFAULT_TERMS)


def tokenize(text: str) -> list[dict[str, Any]]:
    return [{"token": m.group(0), "start": m.start(), "end": m.end()} for m in WORD_RE.finditer(text)]


def chunk_text_tokens(text: str, *, max_tokens: int = 500, overlap_tokens: int = 0, source_ref: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be >=0 and < max_tokens")
    toks = tokenize(text)
    source_ref = dict(source_ref or {})
    if not toks:
        cid = "chunk:" + sha256_json({"source_ref": source_ref, "empty": True})[:24]
        return [{"chunk_id": cid, "chunk_index": 0, "token_start": 0, "token_end": 0, "char_start": 0, "char_end": 0, "token_count": 0, "text": "", "source_ref": source_ref}]
    chunks: list[dict[str, Any]] = []
    token_start = 0
    idx = 0
    while token_start < len(toks):
        token_end = min(len(toks), token_start + max_tokens)
        char_start = toks[token_start]["start"]
        char_end = toks[token_end - 1]["end"]
        chunk_text = text[char_start:char_end]
        cid = "chunk:" + sha256_json({"source_ref": source_ref, "token_start": token_start, "token_end": token_end, "text": chunk_text})[:24]
        chunks.append({
            "chunk_id": cid,
            "chunk_index": idx,
            "token_start": token_start,
            "token_end": token_end,
            "char_start": char_start,
            "char_end": char_end,
            "token_count": token_end - token_start,
            "text": chunk_text,
            "source_ref": source_ref,
        })
        if token_end == len(toks):
            break
        token_start = token_end - overlap_tokens
        idx += 1
    return chunks


def ontology_hits_for_text(text: str, terms: list[str] | None = None) -> list[dict[str, Any]]:
    terms = terms or load_go_terms()
    upper = text.upper()
    hits: list[dict[str, Any]] = []
    for term in terms:
        if re.search(r"\b" + re.escape(term.upper()) + r"\b", upper):
            hits.append({"term": term.upper(), "count": len(re.findall(r"\b" + re.escape(term.upper()) + r"\b", upper))})
    return hits


def build_learning_vector(*, chunks: list[dict[str, Any]], source_ref: dict[str, Any], terms: list[str] | None = None) -> dict[str, Any]:
    terms = terms or load_go_terms()
    counter: Counter[str] = Counter()
    for c in chunks:
        for hit in ontology_hits_for_text(c.get("text", ""), terms):
            counter[hit["term"]] += int(hit["count"])
    hits = [{"term": term, "count": count} for term, count in sorted(counter.items())]
    source_id = "book:" + sha256_json(source_ref)[:24]
    event_id = "book_event:" + sha256_json({"source_ref": source_ref, "chunks": [c.get("chunk_id") for c in chunks]})[:24]
    jzloads: list[dict[str, Any]] = [
        {"kind": "OBJECT", "id": source_id, "type": "BOOK_SOURCE", "source_ref": source_ref, "chunk_count": len(chunks)},
        {"kind": "EVENT", "id": event_id, "type": "BOOK_CHUNKED", "source": source_id, "chunk_count": len(chunks), "ontology_hit_count": sum(counter.values())},
    ]
    for hit in hits:
        term_id = "go_term:" + hit["term"]
        jzloads.append({"kind": "EDGE", "from": source_id, "to": term_id, "type": "HAS_ONTOLOGY_SIGNAL", "weight": hit["count"]})
    for c in chunks[:64]:
        jzloads.append({"kind": "EDGE", "from": source_id, "to": c["chunk_id"], "type": "HAS_CHUNK", "chunk_index": c.get("chunk_index", 0)})
    return {
        "schema": "lucidota.indy_reads.learning_vector.v1",
        "source_id": source_id,
        "event_id": event_id,
        "source_ref": source_ref,
        "chunk_count": len(chunks),
        "ontology_hits": hits,
        "jzloads": jzloads,
        "canonical_graph_writes_performed": False,
        "model_calls_performed": False,
    }


def build_lora_rows(*, chunks: list[dict[str, Any]], source_ref: dict[str, Any], limit: int = 256) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    title = str(source_ref.get("title") or source_ref.get("custody_id") or "book")
    for c in chunks[:limit]:
        hits = ", ".join(h["term"] for h in ontology_hits_for_text(c.get("text", ""))[:12]) or "NONE"
        rows.append({
            "instruction": "Extract GO ontology signals from this INDY_READs chunk and cite the chunk id.",
            "input": f"title={title}\nchunk_id={c['chunk_id']}\ntext={c.get('text','')}",
            "output": f"chunk_id={c['chunk_id']} ontology_terms={hits}",
        })
    return rows
