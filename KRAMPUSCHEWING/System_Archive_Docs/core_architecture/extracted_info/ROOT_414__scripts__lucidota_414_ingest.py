#!/usr/bin/env python3
"""Build a machine-clean ROOT-414 manifest/knowledge graph from 414_PRIMITIVE_CRIES.

Local-first: no Postgres required. Produces:
- 05_OUTPUTS/root414_primitive_cries_manifest.json
- 05_OUTPUTS/root414_primitive_cries_packets.jsonl
- 04_RUNTIME/root414_knowledge.sqlite
- 00_PROJECT_BRAIN/414_PRIMITIVE_CRIES/000_INDEX.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CRIES = ROOT / "00_PROJECT_BRAIN" / "414_PRIMITIVE_CRIES"
OUT = ROOT / "05_OUTPUTS"
RUNTIME = ROOT / "04_RUNTIME"
INDEX_MD = CRIES / "000_INDEX.md"
DB = RUNTIME / "root414_knowledge.sqlite"

# Import the canonical global ontology without requiring package install.
import sys
sys.path.insert(0, str(ROOT / "01_REPOS" / "doggystyle"))
from kernel.global_ontology import (  # type: ignore  # noqa: E402
    GLOBAL_ONTOLOGY_BLOCK_BY_SYMBOL,
    GLOBAL_ONTOLOGY_BY_SYMBOL,
    GLOBAL_ONTOLOGY_SYMBOLS,
)

SYMBOL_SET = set(GLOBAL_ONTOLOGY_SYMBOLS)
ORD_TO_SYMBOL = {i: s for i, s in enumerate(GLOBAL_ONTOLOGY_SYMBOLS, start=1)}
TOKEN_RE = re.compile(r"\b[A-Z][A-Z0-9_?]{2,}\b")
ORD_RE = re.compile(r"(?<!\d)(\d{3})(?!\d)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)
PACKET_RE = re.compile(r"\bpacket_id\s*:\s*([A-Za-z0-9_.:-]+)")
BPS_RE = re.compile(r"(?<!\d)(0|2|4|6|10|50|69|150)\s*bps\b", re.I)

CANONICAL_BPS = {0, 2, 4, 6, 10, 50, 69, 150}

@dataclass
class PrimitiveRef:
    ordinal: int
    symbol: str
    block_name: str
    count: int

@dataclass
class CryDoc:
    doc_id: str
    path: str
    title: str
    sha256: str
    size_bytes: int
    line_count: int
    headings: list[str]
    packet_ids: list[str]
    bps_values: list[int]
    primitive_refs: list[PrimitiveRef]
    flags: list[str]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def title_from_text(path: Path, text: str) -> str:
    m = HEADING_RE.search(text)
    if m:
        return m.group(2).strip()
    return path.stem.replace("_", " ").title()


def doc_id_from_path(path: Path) -> str:
    return path.stem


def extract_primitives(text: str) -> list[PrimitiveRef]:
    counts: dict[str, int] = {}
    for tok in TOKEN_RE.findall(text):
        if tok in SYMBOL_SET:
            counts[tok] = counts.get(tok, 0) + 1
    # Also capture ordinal refs, but only add symbol refs where symbol not already present.
    for raw in ORD_RE.findall(text):
        n = int(raw)
        if 1 <= n <= 414:
            sym = ORD_TO_SYMBOL[n]
            # ordinal-only mentions are lower confidence; still count as one ref.
            counts[sym] = counts.get(sym, 0) + 1
    refs = [
        PrimitiveRef(
            ordinal=GLOBAL_ONTOLOGY_BY_SYMBOL[sym],
            symbol=sym,
            block_name=GLOBAL_ONTOLOGY_BLOCK_BY_SYMBOL[sym],
            count=count,
        )
        for sym, count in counts.items()
    ]
    return sorted(refs, key=lambda r: (r.ordinal, r.symbol))


def flags_for_doc(text: str, bps_values: list[int], packet_ids: list[str], refs: list[PrimitiveRef]) -> list[str]:
    flags: list[str] = []
    lowered = text.lower()
    if "falsifier" not in lowered and any(word in lowered for word in ["route", "parser", "packet", "claim"]):
        flags.append("NO_FALSIFIER_VISIBLE")
    all_bps = [int(x) for x in re.findall(r"(?<!\d)(\d{1,5})\s*bps\b", text, re.I)]
    noncanonical = sorted({v for v in all_bps if v not in CANONICAL_BPS})
    if noncanonical:
        flags.append("NONCANONICAL_BPS:" + ",".join(map(str, noncanonical[:10])))
    if "ARCHONIC_CONTROL_GRID" in text and not any(g.symbol in {"DOCUMENT_EXAMINATION", "PLAUSIBILITY_GATE", "SOURCE_INDEPENDENCE", "CHAIN_OF_CUSTODY", "TEMPORAL_PRECEDENCE"} for g in refs):
        flags.append("POSSIBLE_HIGH_LABEL_GRAVITY")
    if "packet_id" in text and not packet_ids:
        flags.append("PACKET_PARSE_FAILED")
    return flags


def build_docs(root: Path) -> list[CryDoc]:
    docs: list[CryDoc] = []
    for path in sorted(root.glob("*.md")):
        if path.name == "000_INDEX.md":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        refs = extract_primitives(text)
        packet_ids = sorted(set(PACKET_RE.findall(text)))
        bps_values = sorted({int(v) for v in BPS_RE.findall(text)})
        headings = [m.group(2).strip() for m in HEADING_RE.finditer(text)]
        docs.append(
            CryDoc(
                doc_id=doc_id_from_path(path),
                path=path.relative_to(ROOT).as_posix(),
                title=title_from_text(path, text),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                line_count=text.count("\n") + 1,
                headings=headings[:40],
                packet_ids=packet_ids,
                bps_values=bps_values,
                primitive_refs=refs,
                flags=flags_for_doc(text, bps_values, packet_ids, refs),
            )
        )
    return docs


def write_outputs(docs: list[CryDoc]) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "root414_primitive_cries_manifest.json"
    jsonl_path = OUT / "root414_primitive_cries_packets.jsonl"

    primitive_totals: dict[str, int] = {}
    for d in docs:
        for ref in d.primitive_refs:
            primitive_totals[ref.symbol] = primitive_totals.get(ref.symbol, 0) + ref.count
    top_primitives = sorted(primitive_totals.items(), key=lambda kv: (-kv[1], GLOBAL_ONTOLOGY_BY_SYMBOL[kv[0]]))[:40]

    manifest = {
        "ok": True,
        "parser_name": "root414_machine_clean_parser_v0.50",
        "source_dir": CRIES.relative_to(ROOT).as_posix(),
        "doc_count": len(docs),
        "primitive_coverage_count": len(primitive_totals),
        "top_primitives": [
            {"symbol": sym, "ordinal": GLOBAL_ONTOLOGY_BY_SYMBOL[sym], "count": count}
            for sym, count in top_primitives
        ],
        "docs": [
            {
                **{k: v for k, v in asdict(d).items() if k != "primitive_refs"},
                "primitive_refs": [asdict(r) for r in d.primitive_refs],
            }
            for d in docs
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for d in docs:
            packet = {
                "packet_id": f"cry::{d.doc_id}",
                "source_id": d.path,
                "parser_name": "root414_machine_clean_parser_v0.50",
                "timestamp": "2026-05-14",
                "raw_text_anchor": d.title,
                "evidence_units": [
                    {"evidence_id": f"{d.doc_id}::file", "quote_or_fact": d.title, "channel": "text"}
                ],
                "routes": [
                    {
                        "route_id": f"{d.doc_id}::mentions",
                        "operator": "mixed",
                        "primitives": [asdict(r) for r in d.primitive_refs[:32]],
                        "resolution": "ROOT414_PRIMITIVE_CRY_IMPORTED",
                        "local_gates": [],
                        "route_risk": "clean" if not d.flags else "underfit",
                    }
                ],
                "ternary_state": {"inside_scope": 1, "outside_scope": 0},
                "claim_lifecycle": "CLAIM_UNVERIFIED",
                "confidence_bps": 10,
                "falsifier": "Document is removed, hash changes, or HITL rejects import classification.",
                "rejected_routes": [],
                "indy_note": "Manifest import; semantic packet extraction pending deeper parser run.",
                "river_note": "Use as graph seed, not deterministic semantic proof.",
                "hitl_status": "pending",
                "flags": d.flags,
            }
            fh.write(json.dumps(packet, sort_keys=True) + "\n")
    write_sqlite(docs)
    write_index(docs, manifest)
    return {"manifest": str(manifest_path), "jsonl": str(jsonl_path), "sqlite": str(DB), "index": str(INDEX_MD), "doc_count": len(docs), "primitive_coverage_count": len(primitive_totals)}


def write_sqlite(docs: list[CryDoc]) -> None:
    if DB.exists():
        DB.unlink()
    conn = sqlite3.connect(DB)
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE doc(
          doc_id TEXT PRIMARY KEY,
          path TEXT NOT NULL,
          title TEXT NOT NULL,
          sha256 TEXT NOT NULL,
          size_bytes INTEGER NOT NULL,
          line_count INTEGER NOT NULL,
          flags_json TEXT NOT NULL
        );
        CREATE TABLE primitive(
          ordinal INTEGER PRIMARY KEY,
          symbol TEXT UNIQUE NOT NULL,
          block_name TEXT NOT NULL
        );
        CREATE TABLE doc_primitive(
          doc_id TEXT NOT NULL,
          ordinal INTEGER NOT NULL,
          symbol TEXT NOT NULL,
          count INTEGER NOT NULL,
          PRIMARY KEY(doc_id, ordinal)
        );
        CREATE TABLE packet_seed(
          packet_id TEXT PRIMARY KEY,
          doc_id TEXT NOT NULL,
          hitl_status TEXT NOT NULL,
          confidence_bps INTEGER NOT NULL,
          packet_json TEXT NOT NULL
        );
        """
    )
    for ordinal, symbol in enumerate(GLOBAL_ONTOLOGY_SYMBOLS, start=1):
        conn.execute("INSERT INTO primitive VALUES(?,?,?)", (ordinal, symbol, GLOBAL_ONTOLOGY_BLOCK_BY_SYMBOL[symbol]))
    for d in docs:
        conn.execute("INSERT INTO doc VALUES(?,?,?,?,?,?,?)", (d.doc_id, d.path, d.title, d.sha256, d.size_bytes, d.line_count, json.dumps(d.flags)))
        for r in d.primitive_refs:
            conn.execute("INSERT INTO doc_primitive VALUES(?,?,?,?)", (d.doc_id, r.ordinal, r.symbol, r.count))
        conn.execute("INSERT INTO packet_seed VALUES(?,?,?,?,?)", (f"cry::{d.doc_id}", d.doc_id, "pending", 10, json.dumps({"source_id": d.path, "title": d.title, "flags": d.flags}, sort_keys=True)))
    conn.commit(); conn.close()


def write_index(docs: list[CryDoc], manifest: dict[str, Any]) -> None:
    lines = ["# ROOT-414 Primitive Cries Index", "", "Generated by `scripts/lucidota_414_ingest.py`.", ""]
    lines.append(f"- Documents: `{len(docs)}`")
    lines.append(f"- Primitive coverage: `{manifest['primitive_coverage_count']}` / 414")
    lines.append(f"- Active parser: `root414_machine_clean_parser_v0.50`")
    lines.append("")
    lines.append("## Top Primitive Mentions")
    lines.append("")
    for item in manifest["top_primitives"][:20]:
        lines.append(f"- `{item['symbol']} ({item['ordinal']:03d})` — {item['count']} mentions")
    lines.append("")
    lines.append("## Documents")
    lines.append("")
    for d in docs:
        flag = f" ⚑ {', '.join(d.flags)}" if d.flags else ""
        prims = ", ".join(f"`{r.symbol}`" for r in d.primitive_refs[:6])
        lines.append(f"- [`{d.path}`](../../{d.path}) — {d.title}{flag}")
        if prims:
            lines.append(f"  - primitives: {prims}")
    INDEX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=CRIES)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    docs = build_docs(args.source)
    report = write_outputs(docs)
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
