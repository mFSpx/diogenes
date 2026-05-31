#!/usr/bin/env python3
"""ORNAMENT pack loader — reads YAML signal packs → lucidota_go.staging_packet candidates.

Mutation class: candidate_writer
Receipt mode: ABSURD_POSTGRES_RUNTIME
Authority: model_computed_finding (ORNAMENT signal scoring) → staging only, never canonical.

Maps ORNAMENT signal structure to GO-25 term_registry:
  signal         → OBJECT (investigation observable)
  cluster        → EVENT (pattern cluster fire)
  hypothesis     → OBJECT with proposed_term=CLAIM (hypothesis candidate)
  question       → OBJECT with proposed_term=SOURCE (investigator question seam)

Usage:
    python3 scripts/ornament_pack_loader.py [--pack PACK_ID] [--dry-run] [--all]
    python3 scripts/ornament_pack_loader.py --all --dry-run
    python3 scripts/ornament_pack_loader.py --pack vulture
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not available. pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    import psycopg2
    import psycopg2.extras
    _HAS_PG = True
except ImportError:
    _HAS_PG = False

ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = ROOT / "BOOKS" / "ontology_packs" / "ornament"
STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")

# GO-25 + CO/IO term mapping for ORNAMENT dimensions
_DIMENSION_TERM = {
    "financial":     "EVIDENCE",     # financial signals are evidence items
    "real_estate":   "ENTITY",       # properties are entities
    "corporate":     "GROUP",        # corporate actors are groups
    "legal":         "LAW",          # legal signals map to law/rule
    "regulatory":    "REGULATOR",    # regulatory signals map to regulator
    "social":        "RELATIONSHIP", # social signals are relationships
    "community":     "GROUP",        # community actors are groups
    "class":         "ATTRIBUTE",    # class signals are attributes
    "aesthetic":     "SIGNAL",       # aesthetic signals are signals
    "media":         "SOURCE",       # media signals are sources
    "profession":    "ATTRIBUTE",    # profession signals are attributes
    "predatory":     "GRIP",         # IO: GRIP = force/capture pattern
    "displacement":  "SNARE",        # IO: SNARE = systematic trap
}
_DEFAULT_TERM = "ENTITY"


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def _map_term(dimensions: list[str]) -> str:
    """Pick the most specific GO-25 term from signal dimensions."""
    for d in dimensions:
        if d in _DIMENSION_TERM:
            return _DIMENSION_TERM[d]
    return _DEFAULT_TERM


def _pack_id(meta: dict) -> str:
    return meta.get("id", "unknown")


def load_pack(path: Path) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ornament_loader] WARN: failed to load {path}: {e}", file=sys.stderr)
        return None


def pack_to_staging_rows(pack: dict, pack_path: Path) -> list[dict]:
    """Convert one ORNAMENT pack → list of staging_packet row dicts."""
    meta = pack.get("meta", {})
    pack_id = _pack_id(meta)
    pack_name = meta.get("name", pack_id)
    pack_desc = meta.get("description", "")
    rows = []

    # Signals → OBJECT candidates
    for sig in pack.get("signals", []):
        sig_id = sig.get("id", "")
        dims = sig.get("default_dimensions", [])
        weight = sig.get("default_weight", "weak")
        confidence_bps = {"strong": 150, "moderate": 69, "weak": 50}.get(weight, 10)
        term = _map_term(dims)
        source_id = f"ornament:{pack_id}:signal:{sig_id}"
        row = {
            "source_id": source_id,
            "parser_name": "ornament_pack_loader",
            "proposed_term": term,
            "raw_anchor": sig.get("display", sig_id)[:500],
            "claim": f"[{pack_id}] {sig.get('display', sig_id)}"[:2000],
            "proposed_item": json.dumps({
                "ornament_pack_id": pack_id,
                "ornament_pack_name": pack_name,
                "signal_id": sig_id,
                "dimensions": dims,
                "weight": weight,
                "ownership_hint": sig.get("ownership_hint", ""),
                "intentionality_hint": sig.get("intentionality_hint", ""),
                "source_path": str(pack_path),
            }),
            "proposed_edges": json.dumps([]),
            "status": "pending",
            "confidence_bps": confidence_bps,
        }
        rows.append(row)

    # Clusters → EVENT candidates
    for cluster in pack.get("clusters", []):
        cluster_id = cluster.get("id", "")
        source_id = f"ornament:{pack_id}:cluster:{cluster_id}"
        row = {
            "source_id": source_id,
            "parser_name": "ornament_pack_loader",
            "proposed_term": "EVENT",
            "raw_anchor": cluster.get("name", cluster_id)[:500],
            "claim": f"[{pack_id}] cluster: {cluster.get('name', cluster_id)}"[:2000],
            "proposed_item": json.dumps({
                "ornament_pack_id": pack_id,
                "cluster_id": cluster_id,
                "signal_ids": cluster.get("signal_ids", []),
                "min_signals": cluster.get("min_signals", 1),
            }),
            "proposed_edges": json.dumps([]),
            "status": "pending",
            "confidence_bps": 69,
        }
        rows.append(row)

    # Hypotheses → CLAIM candidates
    for hyp in pack.get("hypotheses", []):
        hyp_id = hyp.get("id", "")
        source_id = f"ornament:{pack_id}:hypothesis:{hyp_id}"
        row = {
            "source_id": source_id,
            "parser_name": "ornament_pack_loader",
            "proposed_term": "CLAIM",
            "raw_anchor": hyp.get("statement", hyp_id)[:500],
            "claim": f"[{pack_id}] hypothesis: {hyp.get('statement', hyp_id)}"[:2000],
            "proposed_item": json.dumps({
                "ornament_pack_id": pack_id,
                "hypothesis_id": hyp_id,
                "required_clusters": hyp.get("required_clusters", []),
                "supporting_clusters": hyp.get("supporting_clusters", []),
                "min_cluster_strength": hyp.get("min_cluster_strength", 0.5),
                "category": hyp.get("category", "unknown"),
            }),
            "proposed_edges": json.dumps([]),
            "status": "pending",
            "confidence_bps": 50,
        }
        rows.append(row)

    return rows


INSERT_SQL = """
INSERT INTO lucidota_go.staging_packet
  (source_id, parser_name, proposed_term, raw_anchor, claim,
   proposed_item, proposed_edges, status, confidence_bps)
VALUES
  (%(source_id)s, %(parser_name)s, %(proposed_term)s, %(raw_anchor)s, %(claim)s,
   %(proposed_item)s::jsonb, %(proposed_edges)s::jsonb, %(status)s, %(confidence_bps)s)
"""


def upsert_rows(rows: list[dict], dsn: str) -> int:
    if not _HAS_PG:
        print("ERROR: psycopg2 not available", file=sys.stderr)
        return 0
    conn = psycopg2.connect(dsn)
    try:
        # Fetch existing ornament source_ids so we skip already-staged rows
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source_id FROM lucidota_go.staging_packet WHERE parser_name='ornament_pack_loader'"
            )
            existing = {r[0] for r in cur.fetchall()}
        new_rows = [r for r in rows if r["source_id"] not in existing]
        if not new_rows:
            return 0
        with conn, conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, INSERT_SQL, new_rows, page_size=100)
        return len(new_rows)
    finally:
        conn.close()


def find_packs(pack_id: str | None = None) -> list[Path]:
    if not PACK_DIR.exists():
        print(f"[ornament_loader] PACK_DIR not found: {PACK_DIR}", file=sys.stderr)
        return []
    if pack_id:
        # Search recursively for the named pack
        matches = list(PACK_DIR.rglob(f"*{pack_id}*.yaml"))
        if not matches:
            matches = list(PACK_DIR.rglob(f"{pack_id}_pack.yaml"))
        return matches
    return list(PACK_DIR.rglob("*.yaml"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Load ORNAMENT signal packs → staging_packet candidates")
    parser.add_argument("--pack", help="Pack ID to load (partial match). Default: all.")
    parser.add_argument("--all", action="store_true", help="Load all packs.")
    parser.add_argument("--dry-run", action="store_true", help="Print row count; do not write.")
    parser.add_argument("--dsn", default=STORAGE_DSN, help="Storage DSN.")
    args = parser.parse_args()

    pack_id = args.pack or None
    paths = find_packs(pack_id)
    if not paths:
        print("[ornament_loader] No packs found.", file=sys.stderr)
        return 1

    all_rows = []
    for path in sorted(paths):
        pack = load_pack(path)
        if not pack:
            continue
        rows = pack_to_staging_rows(pack, path)
        all_rows.extend(rows)
        if not args.dry_run:
            print(f"[ornament_loader] {path.name}: {len(rows)} candidates")

    if args.dry_run:
        print(f"[ornament_loader dry-run] packs={len(paths)} total_rows={len(all_rows)}")
        if all_rows:
            print("[ornament_loader dry-run] sample row:")
            sample = {k: v[:100] if isinstance(v, str) and len(v) > 100 else v for k, v in all_rows[0].items()}
            print(json.dumps(sample, indent=2))
        return 0

    n = upsert_rows(all_rows, args.dsn)
    print(f"[receipt] ornament_pack_loader upsert={n} packs={len(paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
