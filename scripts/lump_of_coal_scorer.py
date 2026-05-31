#!/usr/bin/env python3
"""
scripts/lump_of_coal_scorer.py

LUCIDOTA-native port of the LUMP_OF_COAL risk scorer from KRAMPUSCHEWING.

Original: KRAMPUSCHEWING/Lucidota/Lucidota/PROJECTS/KRAMPUS_EXPRESS/LUMP_OF_COAL/
Source modules studied: classifier.py (6-feature composite scorer, 5 risk bands),
                        risk_aggregator.py (keyword-weight triage, 5 bands).

LUCIDOTA adaptation:
- Input: lucidota_go.staging_packet rows (status='pending')
- Scoring features adapted from original 6-signal composite:
    1. confidence_bps         (0-150 range observed; maps to 0-40 pts)
    2. proposed_term weight   (CLAIM/GRIP=3, EVENT/SNARE=3, EVIDENCE/LIE=2,
                               ENTITY=1, GROUP=1, others=0)
    3. claim length signal    (longer claims carry more information density)
    4. parser_name trust      (known high-trust parsers get bonus)
    5. proposed_item density  (non-empty structured extraction bonus)
    6. proposed_edges density (graph edges proposed = evidence of connectivity)
- 4 output bands (0-100 normalized score):
    Watch       0-25   -- routine monitoring
    Investigate 25-50  -- analyst attention warranted
    High        50-75  -- escalate for review
    Priority    75-100 -- immediate operator action

Mutation class: read_only (DB reads) + receipt_only (JSON output, no DB writes).

Usage:
    .venv/bin/python3 scripts/lump_of_coal_scorer.py
    .venv/bin/python3 scripts/lump_of_coal_scorer.py --dry-run
    .venv/bin/python3 scripts/lump_of_coal_scorer.py --limit 200
    .venv/bin/python3 scripts/lump_of_coal_scorer.py --min-score 25
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# -- Psycopg import (available in LUCIDOTA venv) --------------------------------
try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:
    print("ERROR: psycopg not found. Run: .venv/bin/pip install psycopg[binary]", file=sys.stderr)
    sys.exit(1)

# -- Constants ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "risk"

STORAGE_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN", "postgresql:///lucidota_storage")

# proposed_term -> weight (GRIP/SNARE/LIE are abstract aliases for CLAIM/EVENT/EVIDENCE)
TERM_WEIGHTS: dict[str, int] = {
    # Primary high-signal terms
    "CLAIM":    3,  # alias: GRIP
    "EVENT":    3,  # alias: SNARE
    "EVIDENCE": 2,  # alias: LIE
    "ENTITY":   1,
    "GROUP":    1,
    # Abstract aliases (if ever written into DB)
    "GRIP":     3,
    "SNARE":    3,
    "LIE":      2,
}

# Scored term set (what we query from DB)
SCORED_TERMS = {"CLAIM", "EVENT", "EVIDENCE", "ENTITY", "GROUP", "GRIP", "SNARE", "LIE"}

# Parser trust multipliers (0.5-1.0)
PARSER_TRUST: dict[str, float] = {
    "groq_rickshaw_go25_extractor.v1": 1.0,
    "corpus_to_graph":                  0.9,
    "krampus_stage":                    0.85,
    "ornament_pack_loader":             0.8,
}
DEFAULT_PARSER_TRUST = 0.7

# Band thresholds (exclusive lower bound)
BANDS = [
    ("Priority",    75.0),
    ("High",        50.0),
    ("Investigate", 25.0),
    ("Watch",        0.0),
]


# -- Scoring --------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def score_packet(row: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """
    Compute a 0-100 risk score for a single staging_packet row.

    Returns (score, breakdown).

    Feature map (mirroring LUMP_OF_COAL classifier.py signal structure):
      1. confidence_bps   -> 0-40 pts  (primary signal)
      2. term_weight      -> 0-30 pts  (term type importance)
      3. claim_length     -> 0-10 pts  (information density)
      4. parser_trust     -> 0-10 pts  (source reliability)
      5. item_density     -> 0-5  pts  (structured extraction bonus)
      6. edge_density     -> 0-5  pts  (graph connectivity signal)
    Total max: 100 pts.
    """
    breakdown: dict[str, float] = {}

    # 1. Confidence BPS (raw 0-150 observed range → scaled to 0-40)
    raw_bps = int(row.get("confidence_bps") or 0)
    bps_pts = _clamp((raw_bps / 150.0) * 40.0, 0, 40)
    breakdown["confidence_bps"] = round(bps_pts, 2)

    # 2. Term weight (0-30 pts)
    term = str(row.get("proposed_term") or "").upper()
    tw = TERM_WEIGHTS.get(term, 0)
    term_pts = (tw / 3.0) * 30.0
    breakdown["term_weight"] = round(term_pts, 2)

    # 3. Claim length (0-10 pts; caps at 500 chars)
    claim_len = len(str(row.get("claim") or ""))
    len_pts = _clamp((claim_len / 500.0) * 10.0, 0, 10)
    breakdown["claim_length"] = round(len_pts, 2)

    # 4. Parser trust (0-10 pts)
    parser = str(row.get("parser_name") or "")
    trust = PARSER_TRUST.get(parser, DEFAULT_PARSER_TRUST)
    trust_pts = trust * 10.0
    breakdown["parser_trust"] = round(trust_pts, 2)

    # 5. Proposed item density (0-5 pts; non-empty JSON object bonus)
    raw_item = row.get("proposed_item") or {}
    if isinstance(raw_item, str):
        try:
            raw_item = json.loads(raw_item)
        except (json.JSONDecodeError, TypeError):
            raw_item = {}
    item_keys = len(raw_item) if isinstance(raw_item, dict) else 0
    item_pts = _clamp((item_keys / 5.0) * 5.0, 0, 5)
    breakdown["item_density"] = round(item_pts, 2)

    # 6. Proposed edges density (0-5 pts; each edge = 1 pt, capped at 5)
    raw_edges = row.get("proposed_edges") or []
    if isinstance(raw_edges, str):
        try:
            raw_edges = json.loads(raw_edges)
        except (json.JSONDecodeError, TypeError):
            raw_edges = []
    edge_count = len(raw_edges) if isinstance(raw_edges, list) else 0
    edge_pts = _clamp(edge_count * 1.0, 0, 5)
    breakdown["edge_density"] = round(edge_pts, 2)

    total = sum(breakdown.values())
    total = _clamp(total, 0, 100)
    breakdown["total"] = round(total, 2)

    return round(total, 2), breakdown


def classify_band(score: float) -> str:
    for band, threshold in BANDS:
        if score >= threshold:
            return band
    return "Watch"


# -- DB read --------------------------------------------------------------------

def fetch_pending_packets(
    dsn: str,
    terms: set[str],
    limit: int | None,
) -> list[dict[str, Any]]:
    """Read pending staging_packet rows for target terms. Read-only."""
    where_terms = ", ".join(f"'{t}'" for t in sorted(terms))
    sql = f"""
        SELECT
            packet_uuid,
            source_id,
            parser_name,
            proposed_term,
            raw_anchor,
            claim,
            proposed_item,
            proposed_edges,
            status,
            confidence_bps,
            created_at
        FROM lucidota_go.staging_packet
        WHERE status = 'pending'
          AND proposed_term IN ({where_terms})
        ORDER BY created_at DESC
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    return [dict(r) for r in rows]


# -- Receipt writer -------------------------------------------------------------

def write_receipt(payload: dict[str, Any], ts: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"lump_of_coal_{ts}.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    return out_path


# -- Summary printer ------------------------------------------------------------

def print_summary(
    results: list[dict[str, Any]],
    band_counts: dict[str, int],
    receipt_path: Path,
    dry_run: bool,
) -> None:
    total = len(results)
    print()
    print("=" * 60)
    print("LUMP_OF_COAL RISK SCORER — LUCIDOTA PORT")
    print("=" * 60)
    print(f"  Mode      : {'DRY-RUN (no receipt written)' if dry_run else 'LIVE'}")
    print(f"  Packets   : {total}")
    print()
    print("  Band counts:")
    for band, _ in BANDS:
        count = band_counts.get(band, 0)
        bar = "#" * min(count, 40)
        print(f"    {band:<12} {count:>5}  {bar}")
    print()
    if not dry_run:
        print(f"  Receipt   : {receipt_path}")
    print("=" * 60)
    print()

    # Top-10 highest risk
    top = sorted(results, key=lambda r: r["score"], reverse=True)[:10]
    if top:
        print("  Top-10 highest-risk packets:")
        print(f"  {'UUID':<36}  {'Term':<10}  {'Score':>5}  {'Band':<12}  {'Parser'}")
        print("  " + "-" * 82)
        for r in top:
            print(
                f"  {r['packet_uuid']:<36}  {r['proposed_term']:<10}  "
                f"{r['score']:>5.1f}  {r['band']:<12}  {r['parser_name']}"
            )
    print()


# -- Main -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LUMP_OF_COAL risk scorer — LUCIDOTA port. "
                    "Reads staging_packet, scores, writes JSON receipt."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Score and print summary but do not write receipt file."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max rows to fetch (default: all pending)."
    )
    parser.add_argument(
        "--min-score", type=float, default=0.0,
        help="Only include results at or above this score in receipt (default: 0)."
    )
    parser.add_argument(
        "--dsn", default=STORAGE_DSN,
        help="Postgres DSN for lucidota_storage."
    )
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Fetch
    rows = fetch_pending_packets(args.dsn, SCORED_TERMS, args.limit)

    # Score
    results: list[dict[str, Any]] = []
    band_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        score, breakdown = score_packet(row)
        band = classify_band(score)
        band_counts[band] += 1

        if score >= args.min_score:
            results.append({
                "packet_uuid":   str(row.get("packet_uuid", "")),
                "source_id":     str(row.get("source_id", "")),
                "parser_name":   str(row.get("parser_name", "")),
                "proposed_term": str(row.get("proposed_term", "")),
                "raw_anchor":    str(row.get("raw_anchor", ""))[:200],
                "claim_preview": str(row.get("claim", ""))[:200],
                "confidence_bps": int(row.get("confidence_bps") or 0),
                "score":         score,
                "band":          band,
                "breakdown":     breakdown,
                "created_at":    str(row.get("created_at", "")),
            })

    # Sort by score descending
    results.sort(key=lambda r: r["score"], reverse=True)

    # Build receipt payload
    payload: dict[str, Any] = {
        "scorer":           "lump_of_coal_scorer.py",
        "mutation_class":   ["read_only", "receipt_only"],
        "receipt_mode":     "LOCAL_FILE_PRODUCT",
        "scored_at":        ts,
        "source_dsn_env":   "LUCIDOTA_GO_STORAGE_DSN",
        "rows_fetched":     len(rows),
        "rows_in_receipt":  len(results),
        "min_score_filter": args.min_score,
        "band_counts":      dict(band_counts),
        "term_weights":     TERM_WEIGHTS,
        "bands_definition": {b: t for b, t in BANDS},
        "results":          results,
    }

    receipt_path = Path("/dev/null")  # placeholder for dry-run

    if not args.dry_run:
        receipt_path = write_receipt(payload, ts)

    print_summary(results, band_counts, receipt_path, args.dry_run)

    if args.dry_run:
        print("  [DRY-RUN] No receipt written.")
    else:
        print(f"  Receipt written: {receipt_path}")


if __name__ == "__main__":
    main()
