#!/usr/bin/env python3
"""Extract Thick Text reasoning pairs into TRM training data.

Sources (no raw WhatsApp):
1. BC RTB decisions (1,685 records with sections: issues/background/evidence/conclusion)
2. evidence_labels.csv (120,465 rows: fact_id + text + label)
3. heaux_sample.jsonl (977 rows: text + behavioral risk labels)

Will include ciphered WhatsApp data ONLY if found at 05_OUTPUTS/trm_training/ciphered/

Output: 05_OUTPUTS/trm_training/thicktext/
  - train.jsonl (combined training pairs)
  - bcv_rtb.jsonl      (BC RTB reasoning pairs)
  - evidence_labels.jsonl (evidence -> label pairs)
  - heaux.jsonl         (forum text -> behavioral label pairs)
  + receipt
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "05_OUTPUTS" / "trm_training" / "thicktext"
RECEIPT_DIR = PROJECT_ROOT / "05_OUTPUTS" / "trm_training" / "receipts"

# Krampus training directory where source data lives
KRAMPUS_TRAINING_DIR = (
    PROJECT_ROOT
    / "KRAMPUSCHEWING"
    / "Lucidota"
    / "Lucidota"
    / "PROJECTS"
    / "KRAMPUS_EXPRESS"
    / "runtime"
    / "training"
)

CIPHERED_DIR = PROJECT_ROOT / "05_OUTPUTS" / "trm_training" / "ciphered"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── BC RTB Reasoning ──────────────────────────────────────────────

def extract_rtb_reasoning(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a BC RTB decision to a reasoning training pair."""
    decision_id = row.get("id", "unknown")
    sections_raw = row.get("sections", "{}")

    # sections can be a string or dict
    if isinstance(sections_raw, str):
        # Handle truncated/partial JSON strings
        sections_str = sections_raw.strip()
        if sections_str.endswith(","):
            sections_str = sections_str[:-1]
        if sections_str.endswith("'"):
            sections_str = sections_str[:-1]
        try:
            sections = json.loads(sections_str)
        except json.JSONDecodeError:
            sections = {"issues": sections_str[:2000], "background": "", "evidence": "", "conclusion": ""}
    else:
        sections = sections_raw

    issue = sections.get("issues", "")
    background = sections.get("background", "")
    evidence_raw = sections.get("evidence", "")
    conclusion = sections.get("conclusion", "")

    # Also check full_text for evidence if sections truncated
    full_text = row.get("full_text", "")

    # Truncate long texts
    issue = issue[:3000]
    background = background[:3000]
    evidence = evidence_raw[:3000] if evidence_raw else full_text[:3000]
    conclusion = conclusion[:2000] if conclusion else ""

    if not issue and not background and not evidence and not conclusion:
        return None

    # Build reasoning pair: Background + Evidence -> Question -> Answer
    reasoning_text = ""
    if background:
        reasoning_text += f"Background: {background}\n"
    if evidence:
        reasoning_text += f"Evidence: {evidence}\n"
    reasoning_text += f"Question: {issue}\n"
    reasoning_text += f"Answer: {conclusion}"

    if not reasoning_text.strip():
        return None

    # Additional metadata
    outcome = row.get("outcome", "UNKNOWN")
    dispute_type = row.get("dispute_type", "")
    decision_date = row.get("decision_date", "")
    claim_codes = row.get("claim_codes", "")

    return {
        "id": f"rtb_{decision_id}",
        "source": "bc_rtb_reasoning",
        "text": reasoning_text,
        "content_hash": _sha256_text(reasoning_text),
        "decision_id": decision_id,
        "outcome": outcome,
        "dispute_type": dispute_type,
        "decision_date": decision_date,
        "claim_codes": claim_codes,
        "narrative_type": "legal_reasoning",
    }


# ── Evidence Labels ───────────────────────────────────────────────

def extract_evidence_label(row: dict[str, str]) -> dict[str, Any] | None:
    """Convert an evidence label row to a classification pair."""
    fact_id = row.get("fact_id", "")
    text = row.get("text", "")
    label = row.get("label", "")

    if not text or not label:
        return None

    # Clean label
    label = label.strip()

    return {
        "id": f"evid_{fact_id}",
        "source": "evidence_labels",
        "text": f"Evidence: {text[:2000]}\nLabel: {label}",
        "content_hash": _sha256_text(text),
        "fact_id": fact_id,
        "label": label,
        "narrative_type": "evidence_classification",
    }


def iter_evidence_csv(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    """Read evidence_labels.csv, handling potential encoding issues."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        print(f"  WARNING: not found: {path}", file=sys.stderr)
        return rows

    print(f"  Reading {path}...", file=sys.stderr)
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append(row)
                if limit and i + 1 >= limit:
                    break
    except UnicodeDecodeError:
        with open(path, newline="", encoding="latin-1") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append(row)
                if limit and i + 1 >= limit:
                    break
    except Exception as e:
        print(f"  ERROR reading CSV: {e}", file=sys.stderr)
    return rows


# ── Heaux Sample ──────────────────────────────────────────────────

def extract_heaux(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a heaux sample to a behavioral label pair."""
    text = row.get("text", "")
    signal = row.get("signal", "")
    source_section = row.get("source_section", "")
    row_type = row.get("type", "")

    if not text:
        return None

    text_content = text[:2000]
    row_id = f"heaux_{_sha256_text(text_content)[:12]}"

    return {
        "id": row_id,
        "source": "heaux_behavioral",
        "text": f"Content: {text_content}\nSignal: {signal}\nSource: {source_section}",
        "content_hash": _sha256_text(text_content),
        "signal": signal,
        "source_section": source_section,
        "type": row_type,
        "narrative_type": "behavioral_labeling",
    }


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all JSON objects from a JSONL file."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        print(f"  WARNING: not found: {path}", file=sys.stderr)
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(rows: list[dict], path: Path) -> int:
    """Write list of dicts to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            written += 1
    return written


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Extract Thick Text reasoning data for TRM training"
    )
    ap.add_argument("--rtb-path", default="",
                    help="Path to BC_RTB_TRAINING_DATA_2026.jsonl")
    ap.add_argument("--evidence-path", default="",
                    help="Path to evidence_labels.csv")
    ap.add_argument("--heaux-path", default="",
                    help="Path to heaux_sample.jsonl")
    ap.add_argument("--include-ciphered", action="store_true",
                    help="Include ciphered WhatsApp data from ciphered/ dir")
    ap.add_argument("--rtb-limit", type=int, default=0,
                    help="Limit RTB rows (0=all)")
    ap.add_argument("--evidence-limit", type=int, default=0,
                    help="Limit evidence label rows (0=all)")
    ap.add_argument("--heaux-limit", type=int, default=0,
                    help="Limit heaux rows (0=all)")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="Print discovery info without writing (default)")
    ap.add_argument("--execute", action="store_true",
                    help="Actually write training files")
    ap.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args()

    now_utc = datetime.now(timezone.utc).isoformat()

    # ── Resolve paths ──────────────────────────────────────────
    rtb_path = Path(args.rtb_path) if args.rtb_path else \
        KRAMPUS_TRAINING_DIR / "BC_RTB_TRAINING_DATA_2026.jsonl"
    evidence_path = Path(args.evidence_path) if args.evidence_path else \
        KRAMPUS_TRAINING_DIR / "evidence_labels.csv"
    heaux_path = Path(args.heaux_path) if args.heaux_path else \
        KRAMPUS_TRAINING_DIR / "heaux_sample.jsonl"

    print("TRM Thick Text Extraction")
    print(f"  RTB:  {rtb_path}")
    print(f"  Evidence: {evidence_path}")
    print(f"  Heaux: {heaux_path}")

    # ── 1. RTB Reasoning ──────────────────────────────────────
    rtb_pairs: list[dict[str, Any]] = []
    if rtb_path.exists():
        raw_rtb = iter_jsonl(rtb_path)
        if args.rtb_limit:
            raw_rtb = raw_rtb[:args.rtb_limit]
        for r in raw_rtb:
            pair = extract_rtb_reasoning(r)
            if pair:
                rtb_pairs.append(pair)
        print(f"  RTB: {len(rtb_pairs)} reasoning pairs from {len(raw_rtb)} rows")
    else:
        print(f"  RTB: NOT FOUND (skipping)")

    # ── 2. Evidence Labels ────────────────────────────────────
    evidence_pairs: list[dict[str, Any]] = []
    if evidence_path.exists():
        raw_evidence = iter_evidence_csv(evidence_path, limit=args.evidence_limit)
        for r in raw_evidence:
            pair = extract_evidence_label(r)
            if pair:
                evidence_pairs.append(pair)
        print(f"  Evidence: {len(evidence_pairs)} pairs from {len(raw_evidence)} rows")
    else:
        print(f"  Evidence: NOT FOUND (skipping)")

    # ── 3. Heaux Sample ───────────────────────────────────────
    heaux_pairs: list[dict[str, Any]] = []
    if heaux_path.exists():
        raw_heaux = iter_jsonl(heaux_path)
        if args.heaux_limit:
            raw_heaux = raw_heaux[:args.heaux_limit]
        for r in raw_heaux:
            pair = extract_heaux(r)
            if pair:
                heaux_pairs.append(pair)
        print(f"  Heaux: {len(heaux_pairs)} pairs from {len(raw_heaux)} rows")
    else:
        print(f"  Heaux: NOT FOUND (skipping)")

    # ── 4. Ciphered WhatsApp (optional) ───────────────────────
    ciphered_pairs: list[dict[str, Any]] = []
    if args.include_ciphered and CIPHERED_DIR.exists():
        for f in sorted(CIPHERED_DIR.glob("*.jsonl")):
            raw_ciphered = iter_jsonl(f)
            for r in raw_ciphered:
                # Expect ciphered format to have text + label fields
                text = r.get("text", r.get("content", ""))
                label = r.get("label", r.get("signal", "conversation"))
                if text:
                    ciphered_pairs.append({
                        "id": f"ciphered_{_sha256_text(text[:100])[:12]}",
                        "source": "ciphered_whatsapp",
                        "text": f"Message: {text[:2000]}\nLabel: {label}",
                        "content_hash": _sha256_text(text),
                        "label": label,
                        "narrative_type": "ciphered_conversation",
                    })
            print(f"  Ciphered ({f.name}): {len(ciphered_pairs)} pairs")
    else:
        if not CIPHERED_DIR.exists():
            print(f"  Ciphered: dir not found ({CIPHERED_DIR}) — skipping")
        elif not args.include_ciphered:
            print(f"  Ciphered: excluded (use --include-ciphered to include)")

    # Combine all
    all_pairs = rtb_pairs + evidence_pairs + heaux_pairs + ciphered_pairs
    total = len(all_pairs)
    narrative_types = Counter(p["narrative_type"] for p in all_pairs)

    print(f"\n  Total: {total} training pairs across {len(narrative_types)} narrative types")
    for ntype, count in narrative_types.most_common():
        print(f"    {ntype}: {count}")

    if not all_pairs:
        print("  ERROR: No training pairs generated.", file=sys.stderr)
        sys.exit(1)

    # ── Write or dry-run ──────────────────────────────────────
    out_dir = Path(args.output_dir)

    if args.dry_run and not args.execute:
        print(f"\n  DRY RUN — use --execute to write {total} pairs")

        receipt = {
            "schema": "lucidota.trm.thicktext_extraction_receipt.v1",
            "verdict": "DRY_RUN",
            "sources": {
                "rtb": str(rtb_path) if rtb_path.exists() else None,
                "evidence": str(evidence_path) if evidence_path.exists() else None,
                "heaux": str(heaux_path) if heaux_path.exists() else None,
                "ciphered": str(CIPHERED_DIR) if CIPHERED_DIR.exists() else None,
            },
            "training_pairs_prepared": {
                "rtb_reasoning": len(rtb_pairs),
                "evidence_classification": len(evidence_pairs),
                "behavioral_labeling": len(heaux_pairs),
                "ciphered_whatsapp": len(ciphered_pairs),
                "total": total,
            },
            "narrative_type_distribution": dict(narrative_types),
            "ciphered_included": args.include_ciphered and CIPHERED_DIR.exists(),
            "created_at_utc": now_utc,
            "command": " ".join(sys.argv),
        }
        receipt_path = RECEIPT_DIR / "thicktext_extract_receipt.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = receipt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        tmp.replace(receipt_path)
        print(f"  Receipt (dry-run): {receipt_path}")
        return

    # Write individual files
    out_dir.mkdir(parents=True, exist_ok=True)
    files_written = []

    if rtb_pairs:
        n = write_jsonl(rtb_pairs, out_dir / "bcv_rtb.jsonl")
        files_written.append(str(out_dir / "bcv_rtb.jsonl"))
        print(f"  Wrote {n} RTB pairs to {out_dir / 'bcv_rtb.jsonl'}")

    if evidence_pairs:
        n = write_jsonl(evidence_pairs, out_dir / "evidence_labels.jsonl")
        files_written.append(str(out_dir / "evidence_labels.jsonl"))
        print(f"  Wrote {n} evidence pairs to {out_dir / 'evidence_labels.jsonl'}")

    if heaux_pairs:
        n = write_jsonl(heaux_pairs, out_dir / "heaux.jsonl")
        files_written.append(str(out_dir / "heaux.jsonl"))
        print(f"  Wrote {n} heaux pairs to {out_dir / 'heaux.jsonl'}")

    if ciphered_pairs:
        n = write_jsonl(ciphered_pairs, out_dir / "ciphered.jsonl")
        files_written.append(str(out_dir / "ciphered.jsonl"))
        print(f"  Wrote {n} ciphered pairs to {out_dir / 'ciphered.jsonl'}")

    # Write combined train.jsonl
    combined_path = out_dir / "train.jsonl"
    n_combined = write_jsonl(all_pairs, combined_path)
    files_written.append(str(combined_path))
    print(f"  Wrote {n_combined} combined pairs to {combined_path}")

    # Source file hashes
    source_files = [rtb_path, evidence_path, heaux_path]
    source_hashes = {}
    for s in source_files:
        if s.exists():
            source_hashes[str(s)] = _sha256_file(s)
    if CIPHERED_DIR.exists():
        for f in sorted(CIPHERED_DIR.glob("*.jsonl")):
            source_hashes[str(f)] = _sha256_file(f)

    # Receipt
    receipt = {
        "schema": "lucidota.trm.thicktext_extraction_receipt.v1",
        "verdict": "PASS" if total > 0 else "FAIL",
        "sources": source_hashes,
        "source_sha256": source_hashes,
        "training_pairs_written": {
            "rtb_reasoning": len(rtb_pairs),
            "evidence_classification": len(evidence_pairs),
            "behavioral_labeling": len(heaux_pairs),
            "ciphered_whatsapp": len(ciphered_pairs),
            "total": total,
        },
        "narrative_type_distribution": dict(narrative_types),
        "files_written": files_written,
        "ciphered_included": args.include_ciphered and CIPHERED_DIR.exists(),
        "created_at_utc": now_utc,
        "processed_at_utc": now_utc,
        "verified_at_utc": now_utc,
        "output_sha256": _sha256_file(combined_path) if total > 0 else None,
        "command": " ".join(sys.argv),
    }
    receipt_path = RECEIPT_DIR / "thicktext_extract_receipt.json"
    tmp = receipt_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    tmp.replace(receipt_path)

    print(f"\n  Receipt: {receipt_path}")
    print(f"  Done. {total} Thick Text training pairs extracted.")


if __name__ == "__main__":
    main()
