#!/usr/bin/env python3
"""Ahoy rule audit CLI wrapper for Ahoy receipts."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ahoy_sim.engine.receipts import OUT, sha256_file, stamp, utc_now, write_json_receipt
from ahoy_sim.rules.rule_gaps import all_gaps, blocking_gaps
from ahoy_sim.rules.rule_manifest import MANIFEST_PATH, load_manifest
CARD_MANIFEST = ROOT / "06_SCHEMA/ahoy/ahoy_cards_manifest.v1.json"
RULE_SOURCE_MANIFEST = ROOT / "06_SCHEMA/ahoy/ahoy_rule_sources.v1.json"
EXTRACTED_RULE_TEXT = ROOT / "03_VAULT/external/ahoy_rules/ahoy_rules_pdftotext.txt"

SCRAP_CANDIDATES = [ROOT / "01_REPOS/sharksnailmangame", Path("/repos/sharksnailmangame")]
MANUAL_SEARCH_ROOTS = [ROOT / "01_REPOS/sharksnailmangame", ROOT / "03_VAULT/external/ahoy_rules", ROOT / "06_SCHEMA/ahoy"]
SKIP_ASSET_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".svg", ".mp4", ".mov", ".pdf"}


def inventory_sources() -> dict:
    files = []
    for base in SCRAP_CANDIDATES:
        if base.exists():
            for path in sorted(base.glob("*")):
                if path.is_file() and path.suffix.lower() not in SKIP_ASSET_SUFFIXES:
                    files.append({"path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manual_hits = []
    for root in MANUAL_SEARCH_ROOTS:
        if not root.exists():
            continue
        for pat in ["*ahoy*rule*", "*Ahoy*Rule*", "*manual*", "*Manual*"]:
            manual_hits.extend(sorted(root.rglob(pat)))
    manual_hits = [p for p in manual_hits if p.is_file() and p.suffix.lower() not in SKIP_ASSET_SUFFIXES]
    return {
        "actual_scrap_path": "01_REPOS/sharksnailmangame" if (ROOT / "01_REPOS/sharksnailmangame").exists() else None,
        "operator_provided_abs_path_exists": Path("/repos/sharksnailmangame").exists(),
        "local_scrap_files": files,
        "official_like_manual_candidates": [str(p.relative_to(ROOT)) for p in manual_hits[:50]],
        "official_manual_found": EXTRACTED_RULE_TEXT.exists(),
        "official_card_manifest_found": CARD_MANIFEST.exists(),
        "card_manifest_path": str(CARD_MANIFEST.relative_to(ROOT)) if CARD_MANIFEST.exists() else None,
        "card_manifest_sha256": sha256_file(CARD_MANIFEST) if CARD_MANIFEST.exists() else None,
        "rule_source_manifest_path": str(RULE_SOURCE_MANIFEST.relative_to(ROOT)) if RULE_SOURCE_MANIFEST.exists() else None,
        "rule_source_manifest_sha256": sha256_file(RULE_SOURCE_MANIFEST) if RULE_SOURCE_MANIFEST.exists() else None,
        "extracted_rule_text_path": str(EXTRACTED_RULE_TEXT.relative_to(ROOT)) if EXTRACTED_RULE_TEXT.exists() else None,
        "extracted_rule_text_sha256": sha256_file(EXTRACTED_RULE_TEXT) if EXTRACTED_RULE_TEXT.exists() else None,
        "asset_policy": "rule audit hashes text/code/component manifests only; image/media art assets are skipped except minimal OCR for incomplete card text",
        "skipped_asset_suffixes": sorted(SKIP_ASSET_SUFFIXES),
        "source_law": "local scraps are correction hints; official/card text/component gaps remain blocking",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    manifest = load_manifest()
    gaps = blocking_gaps()
    source_inventory = inventory_sources()
    receipt = {
        "schema": "lucidota.ahoy.rule_audit.v1",
        "created_at": utc_now(),
        "verdict": "BLOCKED" if gaps else "PASS",
        "rules_verdict": "BLOCKED" if gaps else "PASS",
        "manifest_path": str(MANIFEST_PATH.relative_to(ROOT)),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "implemented_rule_count": len(manifest.get("rules", [])),
        "source_inventory": source_inventory,
        "blocking_rule_gaps": gaps,
        "all_rule_gaps": all_gaps(),
        "blockers": [g["gap_id"] for g in gaps],
    }
    out = args.out or (OUT / "rule_audit" / f"rule_audit_{stamp()}.json")
    write_json_receipt(out, receipt)
    gaps_out = OUT / "rule_audit" / f"rule_gaps_{stamp()}.json"
    write_json_receipt(gaps_out, {"schema": "lucidota.ahoy.rule_gaps.v1", "created_at": utc_now(), "gaps": all_gaps(), "blocking_rule_gaps": gaps})
    print(json.dumps({"verdict": receipt["verdict"], "receipt": str(out), "gaps_receipt": str(gaps_out), "blocking_gaps": len(gaps)}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
