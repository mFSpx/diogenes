#!/usr/bin/env python3
"""Ahoy ingest sources CLI wrapper for Ahoy receipts."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RAW_CARDS_URL = "https://ledercards.netlify.app/cards.min.json"
CARDS_SEARCH_URL = 'https://cards.ledergames.com/search?q=game:%22ahoy%22%20product:%22ahoybasegame%22'
PLANS_SEARCH_URL = 'https://cards.ledergames.com/search?q=game:%22ahoy%22%20tag:%22Mollusk%20Plan%22'
PDF_PATH = ROOT / "01_REPOS/sharksnailmangame/ahoy rules.pdf"
RAW_DIR = ROOT / "03_VAULT/external/leder_cards"
RULE_DIR = ROOT / "03_VAULT/external/ahoy_rules"
OUT_DIR = ROOT / "05_OUTPUTS/ahoy/rule_audit"
CARD_MANIFEST = ROOT / "06_SCHEMA/ahoy/ahoy_cards_manifest.v1.json"
RULE_SOURCE_MANIFEST = ROOT / "06_SCHEMA/ahoy/ahoy_rule_sources.v1.json"
BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def download(url: str, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(out)
    return out


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def ref_to_int(ref: str) -> int:
    n = 0
    for ch in ref:
        n = n * len(BASE62) + BASE62.index(ch)
    return n


def decode_value(raw: Any, pool: list[Any]) -> Any:
    if not isinstance(raw, str) or "|" not in raw:
        return raw
    kind, *refs = raw.split("|")
    if kind == "a":
        return [decode_value(pool[ref_to_int(r)], pool) for r in refs]
    if kind == "o":
        if not refs:
            return {}
        keys = decode_value(pool[ref_to_int(refs[0])], pool)
        vals = [decode_value(pool[ref_to_int(r)], pool) for r in refs[1:]]
        return {str(k): v for k, v in zip(keys, vals)}
    return raw


def decode_cards_min(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], list):
        raise ValueError("unexpected_cards_min_shape")
    pool = payload[0]
    cards = [decode_value(item, pool) for item in pool if isinstance(item, str) and item.startswith("o|")]
    return [c for c in cards if isinstance(c, dict) and c.get("id")]



def ocr_card_image_if_needed(card: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    text = (card.get("text") or "").strip()
    if text and text != "Effect is too long to list in text!":
        return text, {"text_source": "leder_cards_min_json"}
    image_url = card.get("image")
    card_id = card.get("id") or "unknown"
    if not image_url or not shutil.which("tesseract"):
        return text, {"text_source": "leder_cards_min_json_incomplete", "ocr_attempted": False}
    image_dir = RAW_DIR / "ahoy/images"
    ocr_dir = RAW_DIR / "ahoy/ocr"
    image_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir.mkdir(parents=True, exist_ok=True)
    webp_path = image_dir / f"{card_id}.webp"
    png_path = ocr_dir / f"{card_id}_up.png"
    txt_base = ocr_dir / card_id
    txt_path = ocr_dir / f"{card_id}.txt"
    if not webp_path.exists():
        download(image_url, webp_path)
    try:
        from PIL import Image
        im = Image.open(webp_path).convert("RGB")
        im.resize((im.width * 3, im.height * 3)).save(png_path)
        subprocess.run(["tesseract", str(png_path), str(txt_base), "--psm", "6", "-l", "eng"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ocr_text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if ocr_text:
            return ocr_text, {"text_source": "minimal_ocr_from_card_image", "ocr_image_path": rel(webp_path), "ocr_text_path": rel(txt_path), "ocr_text_sha256": sha256_file(txt_path)}
    except Exception as exc:
        return text, {"text_source": "leder_cards_min_json_incomplete", "ocr_attempted": True, "ocr_error": type(exc).__name__}
    return text, {"text_source": "leder_cards_min_json_incomplete", "ocr_attempted": True, "ocr_error": "empty_ocr"}

def normalize_ahoy_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for c in cards:
        if c.get("game") != "ahoy" or c.get("product") != "ahoybasegame" or c.get("locale") != "en-US":
            continue
        tags = c.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        card_type = "mollusk_plan" if "Mollusk Plan" in tags else "market_crew_or_cargo"
        card_text, text_meta = ocr_card_image_if_needed(c)
        out.append({
            "id": c.get("id"),
            "name": c.get("name") or "",
            "text": card_text,
            "text_meta": text_meta,
            "tags": tags,
            "game": c.get("game"),
            "product": c.get("product"),
            "locale": c.get("locale"),
            "card_type": card_type,
            "image_url": c.get("image"),
            "source_url": f"https://cards.ledergames.com/card/{c.get('id')}",
        })
    def key(card: dict[str, Any]) -> tuple[int, str]:
        digits = "".join(ch for ch in card["id"] if ch.isdigit())
        return (int(digits or 0), card["id"])
    return sorted(out, key=key)


def extract_pdf_text(pdf_path: Path, text_out: Path) -> dict[str, Any]:
    if not pdf_path.exists():
        return {"status": "missing", "path": rel(pdf_path), "blocker": "rules_pdf_missing"}
    text_out.parent.mkdir(parents=True, exist_ok=True)
    method = None
    if shutil.which("pdftotext"):
        subprocess.run(["pdftotext", "-layout", str(pdf_path), str(text_out)], check=True)
        method = "pdftotext_layout"
    else:
        raise RuntimeError("pdftotext_missing_and_ocr_fallback_not_configured")
    text = text_out.read_text(encoding="utf-8", errors="replace") if text_out.exists() else ""
    non_ws = sum(1 for ch in text if not ch.isspace())
    status = "extracted_text_layer" if non_ws > 1000 else "needs_ocr"
    return {
        "status": status,
        "method": method,
        "pdf_path": rel(pdf_path),
        "pdf_sha256": sha256_file(pdf_path),
        "text_path": rel(text_out),
        "text_sha256": sha256_file(text_out),
        "text_chars": len(text),
        "text_non_ws_chars": non_ws,
        "pages_hint": 24,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RULE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_cards = RAW_DIR / "cards.min.json"
    if not args.skip_download or not raw_cards.exists():
        download(RAW_CARDS_URL, raw_cards)
    cards = decode_cards_min(raw_cards)
    ahoy_cards = normalize_ahoy_cards(cards)
    if not ahoy_cards:
        raise RuntimeError("no_ahoy_basegame_cards_decoded")
    card_manifest = {
        "schema": "lucidota.ahoy.cards_manifest.v1",
        "created_at": utc_now(),
        "source": {
            "cards_search_url": CARDS_SEARCH_URL,
            "plans_search_url": PLANS_SEARCH_URL,
            "raw_cards_url": RAW_CARDS_URL,
            "raw_cards_path": rel(raw_cards),
            "raw_cards_sha256": sha256_file(raw_cards),
            "art_asset_policy": "image URLs are recorded as source pointers; image/art assets are not bulk-downloaded. Minimal OCR downloads only cards whose text field is incomplete.",
        },
        "game": "ahoy",
        "product": "ahoybasegame",
        "locale": "en-US",
        "card_count": len(ahoy_cards),
        "mollusk_plan_count": sum(1 for c in ahoy_cards if c["card_type"] == "mollusk_plan"),
        "cards": ahoy_cards,
    }
    write_json_atomic(CARD_MANIFEST, card_manifest)

    rule_text = RULE_DIR / "ahoy_rules_pdftotext.txt"
    pdf_info = extract_pdf_text(PDF_PATH, rule_text)
    rule_source_manifest = {
        "schema": "lucidota.ahoy.rule_sources.v1",
        "created_at": utc_now(),
        "manual": pdf_info,
        "cards_manifest_path": rel(CARD_MANIFEST),
        "cards_manifest_sha256": sha256_file(CARD_MANIFEST),
        "art_asset_policy": "rule fidelity uses text/component data; no image/media art assets required",
    }
    write_json_atomic(RULE_SOURCE_MANIFEST, rule_source_manifest)

    receipt = {
        "schema": "lucidota.ahoy.source_ingest.v1",
        "created_at": utc_now(),
        "verdict": "PASS" if pdf_info.get("status") in {"extracted_text_layer", "ocr_complete"} and len(ahoy_cards) else "FAIL",
        "cards_decoded": len(ahoy_cards),
        "mollusk_plans_decoded": card_manifest["mollusk_plan_count"],
        "card_manifest": rel(CARD_MANIFEST),
        "card_manifest_sha256": sha256_file(CARD_MANIFEST),
        "rule_source_manifest": rel(RULE_SOURCE_MANIFEST),
        "rule_source_manifest_sha256": sha256_file(RULE_SOURCE_MANIFEST),
        "manual_text": pdf_info,
        "files_written": [rel(CARD_MANIFEST), rel(RULE_SOURCE_MANIFEST), rel(raw_cards), rel(rule_text)],
        "source_urls": [RAW_CARDS_URL, CARDS_SEARCH_URL, PLANS_SEARCH_URL],
        "blockers": [] if pdf_info.get("status") in {"extracted_text_layer", "ocr_complete"} else [pdf_info.get("blocker", "manual_text_extraction_failed")],
    }
    receipt_path = OUT_DIR / f"source_ingest_{stamp()}.json"
    write_json_atomic(receipt_path, receipt)
    print(json.dumps({"verdict": receipt["verdict"], "receipt": rel(receipt_path), "cards_decoded": len(ahoy_cards), "mollusk_plans_decoded": card_manifest["mollusk_plan_count"], "manual_status": pdf_info.get("status")}, sort_keys=True))
    return 0 if receipt["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
