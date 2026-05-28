#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from spine_common import ROOT, now, receipt, rel

INDEX = ROOT / "00_PROJECT_BRAIN" / "KNOWLEDGE_LIBRARY" / "index.json"
REQUIRED = {"id", "title", "authority_class", "source_url", "local_path", "knowledge_card", "tags", "status"}
VALID_AUTHORITY = {"pseudolaw", "research_reference", "candidate_tool"}


def load_index(path: Path = INDEX) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("index_not_object")
    return data


def validate(index: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    entries = index.get("entries")
    if index.get("schema") != "lucidota.knowledge_library.index.v1":
        blockers.append({"kind": "bad_schema", "value": index.get("schema")})
    if not isinstance(entries, list):
        return blockers + [{"kind": "entries_not_list"}], []
    seen: set[str] = set()
    checked: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            blockers.append({"kind": "entry_not_object", "idx": idx})
            continue
        missing = sorted(REQUIRED - set(entry))
        if missing:
            blockers.append({"kind": "entry_missing_fields", "id": entry.get("id"), "missing": missing})
        eid = str(entry.get("id") or "")
        if not eid:
            blockers.append({"kind": "entry_missing_id", "idx": idx})
        elif eid in seen:
            blockers.append({"kind": "duplicate_entry_id", "id": eid})
        seen.add(eid)
        if entry.get("authority_class") not in VALID_AUTHORITY:
            blockers.append({"kind": "bad_authority_class", "id": eid, "value": entry.get("authority_class")})
        for key in ("local_path", "knowledge_card"):
            p = ROOT / str(entry.get(key, ""))
            if not p.exists():
                blockers.append({"kind": f"{key}_missing", "id": eid, "path": rel(p)})
        tags = entry.get("tags")
        if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
            blockers.append({"kind": "bad_tags", "id": eid})
        checked.append({"id": eid, "authority_class": entry.get("authority_class"), "knowledge_card": entry.get("knowledge_card")})
    return blockers, checked


def query(index: dict[str, Any], term: str) -> list[dict[str, Any]]:
    q = term.lower()
    out = []
    for entry in index.get("entries", []):
        blob = json.dumps(entry, sort_keys=True).lower()
        if q in blob:
            out.append(entry)
            continue
        card = ROOT / str(entry.get("knowledge_card", ""))
        if card.exists() and q in card.read_text(encoding="utf-8", errors="ignore").lower():
            out.append(entry)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate/query the LUCIDOTA knowledge library index.")
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--query")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    index_path = Path(args.index)
    if not index_path.is_absolute():
        index_path = ROOT / index_path
    index = load_index(index_path)
    blockers, checked = validate(index)
    matches = query(index, args.query) if args.query else []
    payload = {
        "schema": "lucidota.knowledge_library_check.v1",
        "generated_at": now(),
        "index_path": rel(index_path),
        "entries_checked": len(checked),
        "checked": checked,
        "query": args.query,
        "matches": [{"id": e.get("id"), "title": e.get("title"), "knowledge_card": e.get("knowledge_card")} for e in matches],
        "blockers": blockers,
        "verdict": "PASS" if not blockers else "FAIL",
        "model_calls_performed": False,
        "network_calls_performed": False,
    }
    receipt("knowledge_library_check", payload, root="05_OUTPUTS/knowledge_library")
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    if args.query:
        for match in matches:
            print(f"{match.get('id')} | {match.get('authority_class')} | {match.get('knowledge_card')}")
    print("KNOWLEDGE_LIBRARY_CHECK=" + payload["verdict"])
    return 0 if payload["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
