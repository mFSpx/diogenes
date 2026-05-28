#!/usr/bin/env python3
"""Validate generated-surface sidecars and HTML affordances.

Enforces the LUCIDOTA surface law: generated surfaces compile interactions into
plain-language command envelopes and must not directly mutate DB/API/canonical state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "surfaces"
DIRECT_MUTATION_PATTERNS = [
    r"fetch\s*\(",
    r"XMLHttpRequest",
    r"indexedDB",
    r"localStorage\.setItem",
    r"navigator\.sendBeacon",
    r"/api/",
    r"direct_db_write\s*[:=]\s*true",
    r"canonical_mutation_allowed\s*[:=]\s*true",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("sidecar_json_must_be_object")
    return data


def validate(sidecar_path: Path, html_path: Path | None) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    data = load_json(sidecar_path)
    if data.get("canonical_mutation_allowed") is not False:
        blockers.append("canonical_mutation_allowed_must_be_false")
    if data.get("conversation_required") is False:
        blockers.append("conversation_required_must_not_be_false")
    interaction_model = str(data.get("interaction_model", ""))
    if "command envelope" not in interaction_model.lower() and "command-envelope" not in interaction_model.lower():
        blockers.append("interaction_model_must_reference_command_envelopes")
    if "direct db" in interaction_model.lower() and "no direct" not in interaction_model.lower():
        blockers.append("interaction_model_mentions_direct_db_without_prohibition")
    if data.get("direct_db_write") is True or data.get("direct_api_call") is True:
        blockers.append("sidecar_direct_write_flags_forbidden")
    if "plain_language_instruction_template" in data and not str(data.get("plain_language_instruction_template") or "").strip():
        blockers.append("plain_language_instruction_template_empty")
    if "command_envelope_schema" in data and "envelope" not in str(data.get("command_envelope_schema", "")).lower():
        blockers.append("command_envelope_schema_malformed")
    html_result: dict[str, Any] = {"html_path": None, "html_sha256": None, "direct_mutation_hits": [], "button_count": 0, "command_envelope_mentions": 0}
    candidate_html = html_path or (ROOT / str(data.get("html_path"))) if data.get("html_path") else None
    if candidate_html:
        candidate_html = candidate_html if candidate_html.is_absolute() else ROOT / candidate_html
        if not candidate_html.exists():
            blockers.append("html_path_missing")
        else:
            text = candidate_html.read_text(encoding="utf-8", errors="replace")
            hits = []
            for pattern in DIRECT_MUTATION_PATTERNS:
                for m in re.finditer(pattern, text, flags=re.I):
                    hits.append({"pattern": pattern, "offset": m.start()})
            html_result.update({
                "html_path": rel(candidate_html),
                "html_sha256": sha_file(candidate_html),
                "direct_mutation_hits": hits,
                "button_count": len(re.findall(r"<button\b", text, flags=re.I)),
                "command_envelope_mentions": len(re.findall(r"command[_ -]?envelope|plain[_ -]?language", text, flags=re.I)),
            })
            if hits:
                blockers.append("html_direct_mutation_pattern_detected")
            if html_result["button_count"] and html_result["command_envelope_mentions"] == 0:
                blockers.append("html_buttons_without_command_envelope_language")
    else:
        warnings.append("no_html_path_to_validate")
    return {
        "sidecar_path": rel(sidecar_path),
        "sidecar_sha256": sha_file(sidecar_path),
        "view_key": data.get("view_key"),
        "surface_id": data.get("surface_id"),
        "html": html_result,
        "blockers": blockers,
        "warnings": warnings,
        "passed": not blockers,
    }


def write_report(name: str, payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"surface_sidecar_validator_{name}_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="Validate generated surface sidecars enforce command-envelope-only interactions")
    p.add_argument("--sidecar", action="append", required=True)
    p.add_argument("--html")
    args = p.parse_args()
    results = []
    for raw in args.sidecar:
        results.append(validate((ROOT / raw).resolve() if not Path(raw).is_absolute() else Path(raw), Path(args.html) if args.html else None))
    passed = all(r["passed"] for r in results)
    report = {"action": "validate", "status": "PASS" if passed else "FAIL", "results": results, "db_writes_performed": False, "graph_writes_performed": False, "blockers": [b for r in results for b in r["blockers"]]}
    write_report("pass" if passed else "fail", report)
    print("SURFACE_SIDECAR_VALIDATION=" + report["status"])
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
