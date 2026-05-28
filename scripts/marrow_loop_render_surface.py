#!/usr/bin/env python3
"""Render Marrow Loop status surface from append-only marrow_state.md."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "marrow_loop"
HTML_PATH = ROOT / "07_SURFACES" / "generated" / "marrow_loop_status.html"
SIDECAR_PATH = ROOT / "07_SURFACES" / "sidecars" / "marrow_loop_status.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Marrow Loop generated surface")
    parser.add_argument("--state", default="05_OUTPUTS/marrow_loop/marrow_state.md")
    parser.add_argument("--dry-run", action="store_true", help="write report only; do not rewrite HTML/sidecar")
    args = parser.parse_args()
    state_path = Path(args.state)
    if not state_path.is_absolute():
        state_path = ROOT / state_path

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []
    generated = False
    state_sha = None
    html_sha = None

    if not state_path.exists():
        blockers.append("state_file_missing")
    else:
        state_text = state_path.read_text(encoding="utf-8")
        state_sha = sha256_file(state_path)
        envelope = {
            "protocol": "lucidota.command_envelope.v1",
            "intent": "marrow_loop.surface_action",
            "source_surface": "marrow_loop_status",
            "direct_db_write": False,
            "direct_api_call": False,
        }
        escaped_state = html.escape(state_text)
        escaped_envelope = html.escape(json.dumps(envelope, sort_keys=True, separators=(",", ":")))
        html_text = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Marrow Loop Status</title>
<style>body{{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:2rem}}pre{{white-space:pre-wrap;background:#1d1d1d;padding:1rem}}button{{padding:.5rem;margin:.25rem}}</style></head>
<body><h1>Marrow Loop Status</h1>
<p>Interaction model: Buttons emit command envelopes; no direct DB/API mutation.</p>
<button onclick=\"document.getElementById('envelope').hidden=false\">Show Command Envelope JSON</button>
<pre id=\"envelope\" hidden>{escaped_envelope}</pre>
<h2>State</h2><pre>{escaped_state}</pre>
</body></html>\n"""
        if not args.dry_run:
            HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
            SIDECAR_PATH.parent.mkdir(parents=True, exist_ok=True)
            HTML_PATH.write_text(html_text, encoding="utf-8")
            html_sha = sha256_file(HTML_PATH)
            sidecar = {
                "view_key": "marrow_loop_status",
                "title": "Marrow Loop Status",
                "source_state_path": str(state_path.relative_to(ROOT)),
                "state_sha256": state_sha,
                "html_sha256": html_sha,
                "generated_at": utc_now(),
                "why_it_exists": "Shows append-only Marrow Loop command state as a reusable generated surface.",
                "reuse_policy": "Read-only projection; regenerate from marrow_state.md. Declarative metadata only until promotion/pheromone lifecycle is wired.",
                "interaction_model": "Buttons emit command envelopes; no direct DB/API mutation.",
                "surface_lifecycle_status": "scaffolded",
                "pheromone_enforcement": "NOT_YET_WIRED",
                "promotion_status": "generated_not_promoted",
                "canonical_mutation_allowed": False,
            }
            SIDECAR_PATH.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
            generated = True

    report = {
        "schema": "lucidota.marrow_loop.surface_report.v0",
        "generated_at": utc_now(),
        "surface_generated": generated,
        "dry_run": bool(args.dry_run),
        "state_path": str(state_path.relative_to(ROOT)) if state_path.exists() else str(args.state),
        "surface_html": str(HTML_PATH.relative_to(ROOT)) if generated else None,
        "sidecar": str(SIDECAR_PATH.relative_to(ROOT)) if generated else None,
        "state_sha256": state_sha,
        "html_sha256": html_sha,
        "blockers": blockers,
    }
    report_path = OUT_DIR / f"marrow_loop_surface_report_{stamp()}.json"
    report["report_path"] = str(report_path.relative_to(ROOT))
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"SURFACE_REPORT_PATH={report_path.relative_to(ROOT)}")
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
