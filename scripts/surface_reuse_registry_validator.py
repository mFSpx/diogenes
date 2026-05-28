#!/usr/bin/env python3
"""Validate generated surface sidecars are discoverable through TICKLETRUNK.

The registry is TICKLETRUNK. This checker keeps generated/reusable surfaces
findable without mutating canonical graph or surface state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "00_PROJECT_BRAIN" / "TICKLETRUNK.json"
SIDECAR_DIR = ROOT / "07_SURFACES" / "sidecars"
OUT = ROOT / "05_OUTPUTS" / "surfaces"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
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


def load_manifest() -> dict[str, Any]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"missing manifest {rel(MANIFEST)}")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("tickletrunk_manifest_must_be_object")
    return data


def surface_manifest_paths(manifest: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for entry in manifest.get("toolboxes", {}).get("SURFACES", []):
        for key in ("relative_path", "path", "realpath"):
            raw = entry.get(key)
            if not raw:
                continue
            p = Path(str(raw))
            if p.is_absolute():
                paths.add(rel(p))
            else:
                paths.add(str(p))
    return paths


def candidate_paths_from_sidecar(sidecar: Path, data: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [{"role": "sidecar", "path": rel(sidecar), "required": True}]
    for key in ("html_path", "rendered_artifact_path", "artifact_path"):
        raw = data.get(key)
        if raw:
            p = Path(str(raw))
            if not p.is_absolute():
                p = ROOT / p
            candidates.append({"role": key, "path": rel(p), "required": True})
    # common convention: sidecars/foo.json -> generated/foo.html
    conventional = ROOT / "07_SURFACES" / "generated" / f"{sidecar.stem}.html"
    if conventional.exists() and not any(c["path"] == rel(conventional) for c in candidates):
        candidates.append({"role": "conventional_html", "path": rel(conventional), "required": True})
    return candidates


def validate_sidecars(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest()
    manifest_paths = surface_manifest_paths(manifest)
    sidecars = [ROOT / s for s in args.sidecar] if args.sidecar else sorted(SIDECAR_DIR.glob("*.json"))
    results = []
    blockers: list[str] = []
    for sidecar in sidecars:
        if not sidecar.exists():
            results.append({"sidecar": rel(sidecar), "passed": False, "blockers": ["sidecar_missing"]})
            blockers.append(f"sidecar_missing:{rel(sidecar)}")
            continue
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("sidecar_must_be_object")
        except Exception as exc:
            results.append({"sidecar": rel(sidecar), "passed": False, "blockers": [f"sidecar_parse_error:{exc}"]})
            blockers.append(f"sidecar_parse_error:{rel(sidecar)}")
            continue
        checks = []
        sidecar_blockers: list[str] = []
        for cand in candidate_paths_from_sidecar(sidecar, data):
            present = cand["path"] in manifest_paths
            checks.append({**cand, "tickletrunk_registered": present})
            if cand.get("required") and not present:
                sidecar_blockers.append(f"not_registered:{cand['path']}")
        if data.get("canonical_mutation_allowed") is True:
            sidecar_blockers.append("canonical_mutation_allowed_true")
        interaction = str(data.get("interaction_model", ""))
        if "command envelope" not in interaction.lower() and "command-envelope" not in interaction.lower():
            sidecar_blockers.append("interaction_model_missing_command_envelope")
        blockers.extend(f"{rel(sidecar)}:{b}" for b in sidecar_blockers)
        results.append({
            "sidecar": rel(sidecar),
            "sidecar_sha256": sha_file(sidecar),
            "surface_id": data.get("surface_id") or data.get("view_key"),
            "checks": checks,
            "passed": not sidecar_blockers,
            "blockers": sidecar_blockers,
        })
    return {
        "action": "surface_reuse_registry_validate",
        "generated_at": now(),
        "manifest": rel(MANIFEST),
        "surface_manifest_entries": len(manifest_paths),
        "sidecars_checked": len(sidecars),
        "results": results,
        "blockers": blockers,
        "status": "PASS" if not blockers else "FAIL",
        "db_writes_performed": False,
        "graph_writes_performed": False,
    }


def write_report(payload: dict[str, Any]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"surface_reuse_registry_validator_{payload['status'].lower()}_{stamp()}.json"
    payload["report_path"] = rel(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"REPORT_PATH={rel(path)}")
    print(f"SURFACE_REUSE_REGISTRY={payload['status']}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate generated surface sidecars are indexed in TICKLETRUNK")
    ap.add_argument("--sidecar", action="append", help="Specific sidecar path; default: all sidecars")
    args = ap.parse_args()
    payload = validate_sidecars(args)
    write_report(payload)
    return 0 if payload["status"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
