#!/usr/bin/env python3
"""Build a local Indy_Reads persona corpus/distillation from project-brain docs.

No Drive, Gmail, Calendar, network, or ambient filesystem search: this reads only the
small allow-listed project-brain files that define current Indy_Reads behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
PROJECT_BRAIN = ROOT / "00_PROJECT_BRAIN"
SOURCE_FILES = [
    PROJECT_BRAIN / "INDY_SOUL.md",
    PROJECT_BRAIN / "INDY_MODEL_STACK.md",
    PROJECT_BRAIN / "DIOGENES_OPERATIONAL_SPEC.md",
    PROJECT_BRAIN / "ACTIVE_SPEC" / "05_COMPONENT_AUTHORITY_MAP.md",
    PROJECT_BRAIN / "RFCS" / "RFC-130-INDY-READS.md",
    PROJECT_BRAIN / "INDY_READS_POLYCAREER_WORKFLOW_WIZARD" / "WORKFLOW_CONTRACT.md",
    PROJECT_BRAIN / "INDY_READS_POLYCAREER_WORKFLOW_WIZARD" / "ARCHITECTURE.md",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "section"


def heading_section(path: Path, heading: str) -> str:
    text = read(path)
    lines = text.splitlines()
    out: list[str] = []
    in_section = False
    target = heading.lower().strip()
    target_level = 0
    for line in lines:
        m = re.match(r"^(#+)\s+(.+?)\s*$", line)
        if m:
            level = len(m.group(1))
            name = m.group(2).strip().lower()
            if in_section and level <= target_level:
                break
            if name == target:
                in_section = True
                target_level = level
                continue
        if in_section:
            out.append(line)
    return "\n".join(out).strip()


def matching_block(path: Path, pattern: str, lookahead: str = r"\n### |\Z") -> str:
    text = read(path)
    m = re.search(pattern + r"(?P<body>.*?)" + lookahead, text, re.S | re.I | re.M)
    return (m.group("body") or "").strip() if m else ""


def bullets(text: str) -> list[str]:
    found: list[str] = []
    for line in text.splitlines():
        x = line.strip()
        if x.startswith("- "):
            item = re.sub(r"^- \[[ xX]\]\s*", "", x)
            item = item[2:] if item.startswith("- ") else item
            found.append(re.sub(r"\s+", " ", item).strip())
    return found


def quote_units(path: Path, section_name: str, label: str, limit: int = 10) -> list[dict]:
    body = heading_section(path, section_name)
    units = bullets(body)[:limit]
    if not units and body:
        units = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.startswith("```")][:limit]
    return [
        {
            "label": label,
            "text": unit,
            "source": f"{path.relative_to(ROOT)}#{slug(section_name)}",
        }
        for unit in units
    ]


def build_corpus() -> dict:
    soul = PROJECT_BRAIN / "INDY_SOUL.md"
    stack = PROJECT_BRAIN / "INDY_MODEL_STACK.md"
    ops = PROJECT_BRAIN / "DIOGENES_OPERATIONAL_SPEC.md"
    authority = PROJECT_BRAIN / "ACTIVE_SPEC" / "05_COMPONENT_AUTHORITY_MAP.md"
    rfc = PROJECT_BRAIN / "RFCS" / "RFC-130-INDY-READS.md"
    contract = PROJECT_BRAIN / "INDY_READS_POLYCAREER_WORKFLOW_WIZARD" / "WORKFLOW_CONTRACT.md"
    architecture = PROJECT_BRAIN / "INDY_READS_POLYCAREER_WORKFLOW_WIZARD" / "ARCHITECTURE.md"

    units: list[dict] = []
    units += quote_units(soul, "WHO SHE IS", "identity", 12)
    units += quote_units(soul, "THE MANDATE", "mandate", 12)
    units += quote_units(soul, "OPERATING DOCTRINE (non-negotiable)", "doctrine", 16)
    units += quote_units(soul, "WRITE PATHS (hard gates)", "write_gate", 10)
    units += quote_units(soul, "THE 7 RESOLUTIONS (locked 2026-05-28)", "resolution", 12)
    units += quote_units(soul, "15 POLYCAREER ROLE MODES", "role_mode", 20)
    units += quote_units(stack, "THE POINT", "model_stack", 10)
    units += quote_units(stack, "VIBES / CODESTRAL (external code-work lane)", "external_lane", 10)
    units += quote_units(ops, "0. Roles & operating law", "ops_role", 8)
    units += quote_units(ops, "3. UX — conversational, deterministic routing (no mandatory slash commands)", "ux_route", 10)
    units += quote_units(ops, "4. OUTPUT multiplexer — better than any AI chat", "output_hyperplex", 8)
    units += quote_units(authority, "6. Indy_READs", "authority", 8)
    units += quote_units(authority, "8. Language Membrane / Multiplexing / Hyperplexing", "language_membrane", 8)
    units += quote_units(rfc, "3. Indy Contract", "rfc_contract", 12)
    units += quote_units(rfc, "5. Whole-System Interaction", "rfc_interaction", 10)
    units += quote_units(rfc, "8. Falsifiers", "rfc_falsifier", 10)
    units += quote_units(contract, "Non-negotiables", "workflow_contract", 12)
    units += quote_units(contract, "Default product format", "product_format", 12)
    units += quote_units(contract, "Glow Watch hook contract", "glow_watch", 8)
    units += quote_units(architecture, "Existing Indy_READs footprint found in this repo", "polycareer_foundation", 12)
    units += quote_units(architecture, "Transferable moves for Indy", "transferable_move", 12)
    units += quote_units(architecture, "Should Indy learn this?", "learning_rule", 8)

    # Keep it small and deterministic: de-duplicate exact text while preserving order.
    deduped: list[dict] = []
    seen: set[str] = set()
    for unit in units:
        key = unit["text"]
        if key and key not in seen:
            seen.add(key)
            deduped.append(unit)

    source_manifest = []
    for path in SOURCE_FILES:
        data = read(path).encode("utf-8")
        source_manifest.append(
            {
                "path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "sha256": hashlib.sha256(data).hexdigest() if path.exists() else None,
            }
        )

    distilled = {
        "voice": "concise, evidence-first, build-focused, calm under ambiguity",
        "boundaries": [
            "local project brain and Postgres only unless the operator explicitly authorizes another source",
            "no fake green; progress claims need a checklist, test, schema, row, hash, or artifact",
            "external writes remain draft/preview unless explicitly confirmed",
            "quiet side work must not interrupt the operator's active build flow",
        ],
        "runtime_jobs": [
            "brief current verified state and smallest next move",
            "remember operator corrections and decisions",
            "queue reminders, calendar intent, wiki notes, auth inventory, and reviews",
            "surface auth gaps without storing raw secrets",
            "cite source paths for claims that matter",
        ],
    }

    artifact = {
        "ok": True,
        "artifact": "indy_reads_local_persona_corpus_v1",
        "source_scope": "allow-listed local project-brain markdown only; no Drive access",
        "unit_count": len(deduped),
        "labels": sorted({u["label"] for u in deduped}),
        "distilled": distilled,
        "units": deduped,
        "sources": source_manifest,
    }
    payload = json.dumps({k: v for k, v in artifact.items() if k != "sources"}, sort_keys=True).encode("utf-8")
    artifact["artifact_sha256"] = hashlib.sha256(payload).hexdigest()
    return artifact


def render(artifact: dict) -> str:
    lines = [
        "INDY_READS LOCAL PERSONA CORPUS",
        "=================================",
        f"Artifact: {artifact['artifact']}",
        f"Scope: {artifact['source_scope']}",
        f"Units: {artifact['unit_count']}",
        f"SHA256: {artifact['artifact_sha256']}",
        "",
        "Distillation:",
        f"  - voice: {artifact['distilled']['voice']}",
    ]
    lines += [f"  - boundary: {x}" for x in artifact["distilled"]["boundaries"]]
    lines += [f"  - job: {x}" for x in artifact["distilled"]["runtime_jobs"]]
    lines += ["", "Source units:"]
    for unit in artifact["units"][:20]:
        lines.append(f"  - [{unit['label']}] {unit['text']} ({unit['source']})")
    if artifact["unit_count"] > 20:
        lines.append(f"  - ... {artifact['unit_count'] - 20} more")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-indy-corpus")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", type=Path, help="optional output path for the generated JSON artifact")
    args = ap.parse_args()
    artifact = build_corpus()
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, sort_keys=True) if args.json else render(artifact))
    return 0 if artifact.get("ok") and artifact.get("unit_count", 0) >= 12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
