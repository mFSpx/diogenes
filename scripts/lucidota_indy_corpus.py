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
    PROJECT_BRAIN / "INDY_READS_RUNTIME_CONTRACT.md",
    PROJECT_BRAIN / "THEPLAN.md",
    PROJECT_BRAIN / "BUILD_PLAN_AUDIT.md",
    PROJECT_BRAIN / "STATUS.md",
    PROJECT_BRAIN / "DECISIONS.md",
    PROJECT_BRAIN / "GLOSSARY.md",
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
    return m.group("body").strip() if m else ""


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
    contract = PROJECT_BRAIN / "INDY_READS_RUNTIME_CONTRACT.md"
    plan = PROJECT_BRAIN / "THEPLAN.md"
    audit = PROJECT_BRAIN / "BUILD_PLAN_AUDIT.md"
    status = PROJECT_BRAIN / "STATUS.md"
    decisions = PROJECT_BRAIN / "DECISIONS.md"
    glossary = PROJECT_BRAIN / "GLOSSARY.md"

    units: list[dict] = []
    units += quote_units(contract, "Identity", "identity", 8)
    units += quote_units(contract, "Prime Directives", "directive", 12)
    units += quote_units(contract, "Answer Contract", "answer_contract", 10)
    units += quote_units(contract, "Memory Routines", "routine", 12)
    units += quote_units(contract, "Failure Mode", "failure_mode", 4)
    units += quote_units(plan, "Phase 006: Indy_Reads Assistant Layer", "phase_006", 12)
    units += quote_units(plan, "Research / Iteration Method", "method", 12)
    units += quote_units(status, "Current State", "status_current", 12)
    units += quote_units(status, "Next Verification", "status_next", 12)
    units += quote_units(decisions, "Current Decisions", "decision", 12)
    units += quote_units(glossary, "Indy_Reads", "glossary", 8)

    audit_body = matching_block(
        audit,
        r"^###\s+011\s+Indy_Reads\s*/\s*Persona\s*/\s*Assistant Layer[^\n]*\n",
    )
    for line in audit_body.splitlines():
        m = re.match(r"^- \[(?P<mark>[ xX])\]\s*(?P<item>.+)$", line.strip())
        if not m:
            continue
        status_label = "done" if m.group("mark").lower() == "x" else "open"
        units.append(
            {
                "label": f"audit_{status_label}",
                "text": re.sub(r"\s+", " ", m.group("item")).strip(),
                "source": "00_PROJECT_BRAIN/BUILD_PLAN_AUDIT.md#011-indy-reads-persona-assistant-layer",
            }
        )
        if len([u for u in units if u["label"].startswith("audit_")]) >= 24:
            break

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
