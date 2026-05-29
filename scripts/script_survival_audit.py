#!/usr/bin/env python3
"""Append-only script survival audit manifest writer.

This tool records ACTIVE_KEEP, ACTIVE_REPAIR, LEGACY_CORPSE, and UNKNOWN_HOLD
script audit verdicts without deleting or moving scripts. Corpse verdicts also
feed the Krampuschewing script-corpse ingest queue.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_MANIFEST = ROOT / "scripts" / "SCRIPT_AUDIT_MANIFEST.jsonl"
DEFAULT_CORPSE_MANIFEST = ROOT / "scripts" / "CORPSE_MANIFEST.jsonl"
DEFAULT_KRAMPUS_QUEUE = ROOT / "scripts" / "KRAMPUSCHEWING_SCRIPT_CORPSES.jsonl"
DEFAULT_KRAMPUS_CODE_ARCHIVE = ROOT / "KRAMPUSCHEWING" / "Script_Corpses"
DEFAULT_UNKNOWN_HOLD = ROOT / "scripts" / "SCRIPT_AUDIT_UNKNOWN_HOLD.jsonl"
VERDICTS = {"ACTIVE_KEEP", "ACTIVE_REPAIR", "LEGACY_CORPSE", "UNKNOWN_HOLD"}
KRAMPUS_ACTIONS = [
    "index",
    "classify",
    "extract_imports",
    "extract_entrypoints",
    "extract_dead_patterns",
    "link_to_replacements",
    "graph_historical_role",
    "preserve_as_evidence",
]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def archive_to_krampus(path: Path, archive_root: Path) -> str | None:
    """Copy a corpse script into KRAMPUSCHEWING without deleting the source."""
    if not path.exists() or not path.is_file():
        return None
    digest = sha256_file(path)
    safe = rel(path).replace("/", "__")
    target = archive_root / f"{safe}.{digest}.corpse"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(path, target)
    return rel(target)


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path, max_bytes: int = 512 * 1024) -> str:
    if path.stat().st_size > max_bytes:
        return path.read_bytes()[:max_bytes].decode("utf-8", errors="replace")
    return path.read_text(encoding="utf-8", errors="replace")


def python_facts(path: Path) -> dict[str, Any]:
    facts: dict[str, Any] = {"imports": [], "functions": [], "argparse_flags": []}
    if path.suffix != ".py" or not path.exists():
        return facts
    try:
        tree = ast.parse(read_text(path), filename=rel(path))
    except SyntaxError as exc:
        facts["syntax_error"] = str(exc)
        return facts
    imports: set[str] = set()
    functions: list[str] = []
    flags: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".", 1)[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("-"):
                    flags.add(arg.value)
    facts["imports"] = sorted(imports)
    facts["functions"] = sorted(functions)
    facts["argparse_flags"] = sorted(flags)
    return facts


def script_facts(path: Path) -> dict[str, Any]:
    exists = path.exists()
    mode = path.stat().st_mode if exists else 0
    head = read_text(path, 4096) if exists and path.is_file() else ""
    facts = {
        "path": rel(path),
        "exists": exists,
        "language": path.suffix.lstrip(".") or "unknown",
        "executable": bool(mode & stat.S_IXUSR),
        "shebang": head.splitlines()[0] if head.startswith("#!") else None,
        "size_bytes": path.stat().st_size if exists else None,
        "hash_blake3_or_sha256": sha256_file(path),
        "has_module_docstring": path.suffix == ".py" and ('"""' in head[:160] or "'''" in head[:160]),
        "has_strict_shell": path.suffix == ".sh" and "set -euo pipefail" in head[:300],
    }
    facts.update(python_facts(path))
    return facts


def split_items(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        out.extend(item.strip() for item in value.split(";") if item.strip())
    return out


def manifest_row(args: argparse.Namespace, facts: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp_utc": now(),
        "path": facts["path"],
        "verdict": args.verdict,
        "role": args.role,
        "callers": split_items(args.caller),
        "inputs": split_items(args.input),
        "outputs": split_items(args.output),
        "side_effects": split_items(args.side_effect),
        "slop_score": args.slop_score,
        "survival_score": args.survival_score,
        "golden_comparison": {
            "ncnn_alignment": args.ncnn_alignment,
            "flow_alignment": args.flow_alignment,
        },
        "decision_reason": args.decision_reason,
        "changes_made": split_items(args.change),
        "validation": split_items(args.validation),
        "next_action": args.next_action,
        "facts": facts,
    }


def corpse_row(args: argparse.Namespace, facts: dict[str, Any], archived_copy: str | None = None) -> dict[str, Any]:
    return {
        "timestamp_utc": now(),
        "path": facts["path"],
        "verdict": "LEGACY_CORPSE",
        "death_reason": args.death_reason or args.decision_reason,
        "evidence_read": split_items(args.evidence_read),
        "superseded_by": args.superseded_by,
        "risk_if_kept_active": args.risk_if_kept_active or "active trust would be misleading",
        "historical_value": args.historical_value or "preserve script as implementation history and pattern evidence",
        "krampuschewing_ingest": True,
        "krampuschewing_archived_copy": archived_copy,
        "hash_blake3_or_sha256": facts.get("hash_blake3_or_sha256"),
        "auditor": "ouroboros_script_audit",
        "notes": args.notes or "retired without deletion",
    }


def krampus_row(args: argparse.Namespace, facts: dict[str, Any], corpse_manifest: Path, archived_copy: str | None = None) -> dict[str, Any]:
    return {
        "timestamp_utc": now(),
        "source_path": facts["path"],
        "source_manifest": "/" + rel(corpse_manifest),
        "archived_copy": archived_copy,
        "ingest_class": "SCRIPT_CORPSE",
        "desired_actions": KRAMPUS_ACTIONS,
        "never_delete": True,
    }


def unknown_row(args: argparse.Namespace, facts: dict[str, Any], audit_manifest: Path) -> dict[str, Any]:
    return {
        "timestamp_utc": now(),
        "path": facts["path"],
        "source_manifest": "/" + rel(audit_manifest),
        "missing_evidence": split_items(args.missing_evidence) or ["callers_or_runtime_role_unproven"],
        "next_inspection_action": args.next_action or "trace callers and runtime receipts",
        "auditor": "ouroboros_script_audit",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a script survival audit verdict without deleting scripts.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--verdict", required=True, choices=sorted(VERDICTS))
    parser.add_argument("--role", required=True)
    parser.add_argument("--caller", action="append")
    parser.add_argument("--input", action="append")
    parser.add_argument("--output", action="append")
    parser.add_argument("--side-effect", action="append")
    parser.add_argument("--slop-score", type=int, required=True)
    parser.add_argument("--survival-score", type=int, required=True)
    parser.add_argument("--ncnn-alignment", required=True)
    parser.add_argument("--flow-alignment", required=True)
    parser.add_argument("--decision-reason", required=True)
    parser.add_argument("--change", action="append")
    parser.add_argument("--validation", action="append")
    parser.add_argument("--next-action")
    parser.add_argument("--evidence-read", action="append")
    parser.add_argument("--death-reason")
    parser.add_argument("--superseded-by")
    parser.add_argument("--risk-if-kept-active")
    parser.add_argument("--historical-value")
    parser.add_argument("--missing-evidence", action="append")
    parser.add_argument("--notes")
    parser.add_argument("--audit-manifest", default=str(DEFAULT_AUDIT_MANIFEST))
    parser.add_argument("--corpse-manifest", default=str(DEFAULT_CORPSE_MANIFEST))
    parser.add_argument("--krampus-queue", default=str(DEFAULT_KRAMPUS_QUEUE))
    parser.add_argument("--krampus-code-archive", default=str(DEFAULT_KRAMPUS_CODE_ARCHIVE))
    parser.add_argument("--unknown-hold", default=str(DEFAULT_UNKNOWN_HOLD))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.slop_score <= 10 or not 0 <= args.survival_score <= 10:
        raise SystemExit("slop_score_and_survival_score_must_be_0_to_10")

    target = resolve(args.path)
    facts = script_facts(target)
    audit_manifest = resolve(args.audit_manifest)
    corpse_manifest = resolve(args.corpse_manifest)
    krampus_queue = resolve(args.krampus_queue)
    krampus_code_archive = resolve(args.krampus_code_archive)
    unknown_hold = resolve(args.unknown_hold)

    row = manifest_row(args, facts)
    append_jsonl(audit_manifest, row)
    written = [rel(audit_manifest)]
    if args.verdict == "LEGACY_CORPSE":
        archived_copy = archive_to_krampus(target, krampus_code_archive)
        append_jsonl(corpse_manifest, corpse_row(args, facts, archived_copy))
        append_jsonl(krampus_queue, krampus_row(args, facts, corpse_manifest, archived_copy))
        written.extend([rel(corpse_manifest), rel(krampus_queue)])
        if archived_copy:
            written.append(archived_copy)
    elif args.verdict == "UNKNOWN_HOLD":
        append_jsonl(unknown_hold, unknown_row(args, facts, audit_manifest))
        written.append(rel(unknown_hold))

    result = {"schema": "lucidota.script_survival_audit.result.v1", "path": facts["path"], "verdict": args.verdict, "written": written, "facts": facts, "status": "PASS"}
    if args.json:
        print(json.dumps(result, sort_keys=True))
    print("SCRIPT_AUDIT_MANIFEST=" + rel(audit_manifest))
    for path in written[1:]:
        print("SCRIPT_AUDIT_SIDE_EFFECT=" + path)
    print("SCRIPT_SURVIVAL_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
