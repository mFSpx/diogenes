#!/usr/bin/env python3
"""LUCIDOTA security quarantine gate.

Scans configured local roots for secret-bearing file classes. Does not print secret
values, delete files, mutate DB, or call external services.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import fnmatch
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "05_OUTPUTS" / "security"

TOKEN_PATTERNS = [
    ("api_key_token_pattern", re.compile(r"(?i)(api[_-]?key|token|secret|bearer|authorization)\s*[:=]\s*['\"]?[^'\"\s]{12,}")),
    ("private_key_material", re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA |PGP )?PRIVATE KEY-----")),
    ("oauth_client_secret", re.compile(r"(?i)client_secret\s*['\"]?\s*[:=]\s*['\"][^'\"]+")),
]

PATH_RULES = [
    ("env_file", re.compile(r"(^|/|\\)\.env(?:\..*)?$", re.I), "quarantined"),
    ("oauth_credential_json", re.compile(r"(oauth|client_secret|credentials?).*\.json$", re.I), "quarantined"),
    ("ssh_private_key", re.compile(r"(^|/|\\)(id_rsa|id_ed25519|id_ecdsa|id_dsa)(?:\..*)?$|\.(pem|p12|pfx)$", re.I), "quarantined"),
    ("browser_cookies", re.compile(r"(cookies|cookie_store|login data|web data|places\.sqlite|key4\.db)$", re.I), "quarantined"),
    ("codex_logs_sessions_history", re.compile(r"(^|/|\\)\.codex(/|\\).*(sessions|history|logs?|credentials?|tokens?)", re.I), "quarantined"),
    ("sandbox_secret_directory", re.compile(r"(^|/|\\)(secrets?|secret_store|sandbox_secrets?)(/|\\)", re.I), "quarantined"),
    ("token_named_log", re.compile(r"(token|secret|credential|auth).*\.(log|txt|json|jsonl)$", re.I), "deferred"),
]

SKIP_DIRS = {"target", "node_modules", "__pycache__", ".git/objects", ".cache"}
MAX_CONTENT_SCAN_BYTES = 2_000_000


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def configured_roots(extra: list[str]) -> list[Path]:
    roots = [ROOT]
    codex = Path.home() / ".codex"
    if codex.exists():
        roots.append(codex)
    for item in extra:
        p = Path(item).expanduser().resolve()
        if p.exists():
            roots.append(p)
    unique: list[Path] = []
    seen = set()
    for r in roots:
        rr = r.resolve()
        if rr not in seen:
            unique.append(rr)
            seen.add(rr)
    return unique


def iter_files(roots: Iterable[Path], max_files: int) -> Iterable[Path]:
    count = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not any(f"/{skip}" in str(Path(dirpath) / d) for skip in SKIP_DIRS)]
            for name in filenames:
                count += 1
                if count > max_files:
                    return
                yield Path(dirpath) / name


def content_rules(path: Path) -> list[str]:
    try:
        if path.stat().st_size > MAX_CONTENT_SCAN_BYTES:
            return []
        data = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    hits = []
    for rule, pattern in TOKEN_PATTERNS:
        if pattern.search(data):
            hits.append(rule)
    return hits


def default_excludes(mode: str) -> list[str]:
    if mode != "brain_archaeology_ingest":
        return []
    return [
        ".git/*", "node_modules/*", "target/*", "__pycache__/*", ".venv/*", "venv/*",
        ".codex/*", "*.env", "*.env.*", "**/.env", "**/.env.*", "**/secrets/*", "**/secret_store/*",
        "**/runtime/secrets/*", "**/credentials*.json", "**/*credential*.json", "**/*client_secret*.json",
        "**/cookies*", "**/Login Data", "**/key4.db", "**/id_rsa", "**/id_ed25519", "**/*.pem",
    ]


def is_excluded(path: Path, excludes: list[str]) -> tuple[bool, str | None]:
    r = rel(path)
    for pattern in excludes:
        if fnmatch.fnmatch(r, pattern) or fnmatch.fnmatch(str(path), pattern):
            return True, pattern
    return False, None


def scan(
    roots: list[Path],
    max_files: int,
    mode: str = "full",
    excludes: list[str] | None = None,
    *,
    excluded_sample_limit: int = 200,
) -> dict:
    excludes = [*(default_excludes(mode)), *((excludes or []))]
    findings = []
    excluded_findings_sample = []
    scanned = 0
    included_files = 0
    excluded_files = 0
    clean_included_files = 0
    excluded_counts = {"quarantined": 0, "deferred": 0, "unknown": 0}
    for path in iter_files(roots, max_files):
        scanned += 1
        excluded, exclude_pattern = is_excluded(path, excludes)
        if excluded:
            excluded_files += 1
        else:
            included_files += 1
        path_text = str(path)
        path_rules = [(name, decision) for name, pattern, decision in PATH_RULES if pattern.search(path_text)]
        content_hits = []
        # Only content scan plausible text-ish files or path-rule files.
        if path.suffix.lower() in {".txt", ".log", ".json", ".jsonl", ".yaml", ".yml", ".env", ".md", ".toml", ".ini", ".conf"} or path_rules:
            content_hits = content_rules(path)
        if path_rules or content_hits:
            rules = [r[0] for r in path_rules] + content_hits
            decision = "quarantined" if any(d == "quarantined" for _, d in path_rules) or content_hits else "deferred"
            if mode == "brain_archaeology_ingest" and not excluded:
                excluded = True
                exclude_pattern = "auto_exclude_security_finding_for_brain_archaeology"
                excluded_files += 1
                included_files = max(0, included_files - 1)
            finding = {
                "path": rel(path),
                "rule_names": sorted(set(rules)),
                "decision": decision,
                "excluded_from_ingest": bool(excluded),
                "exclude_pattern": exclude_pattern,
                "secret_values_printed": False,
            }
            if excluded:
                excluded_counts[decision] = excluded_counts.get(decision, 0) + 1
                if excluded_sample_limit < 0 or len(excluded_findings_sample) < excluded_sample_limit:
                    excluded_findings_sample.append(finding)
            else:
                findings.append(finding)
        elif not excluded:
            clean_included_files += 1
    counts = {"clean": clean_included_files, "quarantined": 0, "deferred": 0, "unknown": 0}
    for f in findings:
        counts[f["decision"]] += 1
    clean_manifest = counts["quarantined"] == 0 and counts["unknown"] == 0
    brain_allowed = clean_manifest and mode == "brain_archaeology_ingest"
    return {
        "schema": "lucidota.security_quarantine_manifest.v2",
        "generated_at": now(),
        "mode": mode,
        "roots_scanned": [str(r) for r in roots],
        "files_scanned": scanned,
        "included_files_scanned": included_files,
        "excluded_files_scanned": excluded_files,
        "max_files": max_files,
        "findings_count": len(findings),
        "included_findings_count": len(findings),
        "excluded_findings_count": sum(excluded_counts.values()),
        "counts": counts,
        "excluded_counts": excluded_counts,
        "excluded_sample_limit": excluded_sample_limit,
        "clean_manifest": clean_manifest,
        "brain_archaeology_full_ingest_allowed": brain_allowed,
        "brain_archaeology_ingest_rule": "Only included clean files may be used; excluded/quarantined/deferred artifacts remain blocked from embedding, summarization, extraction, design_atom extraction, and graph promotion.",
        "exclusion_patterns": excludes,
        "quarantined_artifacts_may_be_used_for_brain_archaeology": False,
        "secret_values_printed": False,
        "findings": findings,
        "excluded_findings_sample": excluded_findings_sample,
        "blockers": [] if clean_manifest else ["security_quarantine_manifest_not_clean"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate LUCIDOTA security quarantine manifest")
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--max-files", type=int, default=200_000)
    parser.add_argument("--mode", choices=["full", "brain_archaeology_ingest"], default="full")
    parser.add_argument("--exclude", action="append", default=[], help="additional glob pattern excluded from ingest eligibility")
    parser.add_argument("--excluded-sample-limit", type=int, default=200, help="maximum excluded finding rows to retain in manifest; -1 keeps all")
    args = parser.parse_args()
    roots = configured_roots(args.root)
    manifest = scan(roots, args.max_files, mode=args.mode, excludes=args.exclude, excluded_sample_limit=args.excluded_sample_limit)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"security_quarantine_manifest_{stamp()}.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"SECURITY_MANIFEST={out.relative_to(ROOT)}")
    print(f"MODE={manifest['mode']}")
    print(f"CLEAN_MANIFEST={str(manifest['clean_manifest']).lower()}")
    print(f"BRAIN_ARCHAEOLOGY_FULL_INGEST_ALLOWED={str(manifest['brain_archaeology_full_ingest_allowed']).lower()}")
    print(f"INCLUDED_FINDINGS_COUNT={manifest['included_findings_count']}")
    print(f"EXCLUDED_FINDINGS_COUNT={manifest['excluded_findings_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
