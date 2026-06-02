#!/usr/bin/env python3
"""Safe active-source audit dump for Operation Root-Rotor.

The dump is append-only evidence for canon synthesis. It excludes binaries,
caches, bulk bodies, and proof-hoard byte lanes while preserving per-file hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCLUDED_DIRS = [".git", ".venv", "__pycache__", "KRAMPUSCHEWING", "05_OUTPUTS", "03_VAULT"]
DEFAULT_EXTENSIONS = {".py", ".rs", ".sql", ".md", ".json", ".sh", ".toml", ".yaml", ".yml", ".ini", ".js", ".ts", ".tsx", ".txt"}
# Other checked-out/vendor repositories are not canon input. CLAW/LUCI is the active exception.
DEFAULT_EXCLUDED_PREFIXES = [
    "01_REPOS/",
    "04_RUNTIME/",
    "ALGOS/evolved/",
    ".claw/sessions/",
    ".pytest_cache/",
    "09_STORAGE/",
    "07_SURFACES/",
    "coding-agent/",
    "the_other_asshole/",
]
DEFAULT_INCLUDED_PREFIXES = ["01_REPOS/claudecode/"]
DEFAULT_OUTPUT = ROOT / "GOALS" / "ROOT_ROTOR_ACTIVE_SOFTWARE_AUDIT_DUMP.txt"
DEFAULT_MANIFEST = ROOT / "GOALS" / "ROOT_ROTOR_ACTIVE_SOFTWARE_AUDIT_MANIFEST.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def resolves_inside_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def discover_nested_repo_prefixes(root: Path) -> list[str]:
    prefixes: list[str] = []
    for git_marker in root.rglob(".git"):
        repo_root = git_marker.parent
        if repo_root == root:
            continue
        try:
            prefixes.append(rel(repo_root, root).rstrip("/") + "/")
        except ValueError:
            continue
    return sorted(set(prefixes))


def _to_prefix(prefix: str) -> str:
    return prefix.rstrip("/") + "/"


def _has_prefix(path: str, prefix: str) -> bool:
    normalized = _to_prefix(prefix)
    return path == normalized[:-1] or path.startswith(normalized)


def dirty_nested_repo_prefixes(
    root: Path,
    prefixes: Iterable[str],
    excluded_dirs: Iterable[str] | None = None,
    excluded_prefixes: Iterable[str] | None = None,
    allowed_prefixes: Iterable[str] | None = None,
) -> list[str]:
    excluded_dirs = set(excluded_dirs or [])
    excluded_prefixes = set(excluded_prefixes or [])
    allowed_prefixes = set(allowed_prefixes or [])

    dirty: list[str] = []
    for prefix in prefixes:
        repo_root = root / prefix
        try:
            proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                text=True,
                capture_output=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            if any(_has_prefix(prefix, dir_name) for dir_name in excluded_dirs):
                if not any(_has_prefix(prefix, allowed) for allowed in allowed_prefixes):
                    continue
            if any(_has_prefix(prefix, excluded_prefix) for excluded_prefix in excluded_prefixes):
                if not any(_has_prefix(prefix, allowed) for allowed in allowed_prefixes):
                    continue
            dirty.append(prefix)
    return dirty


def should_skip(
    path: Path,
    root: Path,
    excluded_dirs: Iterable[str],
    excluded_prefixes: Iterable[str] | None = None,
    included_prefixes: Iterable[str] | None = None,
    nested_repo_prefixes: Iterable[str] | None = None,
) -> bool:
    if path.is_symlink() and not resolves_inside_root(path, root):
        return True
    rel_path = rel(path, root)
    names = set(Path(rel_path).parts)
    if names.intersection(set(excluded_dirs)):
        return True
    for prefix in included_prefixes or []:
        if rel_path.startswith(prefix):
            return False
    for prefix in nested_repo_prefixes or []:
        if rel_path.startswith(prefix):
            return True
    for prefix in excluded_prefixes or []:
        if rel_path.startswith(prefix):
            return True
    return False


def iter_active_sources(
    root: Path,
    excluded_dirs: list[str],
    extensions: set[str],
    excluded_prefixes: list[str] | None = None,
    included_prefixes: list[str] | None = None,
    nested_repo_prefixes: list[str] | None = None,
) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if should_skip(path, root, excluded_dirs, excluded_prefixes, included_prefixes, nested_repo_prefixes):
            continue
        if path.suffix.lower() not in extensions:
            continue
        files.append(path)
    return sorted(files, key=lambda p: rel(p, root))


def write_audit_dump(
    root: Path = ROOT,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    manifest_path: Path | None = None,
    max_bytes: int = 100_000,
    excluded_dirs: list[str] | None = None,
    extensions: set[str] | None = None,
    excluded_prefixes: list[str] | None = None,
    included_prefixes: list[str] | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    excluded = list(excluded_dirs or DEFAULT_EXCLUDED_DIRS)
    exts = set(extensions or DEFAULT_EXTENSIONS)
    excluded_pfx = list(excluded_prefixes or DEFAULT_EXCLUDED_PREFIXES)
    base_included = list(included_prefixes or DEFAULT_INCLUDED_PREFIXES)
    nested_pfx = discover_nested_repo_prefixes(root)
    dirty_pfx = dirty_nested_repo_prefixes(
        root,
        nested_pfx,
        excluded_dirs=excluded,
        excluded_prefixes=excluded_pfx,
        allowed_prefixes=base_included,
    )
    included_pfx = sorted(set(base_included + dirty_pfx))
    output_path = output_path.resolve()
    manifest_path = (manifest_path or output_path.with_suffix(".manifest.json")).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    generated_at = now()
    file_entries: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as out:
        out.write("ROOT_ROTOR_ACTIVE_SOFTWARE_AUDIT_DUMP\n")
        out.write(f"generated_at={generated_at}\n")
        out.write(f"root={root}\n")
        out.write(f"max_bytes_per_file={max_bytes}\n")
        for path in iter_active_sources(root, excluded, exts, excluded_pfx, included_pfx, nested_pfx):
            if path.resolve() in {output_path, manifest_path}:
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            chunk = data[:max_bytes]
            path_rel = rel(path, root)
            entry = {
                "path": path_rel,
                "size_bytes": len(data),
                "bytes_read": len(chunk),
                "truncated": len(data) > max_bytes,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            file_entries.append(entry)
            out.write(f"\n\n=== FILE: {path_rel} ===\n")
            out.write(f"size_bytes={entry['size_bytes']} bytes_read={entry['bytes_read']} truncated={str(entry['truncated']).lower()} sha256={entry['sha256']}\n")
            out.write(chunk.decode("utf-8", errors="replace"))

    result: dict[str, Any] = {
        "schema": "lucidota.root_rotor.audit_dump.v1",
        "generated_at": generated_at,
        "root": str(root),
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "max_bytes_per_file": max_bytes,
        "excluded_dirs": excluded,
        "extensions": sorted(exts),
        "excluded_prefixes": excluded_pfx,
        "included_prefixes": included_pfx,
        "nested_repo_prefixes": nested_pfx,
        "dirty_nested_repo_prefixes": dirty_pfx,
        "files_written": len(file_entries),
        "files": file_entries,
    }
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a safe Root-Rotor active-source audit dump.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--max-bytes", type=int, default=100_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = write_audit_dump(
        Path(args.root),
        Path(args.output),
        manifest_path=Path(args.manifest) if args.manifest else None,
        max_bytes=args.max_bytes,
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=False))
    else:
        print(f"AUDIT_DUMP={result['output_path']}")
        print(f"MANIFEST={result['manifest_path']}")
        print(f"FILES_WRITTEN={result['files_written']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
