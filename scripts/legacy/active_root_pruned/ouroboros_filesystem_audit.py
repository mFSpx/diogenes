#!/usr/bin/env python3
"""Generate whole-filesystem audit maps for the Ouroboros script survival loop."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import stat
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "ouroboros_audit"
EXCLUDE_DIRS = {
    ".git", ".venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    "01_REPOS", "02_RECORDS_OFFICE", "03_VAULT", "04_RUNTIME", "05_OUTPUTS",
    "07_SURFACES/generated", "09_STORAGE", "BOOKS", "KRAMPUSCHEWING",
    "TODO6969", ".lucidota_agents",
}
CALLGRAPH_ROOTS = {"scripts", "tests", "06_SCHEMA", "00_PROJECT_BRAIN", ".github", "services", "src", "ALGOS", "ahoy_sim"}
SCRIPT_SUFFIXES = {".py", ".sh", ".bash", ".zsh", ".ps1", ".cmd", ".bat", ".rs", ".js", ".ts", ".sql"}
TEXT_SUFFIXES = SCRIPT_SUFFIXES | {".md", ".json", ".jsonl", ".toml", ".yaml", ".yml", ".service", ".timer", ".env", ""}
PUBLIC_PATTERNS = [
    {
        "problem_class": "CLI argument parsers",
        "public_patterns_checked": ["https://docs.python.org/3/library/argparse.html"],
        "useful_pattern": "Use one explicit ArgumentParser, clear subcommands, and deterministic exit codes; avoid wrapper dependencies for small CLIs.",
        "rejected_pattern": "Do not add higher-level argparse wrapper libraries just for aesthetics.",
        "dependency_warning": "stdlib argparse is enough.",
    },
    {
        "problem_class": "process runners",
        "public_patterns_checked": ["https://docs.python.org/3/library/subprocess.html"],
        "useful_pattern": "Use subprocess.run with args lists, capture_output/text when receipts need tails, and explicit timeout for bounded execution.",
        "rejected_pattern": "Avoid shell=True and unbounded child processes in audit/worker scripts.",
        "dependency_warning": "stdlib subprocess is enough.",
    },
    {
        "problem_class": "JSONL processors and manifests",
        "public_patterns_checked": ["https://jsonlines.org/", "https://jsonlines.readthedocs.io/en/latest/"],
        "useful_pattern": "Append one UTF-8 JSON value per line so large manifests stream and partial rows remain inspectable.",
        "rejected_pattern": "Avoid giant mutable JSON arrays for append-only audit logs.",
        "dependency_warning": "Use json.dumps per line; avoid jsonlines dependency unless it removes substantial code.",
    },
]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def excluded(path: Path) -> bool:
    r = rel(path)
    parts = set(path.relative_to(ROOT).parts) if path.is_absolute() and ROOT in path.parents else set(path.parts)
    if any(part in EXCLUDE_DIRS for part in parts):
        return True
    return any(r == ex or r.startswith(ex + "/") for ex in EXCLUDE_DIRS)


def iter_files() -> list[Path]:
    """Return stable repo files while pruning excluded subtrees before descent."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        base = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not excluded(base / d))
        for filename in sorted(filenames):
            path = base / filename
            if not excluded(path) and path.is_file():
                files.append(path)
    return sorted(files, key=lambda p: rel(p))


def is_script(path: Path) -> bool:
    if path.name in {"Makefile", "makefile"}:
        return True
    if path.suffix in SCRIPT_SUFFIXES:
        return True
    try:
        mode = path.stat().st_mode
        return bool(mode & stat.S_IXUSR) and path.suffix in TEXT_SUFFIXES
    except Exception:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path, max_bytes: int = 512 * 1024) -> str:
    data = path.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="replace")


def py_metrics(path: Path, text: str) -> dict[str, Any]:
    if path.suffix != ".py":
        return {"function_count": 0, "branch_count": 0, "import_count": 0, "cli_arg_count": 0}
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {"function_count": 0, "branch_count": 0, "import_count": 0, "cli_arg_count": 0, "syntax_error": True}
    funcs = branches = imports = cli_args = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs += 1
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp, ast.Match)):
            branches += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            imports += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            cli_args += 1
    return {"function_count": funcs, "branch_count": branches, "import_count": imports, "cli_arg_count": cli_args}


def script_inventory(files: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in files:
        if not is_script(path):
            continue
        text = read_text(path)
        rows.append({
            "path": rel(path),
            "suffix": path.suffix or path.name,
            "executable": bool(path.stat().st_mode & stat.S_IXUSR),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "shebang": text.splitlines()[0] if text.startswith("#!") else None,
            **py_metrics(path, text),
        })
    return rows


def linecount_complexity(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in inventory:
        path = ROOT / row["path"]
        text = read_text(path)
        lines = text.splitlines()
        nonblank = [line for line in lines if line.strip()]
        actual_logic = sum(1 for line in nonblank if not line.strip().startswith(("#", "\"\"\"", "'''")))
        boilerplate = max(0, len(nonblank) - actual_logic)
        unbounded_read = bool(re.search(r"read_text\(|read_bytes\(|\.read\(\)", text)) and "max" not in text[:2000].lower()
        slop = min(10, (1 if unbounded_read else 0) + (2 if len(nonblank) > 300 else 0) + (2 if row.get("branch_count", 0) > 40 else 0) + (1 if row.get("cli_arg_count", 0) > 20 else 0))
        rows.append({
            "path": row["path"],
            "loc": len(lines),
            "nonblank_loc": len(nonblank),
            "actual_logic_loc_estimate": actual_logic,
            "boilerplate_loc_estimate": boilerplate,
            "function_count": row.get("function_count", 0),
            "branch_count": row.get("branch_count", 0),
            "import_count": row.get("import_count", 0),
            "cli_arg_count": row.get("cli_arg_count", 0),
            "unbounded_read_risk": unbounded_read,
            "slop_score_estimate": slop,
            "estimated_minimum_sane_loc": max(20, int(len(nonblank) * (0.65 if len(nonblank) > 150 else 0.85))),
        })
    return rows


def callgraph(inventory: list[dict[str, Any]], files: list[Path]) -> list[dict[str, Any]]:
    text_files = [
        p for p in files
        if p.suffix in TEXT_SUFFIXES
        and p.stat().st_size <= 256 * 1024
        and (p.relative_to(ROOT).parts[0] if p.relative_to(ROOT).parts else "") in CALLGRAPH_ROOTS
    ]
    haystacks: list[tuple[str, str]] = []
    for path in text_files:
        try:
            haystacks.append((rel(path), read_text(path)))
        except Exception:
            pass
    rows = []
    for script in inventory:
        spath = script["path"]
        base = Path(spath).name
        callers = []
        for fpath, text in haystacks:
            if fpath == spath:
                continue
            if spath in text or base in text:
                callers.append(fpath)
        rows.append({"path": spath, "callers": sorted(set(callers))[:200], "caller_count": len(set(callers))})
    return rows


def dataflow(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    patterns = {
        "outputs": [r"05_OUTPUTS/[A-Za-z0-9_./*-]+", r"REPORT_PATH", r"RECEIPT_PATH", r"\.jsonl"],
        "db_tables": [r"lucidota_[a-z_]+\.[a-zA-Z0-9_]+"],
        "env_vars": [r"os\.environ\.get\(['\"]([A-Z0-9_]+)['\"]"],
        "subprocesses": [r"subprocess\.run", r"Popen"],
    }
    for script in inventory:
        text = read_text(ROOT / script["path"])
        row = {"path": script["path"], "outputs": [], "db_tables": [], "env_vars": [], "subprocesses": []}
        for key, regs in patterns.items():
            found = []
            for regex in regs:
                for match in re.finditer(regex, text):
                    found.append(match.group(1) if match.groups() else match.group(0))
            row[key] = sorted(set(found))[:100]
        rows.append(row)
    return rows


def duplication(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    by_normalized: dict[str, list[str]] = defaultdict(list)
    for row in inventory:
        by_hash[row["sha256"]].append(row["path"])
        stem = Path(row["path"]).stem.replace("dbos", "QUEUELEGACY").replace("absurd", "QUEUELEGACY")
        by_normalized[stem].append(row["path"])
    rows = []
    for key, paths in sorted(by_hash.items()):
        if len(paths) > 1:
            rows.append({"cluster_kind": "exact_hash", "key": key, "paths": sorted(paths)})
    for key, paths in sorted(by_normalized.items()):
        if len(paths) > 1:
            rows.append({"cluster_kind": "normalized_stem", "key": key, "paths": sorted(paths)})
    return rows


def golden(inventory: list[dict[str, Any]], complexity: list[dict[str, Any]], flows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    c_by_path = {row["path"]: row for row in complexity}
    f_by_path = {row["path"]: row for row in flows}
    rows = []
    for row in inventory:
        c = c_by_path[row["path"]]
        ncnn_score = max(0, 10 - min(6, c["nonblank_loc"] // 120) - (2 if c["unbounded_read_risk"] else 0))
        flow_score = max(0, 10 - (2 if not f_by_path.get(row["path"], {}).get("outputs") else 0) - (1 if c["branch_count"] > 35 else 0))
        rows.append({
            "path": row["path"],
            "ncnn_alignment": f"{ncnn_score}/10 directness/LOC/bounds heuristic",
            "flow_alignment": f"{flow_score}/10 receipt/dataflow heuristic",
            "line_pressure": f"{c['nonblank_loc']} nonblank LOC vs target {c['estimated_minimum_sane_loc']}",
        })
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows), encoding="utf-8")


def latest_copy(path: Path, latest_name: str) -> None:
    shutil.copyfile(path, OUT / latest_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Ouroboros whole-filesystem audit maps.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ts = stamp()
    files = iter_files()
    inv = script_inventory(files)
    complexity = linecount_complexity(inv)
    flows = dataflow(inv)
    maps = {
        f"script_inventory_{ts}.jsonl": inv,
        f"script_callgraph_{ts}.jsonl": callgraph(inv, files),
        f"script_dataflow_{ts}.jsonl": flows,
        f"duplication_clusters_{ts}.jsonl": duplication(inv),
        f"linecount_complexity_{ts}.jsonl": complexity,
        f"golden_comparison_{ts}.jsonl": golden(inv, complexity, flows),
        f"external_practice_lookup_{ts}.jsonl": [dict(row, script_path="CATEGORY", resulting_action="steal_pattern_not_dependency") for row in PUBLIC_PATTERNS],
    }
    fs_path = OUT / f"filesystem_map_{ts}.txt"
    fs_path.write_text("\n".join(rel(path) for path in files) + "\n", encoding="utf-8")
    latest_copy(fs_path, "filesystem_map_latest.txt")
    written = [fs_path]
    latest_names = {
        "script_inventory": "script_inventory_latest.jsonl",
        "script_callgraph": "script_callgraph_latest.jsonl",
        "script_dataflow": "script_dataflow_latest.jsonl",
        "linecount_complexity": "linecount_complexity_latest.jsonl",
    }
    for name, rows in maps.items():
        path = OUT / name
        write_jsonl(path, rows)
        written.append(path)
        for prefix, latest in latest_names.items():
            if name.startswith(prefix + "_"):
                latest_copy(path, latest)
    receipt = {
        "schema": "lucidota.ouroboros.filesystem_audit.v1",
        "generated_at": now(),
        "files_seen": len(files),
        "scripts_seen": len(inv),
        "outputs": [rel(p) for p in written],
        "latest_outputs": [rel(OUT / name) for name in ["filesystem_map_latest.txt", *latest_names.values()]],
        "excluded_dirs": sorted(EXCLUDE_DIRS),
        "status": "PASS",
    }
    receipt_path = OUT / f"audit_cycle_receipt_{ts}.json"
    receipt["report_path"] = rel(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(receipt, sort_keys=True))
    print("REPORT_PATH=" + rel(receipt_path))
    print("OUROBOROS_FILESYSTEM_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
