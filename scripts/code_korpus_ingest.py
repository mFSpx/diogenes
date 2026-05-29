#!/usr/bin/env python3
"""Legacy code corpus ingestion lane for LUCIDOTA.

This lane maps code artifacts into symbols, dependencies, side effects,
capability cards, and graph-staging candidates. It does not treat code as
generic prose.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "05_OUTPUTS" / "code_korpus"
RUNTIME = ROOT / "04_RUNTIME" / "code_korpus"
DEFAULT_INVENTORY = ROOT / "05_OUTPUTS" / "krampus_inventory" / "krampus_queue_eligible.jsonl"
EXCLUDED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".venv",
    "node_modules",
    "target",
    "03_VAULT",
}
INCLUDE_EXTS = {
    ".py",
    ".rs",
    ".sh",
    ".bash",
    ".sql",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".md",
}
ALGORITHM_KEYWORDS = {
    "embedding": {"embedding", "embed", "vector"},
    "graph": {"graph", "edge", "node", "callgraph"},
    "search": {"search", "query", "match"},
    "ranking": {"rank", "score", "priority"},
    "ocr": {"ocr", "tesseract", "vision"},
    "ingest": {"ingest", "import", "parse", "chunk"},
    "workflow": {"workflow", "queue", "spine", "lane"},
    "audit": {"audit", "receipt", "verdict", "gate"},
    "receipt": {"receipt", "report", "ledger"},
}
GRAPH_TABLE_PATTERNS = (
    "lucidota_go.graph_item",
    "lucidota_go.graph_edge",
    "lucidota_go.graph_journal",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def sha256_file(path: Path, *, max_bytes: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_text(path: Path, *, max_chars: int = 250_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    return any(name in parts for name in EXCLUDED_DIR_NAMES)


def classify_root(path: Path) -> str:
    parts = path.parts
    if parts and parts[0] in {"scripts", "ALGOS", "tests", "06_SCHEMA"}:
        return "active"
    if parts and parts[0] == "00_PROJECT_BRAIN":
        return "reference_only"
    if parts and parts[0] == "01_REPOS":
        return "vendor"
    if parts and parts[0] in {"04_RUNTIME"}:
        return "output"
    if parts and parts[0] in {"05_OUTPUTS"}:
        return "output"
    if parts and parts[0] in {"KRAMPUSCHEWING"}:
        return "archive"
    return "unknown"


def classify_language(path: Path) -> str:
    name = path.name.lower()
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext == ".rs":
        return "rust"
    if ext in {".sh", ".bash"} or name in {"bashrc", "profile"}:
        return "shell"
    if ext == ".sql":
        return "sql"
    if ext == ".json":
        return "json"
    if ext in {".yaml", ".yml"}:
        return "yaml"
    if ext == ".toml":
        return "toml"
    if ext == ".md":
        return "unknown"
    return "unknown"


def guess_algorithm_family(path: Path, text: str, symbols: Iterable[str]) -> str:
    hay = f"{path.as_posix()}\n{text}\n{' '.join(symbols)}".lower()
    for family, kws in ALGORITHM_KEYWORDS.items():
        if any(k in hay for k in kws):
            return family
    return "unknown"


def guess_authority_status(path: Path) -> str:
    low = path.as_posix().lower()
    if low.startswith("scripts/"):
        return "active"
    if low.startswith("algos/"):
        return "candidate"
    if low.startswith("tests/"):
        return "reference_only"
    if low.startswith("00_project_brain/"):
        return "reference_only"
    if low.startswith("01_repos/"):
        return "legacy"
    if low.startswith("06_schema/"):
        return "active"
    if low.startswith("krampuschewing/"):
        return "legacy"
    return "unknown"


def guess_promotion_status(authority_status: str, side_effects: list[str], runtime_probe: str) -> str:
    if "delete" in " ".join(side_effects) or "move" in " ".join(side_effects):
        return "quarantine"
    if "network" in side_effects or "model_call" in side_effects:
        return "needs_review"
    if authority_status in {"legacy", "reference_only"}:
        return "reference_only"
    if authority_status == "active" and runtime_probe == "imports_ok":
        return "keep_python"
    if authority_status == "candidate":
        return "port_to_rust" if "python" in side_effects else "keep_python"
    return "needs_review"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            count += 1
    return count


class PyFacts(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: list[str] = []
        self.symbols: list[str] = []
        self.calls: list[str] = []
        self.entrypoints: list[str] = []
        self.side_effects: set[str] = set()
        self.db_tables: set[str] = set()
        self.graph_tables: set[str] = set()
        self.files_read: set[str] = set()
        self.files_written: set[str] = set()
        self.subprocess_calls: set[str] = set()
        self.network_calls: set[str] = set()
        self.model_calls: set[str] = set()
        self.argparse_seen = False
        self.click_seen = False
        self.typer_seen = False
        self.has_main_guard = False

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        mod = node.module or ""
        for alias in node.names:
            self.imports.append(f"{mod}.{alias.name}".strip("."))
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self.symbols.append(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.symbols.append(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.symbols.append(node.name)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> Any:
        try:
            left = ast.unparse(node.test)
        except Exception:
            left = ""
        if "__name__" in left and "__main__" in left:
            self.has_main_guard = True
            self.entrypoints.append("if __name__ == '__main__'")
        self.generic_visit(node)

    def _call_name(self, node: ast.AST) -> str:
        try:
            return ast.unparse(node)
        except Exception:
            return node.__class__.__name__

    def visit_Call(self, node: ast.Call) -> Any:
        name = self._call_name(node.func)
        self.calls.append(name)
        low = name.lower()
        if any(x in low for x in ("argparse.", "parse_args")):
            self.argparse_seen = True
        if "click." in low:
            self.click_seen = True
        if "typer." in low:
            self.typer_seen = True
        if any(x in low for x in ("subprocess.", "os.system", "os.popen")):
            self.subprocess_calls.add(name)
            self.side_effects.add("subprocess")
        if any(x in low for x in ("requests.", "httpx.", "urllib.", "socket.")):
            self.network_calls.add(name)
            self.side_effects.add("network")
        if any(x in low for x in ("psycopg", "sqlite3", "duckdb", "sqlalchemy", "postgres")):
            self.model_calls.add(name) if "model" in low else None
            self.side_effects.add("db")
        if any(x in low for x in ("open(", "pathlib.path.write", ".write_text", ".write_bytes", "os.remove", "shutil.move")):
            if "remove" in low or "unlink" in low:
                self.side_effects.add("delete")
            if "move" in low or "rename" in low:
                self.side_effects.add("move")
            if ".write" in low or "write_text" in low or "write_bytes" in low:
                self.side_effects.add("file_write")
        if "graph_item" in low or "graph_edge" in low or "graph_journal" in low:
            self.graph_tables.update(t for t in GRAPH_TABLE_PATTERNS if t in low)
            self.side_effects.add("graph_write")
        if any(x in low for x in ("groq", "openai", "model_runner_cli", "llxprt", "model_invocation")):
            self.side_effects.add("model_call")
        self.generic_visit(node)


def parse_python(path: Path, text: str) -> dict[str, Any]:
    facts = PyFacts()
    try:
        tree = ast.parse(text, filename=str(path))
        facts.visit(tree)
        status = "imports_ok"
    except Exception as exc:
        return {
            "symbols": [],
            "imports": [],
            "entrypoints": [],
            "side_effects": ["parse_fail"],
            "db_tables_touched": [],
            "graph_tables_touched": [],
            "files_read": [],
            "files_written": [],
            "subprocess_calls": [],
            "network_calls": [],
            "model_calls": [],
            "runtime_probe": "import_fail",
            "runtime_error": repr(exc),
            "algorithm_family": "unknown",
            "cli_help_available": False,
            "tests_found": [],
        }
    side_effects = sorted(facts.side_effects)
    return {
        "symbols": sorted(set(facts.symbols)),
        "imports": sorted(set(facts.imports)),
        "entrypoints": sorted(set(facts.entrypoints)),
        "side_effects": side_effects,
        "db_tables_touched": sorted(set(facts.db_tables)),
        "graph_tables_touched": sorted(set(facts.graph_tables)),
        "files_read": sorted(facts.files_read),
        "files_written": sorted(facts.files_written),
        "subprocess_calls": sorted(set(facts.subprocess_calls)),
        "network_calls": sorted(set(facts.network_calls)),
        "model_calls": sorted(set(facts.model_calls)),
        "runtime_probe": status,
        "runtime_error": None,
        "algorithm_family": guess_algorithm_family(path, text, facts.symbols),
        "cli_help_available": bool(facts.argparse_seen or facts.click_seen or facts.typer_seen or facts.has_main_guard),
        "tests_found": [],
    }


def parse_shell(path: Path, text: str) -> dict[str, Any]:
    commands = []
    side_effects = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z0-9_./-]+)", stripped)
        if m:
            cmd = m.group(1)
            commands.append(cmd)
            if cmd in {"python", "python3", "bash", "sh"}:
                side_effects.add("subprocess")
            if cmd in {"rm", "mv", "cp", "dd", "ln"}:
                side_effects.add("file_mutation")
    return {
        "symbols": [],
        "imports": [],
        "entrypoints": ["shell_script"],
        "side_effects": sorted(side_effects),
        "db_tables_touched": [],
        "graph_tables_touched": [],
        "files_read": [],
        "files_written": [],
        "subprocess_calls": commands[:20],
        "network_calls": [],
        "model_calls": [cmd for cmd in commands if cmd in {"groq", "openai", "llxprt"}],
        "runtime_probe": "not_run",
        "runtime_error": None,
        "algorithm_family": guess_algorithm_family(path, text, commands),
        "cli_help_available": "--help" in text or "-h" in text,
        "tests_found": [],
    }


def parse_sql(path: Path, text: str) -> dict[str, Any]:
    tables = set(re.findall(r"(?i)\b(?:insert\s+into|update|delete\s+from|create\s+table)\s+([A-Za-z0-9_.\"]+)", text))
    side_effects = []
    if re.search(r"(?i)\binsert\s+into\b", text) or re.search(r"(?i)\bupdate\b", text) or re.search(r"(?i)\bdelete\s+from\b", text):
        side_effects.append("db_write")
    if any(t for t in tables if "graph_" in t.lower() or "lucidota_go." in t.lower()):
        side_effects.append("graph_write")
    return {
        "symbols": sorted(tables),
        "imports": [],
        "entrypoints": [],
        "side_effects": sorted(set(side_effects)),
        "db_tables_touched": sorted(tables),
        "graph_tables_touched": sorted([t for t in tables if "graph_" in t.lower() or "lucidota_go." in t.lower()]),
        "files_read": [],
        "files_written": [],
        "subprocess_calls": [],
        "network_calls": [],
        "model_calls": [],
        "runtime_probe": "not_run",
        "runtime_error": None,
        "algorithm_family": guess_algorithm_family(path, text, tables),
        "cli_help_available": False,
        "tests_found": [],
    }


def parse_generic(path: Path, text: str) -> dict[str, Any]:
    keys = re.findall(r"(?m)^\s*([A-Za-z0-9_.-]+)\s*[:=]", text)
    return {
        "symbols": keys[:50],
        "imports": [],
        "entrypoints": [],
        "side_effects": [],
        "db_tables_touched": [],
        "graph_tables_touched": [],
        "files_read": [],
        "files_written": [],
        "subprocess_calls": [],
        "network_calls": [],
        "model_calls": [],
        "runtime_probe": "not_run",
        "runtime_error": None,
        "algorithm_family": guess_algorithm_family(path, text, keys),
        "cli_help_available": False,
        "tests_found": [],
    }


def safe_probe(path: Path, language: str, parsed: dict[str, Any]) -> dict[str, Any]:
    if path.name.startswith("test_") and language == "python":
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", str(path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return {
            "runtime_probe": "imports_ok" if proc.returncode == 0 else "import_fail",
            "probe_error": proc.stderr[-2000:] if proc.returncode else None,
            "cli_help_available": False,
        }
    if language == "python":
        proc = subprocess.run([sys.executable, "-m", "py_compile", str(path)], cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            return {"runtime_probe": "import_fail", "probe_error": proc.stderr[-2000:], "cli_help_available": bool(parsed.get("cli_help_available"))}
        help_ok = False
        if parsed.get("cli_help_available"):
            try:
                proc2 = subprocess.run([sys.executable, str(path), "--help"], cwd=ROOT, capture_output=True, text=True, timeout=10)
                help_ok = proc2.returncode == 0
                return {"runtime_probe": "imports_ok", "probe_error": proc2.stderr[-2000:] if proc2.returncode else None, "cli_help_available": help_ok}
            except subprocess.TimeoutExpired as exc:
                return {
                    "runtime_probe": "imports_ok",
                    "probe_error": f"help_timeout:{exc}",
                    "cli_help_available": False,
                }
        return {"runtime_probe": "imports_ok", "probe_error": None, "cli_help_available": False}
    if language == "shell":
        proc = subprocess.run(["bash", "-n", str(path)], cwd=ROOT, capture_output=True, text=True)
        return {"runtime_probe": "help_ok" if proc.returncode == 0 else "help_fail", "probe_error": proc.stderr[-2000:] if proc.returncode else None, "cli_help_available": True}
    if path.name.startswith("test_") and language == "python":
        proc = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", str(path)], cwd=ROOT, capture_output=True, text=True, timeout=20)
        return {"runtime_probe": "imports_ok" if proc.returncode == 0 else "import_fail", "probe_error": proc.stderr[-2000:] if proc.returncode else None, "cli_help_available": False}
    return {"runtime_probe": parsed.get("runtime_probe", "not_run"), "probe_error": None, "cli_help_available": bool(parsed.get("cli_help_available"))}


def build_summary(path: Path, parsed: dict[str, Any], probe: dict[str, Any]) -> str:
    parts = [
        f"{path.name}",
        f"lang={classify_language(path)}",
        f"symbols={len(parsed.get('symbols') or [])}",
        f"imports={len(parsed.get('imports') or [])}",
        f"side_effects={','.join(parsed.get('side_effects') or []) or 'none'}",
        f"algo={parsed.get('algorithm_family') or 'unknown'}",
        f"probe={probe.get('runtime_probe')}",
    ]
    return " | ".join(parts)


def build_capability_card(row: dict[str, Any], parsed: dict[str, Any], probe: dict[str, Any], embed: dict[str, Any]) -> dict[str, Any]:
    side_effects = parsed.get("side_effects") or []
    authority_status = row["authority_status"]
    promotion_status = guess_promotion_status(authority_status, side_effects, probe.get("runtime_probe") or "not_run")
    can_run_now = probe.get("runtime_probe") in {"imports_ok", "help_ok"}
    return {
        "schema": "lucidota.code_korpus.capability_card.v1",
        "artifact_type": "LEGACY_CODE",
        "path": row["path"],
        "language": row["language"],
        "algorithm_name": parsed.get("algorithm_family") or "unknown",
        "callable_symbols": parsed.get("symbols") or [],
        "input_guess": "cli/file/config" if row["language"] in {"python", "shell", "sql"} else "unknown",
        "output_guess": "json/rows/files/side effects" if row["language"] in {"python", "shell", "sql"} else "unknown",
        "dependencies": parsed.get("imports") or [],
        "side_effects": side_effects,
        "can_run_now": can_run_now,
        "test_coverage": bool(row.get("tests_found")),
        "useful_for_lucidota": authority_status in {"active", "candidate", "legacy"},
        "recommended_fate": promotion_status,
        "embedding_status": embed.get("status"),
        "embedding_provider": embed.get("provider"),
        "embedding_model": embed.get("model"),
        "summary": build_summary(Path(row["path"]), parsed, probe),
        "evidence_refs": [row["path"]],
    }


def inventory_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if is_excluded(path):
            continue
        if path.suffix.lower() not in INCLUDE_EXTS and path.name not in {"Makefile", "Dockerfile"}:
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.as_posix())


def load_inventory(cache_path: Path | None, root: Path) -> list[dict[str, Any]]:
    if cache_path and cache_path.exists():
        rows: list[dict[str, Any]] = []
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows
    rows = []
    for path in inventory_files(root):
        st = path.stat()
        rows.append(
            {
                "path": rel(path),
                "sha256": sha256_file(path),
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
                "size_bytes": st.st_size,
                "language": classify_language(path),
                "root_class": classify_root(path),
            }
        )
    return rows


def is_code_inventory_row(row: dict[str, Any]) -> bool:
    path = Path(str(row.get("path") or ""))
    name = path.name
    suffix = path.suffix.lower()
    return suffix in INCLUDE_EXTS or name in {"Makefile", "Dockerfile"}


def select_batch(rows: list[dict[str, Any]], start_after: str, batch_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_start = not start_after
    for row in rows:
        rp = row["path"]
        if not seen_start:
            if rp == start_after:
                seen_start = True
            continue
        if rp == start_after:
            continue
        selected.append(row)
        if len(selected) >= batch_size:
            break
    cursor_after = selected[-1]["path"] if selected else start_after
    return selected, {
        "cursor_before": start_after,
        "cursor_after": cursor_after,
        "selected_count": len(selected),
        "inventory_count": len(rows),
    }


def analyze_path(path: Path) -> dict[str, Any]:
    text = load_text(path)
    language = classify_language(path)
    if language == "python":
        parsed = parse_python(path, text)
    elif language == "shell":
        parsed = parse_shell(path, text)
    elif language == "sql":
        parsed = parse_sql(path, text)
    else:
        parsed = parse_generic(path, text)
    probe = safe_probe(path, language, parsed)
    imports = parsed.get("imports") or []
    symbols = parsed.get("symbols") or []
    side_effects = parsed.get("side_effects") or []
    tests_found = []
    if path.name.startswith("test_"):
        tests_found.append(path.as_posix())
    summary = build_summary(path, parsed, probe)
    authority_status = guess_authority_status(path)
    promotion_status = guess_promotion_status(authority_status, side_effects, probe.get("runtime_probe") or "not_run")
    return {
        "path": rel(path),
        "sha256": sha256_file(path),
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
        "language": language,
        "root_class": classify_root(path),
        "imports": imports,
        "symbols": symbols,
        "entrypoints": parsed.get("entrypoints") or [],
        "cli_help_available": bool(probe.get("cli_help_available")),
        "tests_found": tests_found,
        "side_effects": side_effects,
        "db_tables_touched": parsed.get("db_tables_touched") or [],
        "graph_tables_touched": parsed.get("graph_tables_touched") or [],
        "files_read": parsed.get("files_read") or [],
        "files_written": parsed.get("files_written") or [],
        "subprocess_calls": parsed.get("subprocess_calls") or [],
        "network_calls": parsed.get("network_calls") or [],
        "model_calls": parsed.get("model_calls") or [],
        "algorithm_family": parsed.get("algorithm_family") or "unknown",
        "runtime_probe": probe.get("runtime_probe") or parsed.get("runtime_probe") or "not_run",
        "authority_status": authority_status,
        "promotion_status": promotion_status,
        "summary": summary,
        "evidence_refs": [rel(path)],
        "probe_error": probe.get("probe_error"),
        "parse_error": parsed.get("runtime_error"),
    }


def build_edges(code_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    symbol_to_paths: dict[str, list[str]] = defaultdict(list)
    for row in code_rows:
        for sym in row.get("symbols") or []:
            symbol_to_paths[sym].append(row["path"])
    dep_edges: list[dict[str, Any]] = []
    side_effect_edges: list[dict[str, Any]] = []
    for row in code_rows:
        path = row["path"]
        for imp in row.get("imports") or []:
            dep_edges.append({"schema": "lucidota.code_korpus.edge.v1", "edge_type": "IMPORTS", "source": path, "target": imp, "evidence_refs": [path]})
        for sym in row.get("symbols") or []:
            dep_edges.append({"schema": "lucidota.code_korpus.edge.v1", "edge_type": "DEFINES", "source": path, "target": sym, "evidence_refs": [path]})
            for caller in symbol_to_paths.get(sym, [])[:10]:
                if caller != path:
                    dep_edges.append({"schema": "lucidota.code_korpus.edge.v1", "edge_type": "CALLED_BY", "source": sym, "target": caller, "evidence_refs": [path, caller]})
        for table in row.get("db_tables_touched") or []:
            side_effect_edges.append({"schema": "lucidota.code_korpus.edge.v1", "edge_type": "TOUCHES_DB_TABLE", "source": path, "target": table, "evidence_refs": [path]})
        for table in row.get("graph_tables_touched") or []:
            side_effect_edges.append({"schema": "lucidota.code_korpus.edge.v1", "edge_type": "TOUCHES_GRAPH_TABLE", "source": path, "target": table, "evidence_refs": [path]})
        for test in row.get("tests_found") or []:
            dep_edges.append({"schema": "lucidota.code_korpus.edge.v1", "edge_type": "TESTED_BY", "source": path, "target": test, "evidence_refs": [path, test]})
    return dep_edges, side_effect_edges


def main() -> int:
    ap = argparse.ArgumentParser(description="CODE_KORPUS / legacy code ingest lane")
    ap.add_argument("--root", default=".")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--start-after", default="")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--inventory-jsonl", default="")
    args = ap.parse_args()

    root = (ROOT / args.root).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    timestamp = stamp()
    inv_path = OUT / f"code_inventory_{timestamp}.jsonl"
    symbols_path = OUT / f"code_symbols_{timestamp}.jsonl"
    dep_path = OUT / f"code_dependency_edges_{timestamp}.jsonl"
    side_path = OUT / f"code_side_effects_{timestamp}.jsonl"
    cards_path = OUT / f"code_capability_cards_{timestamp}.jsonl"
    graph_path = OUT / f"code_graph_candidates_{timestamp}.jsonl"
    river_path = OUT / f"code_river_rows_{timestamp}.jsonl"
    receipt_path = OUT / f"code_ingest_receipt_{timestamp}.json"
    cursor_path = RUNTIME / "cursor.json"

    inventory_cache = Path(args.inventory_jsonl) if args.inventory_jsonl else (DEFAULT_INVENTORY if DEFAULT_INVENTORY.exists() else None)
    inventory_rows = load_inventory(inventory_cache, root)
    inventory_rows = [row for row in inventory_rows if is_code_inventory_row(row)]
    selected, cursor_meta = select_batch(inventory_rows, args.start_after, int(args.batch_size))

    probe_ok = 0
    probe_fail = 0
    parsed_rows: list[dict[str, Any]] = []
    symbol_rows: list[dict[str, Any]] = []
    capability_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    river_rows: list[dict[str, Any]] = []
    side_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    counts = Counter()
    from embedding_provider import embed_text as provider_embed_text, probe as probe_embedding_provider

    provider_probe = probe_embedding_provider()

    for row in selected:
        counts["files_seen"] += 1
        path = root / row["path"]
        try:
            analysis = analyze_path(path)
            parsed_rows.append(analysis)
            counts["code_files"] += 1
            counts["parsed"] += 1
            counts["symbols"] += len(analysis.get("symbols") or [])
            counts["dependencies"] += len(analysis.get("imports") or [])
            counts["side_effects"] += len(analysis.get("side_effects") or [])
            if analysis.get("runtime_probe") in {"imports_ok", "help_ok"}:
                counts["import_ok"] += 1
            elif analysis.get("runtime_probe") == "import_fail":
                counts["import_fail"] += 1
            if analysis.get("runtime_probe") == "help_ok":
                counts["help_ok"] += 1
            elif analysis.get("runtime_probe") == "help_fail":
                counts["help_fail"] += 1
            if analysis.get("authority_status") == "legacy" or analysis.get("promotion_status") == "quarantine":
                counts["quarantined"] += 1
            if analysis.get("promotion_status") == "port_to_rust":
                counts["port_to_rust_candidates"] += 1

            summary = analysis["summary"]
            embed_result = provider_embed_text(summary, source_path=analysis["path"], chunk_id=analysis["path"], prefer_groq=True)
            embed_row = embed_result.row
            if embed_row.get("status") == "EMBEDDED" and embed_row.get("provider") == "groq":
                counts["embedded_groq"] += 1
            elif embed_row.get("status") == "EMBEDDED" and embed_row.get("provider") == "local":
                counts["embedded_local"] += 1
            elif embed_row.get("status") == "BLOCKED":
                counts["embedding_blocked"] += 1
            elif embed_row.get("status") == "FAILED":
                counts["embedding_failed"] += 1
            else:
                counts["embedding_skipped"] += 1
            capability = build_capability_card(analysis, parsed=analysis, probe=analysis, embed=embed_row)
            capability_rows.append(capability)
            symbol_rows.extend(
                {
                    "schema": "lucidota.code_korpus.symbol.v1",
                    "path": analysis["path"],
                    "symbol": sym,
                    "kind": "function_or_symbol",
                    "language": analysis["language"],
                    "evidence_refs": [analysis["path"]],
                }
                for sym in analysis.get("symbols") or []
            )
            side_rows.extend(
                {
                    "schema": "lucidota.code_korpus.side_effect.v1",
                    "path": analysis["path"],
                    "side_effect": effect,
                    "language": analysis["language"],
                    "evidence_refs": [analysis["path"]],
                }
                for effect in analysis.get("side_effects") or []
            )
        except Exception as exc:
            counts["failed"] += 1
            errors.append(f"{row['path']}:{type(exc).__name__}:{exc}")

    dep_rows, side_edges = build_edges(parsed_rows)
    graph_rows.extend(
        {
            "schema": "lucidota.code_korpus.graph_candidate.v1",
            "candidate_type": "CODE_ARTIFACT",
            "path": row["path"],
            "labels": [row["language"], row["root_class"], row["authority_status"]],
            "evidence_refs": [row["path"]],
        }
        for row in parsed_rows
    )
    graph_rows.extend(
        {
            "schema": "lucidota.code_korpus.graph_candidate.v1",
            "candidate_type": "CAPABILITY",
            "path": row["path"],
            "labels": [row["algorithm_family"], row["promotion_status"]],
            "evidence_refs": [row["path"]],
        }
        for row in parsed_rows
    )
    graph_rows.extend(
        {
            "schema": "lucidota.code_korpus.graph_candidate.v1",
            "candidate_type": edge["edge_type"],
            "path": edge["source"],
            "labels": [edge["edge_type"]],
            "evidence_refs": edge["evidence_refs"],
        }
        for edge in dep_rows + side_edges
    )
    river_rows.extend(
        {
            "schema": "lucidota.code_korpus.river_row.v1",
            "path": row["path"],
            "event_type": "CODE_CAPABILITY_CARD",
            "summary": row["summary"],
            "runtime_probe": row["runtime_probe"],
            "promotion_status": row["promotion_status"],
            "evidence_refs": row["evidence_refs"],
        }
        for row in parsed_rows
    )

    write_jsonl(inv_path, selected)
    write_jsonl(symbols_path, symbol_rows)
    write_jsonl(dep_path, dep_rows)
    write_jsonl(side_path, side_rows)
    write_jsonl(cards_path, capability_rows)
    write_jsonl(graph_path, graph_rows)
    write_jsonl(river_path, river_rows)
    cursor_payload = {
        "schema": "lucidota.code_korpus.cursor.v1",
        "generated_at_utc": now(),
        "cursor": cursor_meta["cursor_after"],
        "start_after": args.start_after,
        "selected_count": cursor_meta["selected_count"],
        "inventory_count": cursor_meta["inventory_count"],
    }
    cursor_path.write_text(json.dumps(cursor_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

    receipt = {
        "schema": "lucidota.code_korpus.ingest_run.v1",
        "verdict": "PASS" if selected and not errors else ("PARTIAL_PASS" if selected else "PARTIAL_FAIL"),
        "generated_at_utc": now(),
        "root": rel(root),
        "execute": bool(args.execute),
        "batch_size": int(args.batch_size),
        "files_seen": counts["files_seen"],
        "code_files": counts["code_files"],
        "parsed": counts["parsed"],
        "symbols": counts["symbols"],
        "dependencies": counts["dependencies"],
        "side_effects": counts["side_effects"],
        "capability_cards": len(capability_rows),
        "graph_candidates": len(graph_rows),
        "river_rows": len(river_rows),
        "import_ok": counts["import_ok"],
        "import_fail": counts["import_fail"],
        "help_ok": counts["help_ok"],
        "help_fail": counts["help_fail"],
        "quarantined": counts["quarantined"],
        "port_to_rust_candidates": counts["port_to_rust_candidates"],
        "embedded_groq": counts["embedded_groq"],
        "embedded_local": counts["embedded_local"],
        "embedding_blocked": counts["embedding_blocked"],
        "embedding_failed": counts["embedding_failed"],
        "embedding_skipped": counts["embedding_skipped"],
        "embedding_provider_probe": {
            "verdict": provider_probe.get("verdict"),
            "provider": provider_probe.get("provider"),
            "blocked_gap": provider_probe.get("blocked_gap"),
            "groq": provider_probe.get("groq"),
        },
        "paths": {
            "inventory": rel(inv_path),
            "symbols": rel(symbols_path),
            "dependencies": rel(dep_path),
            "side_effects": rel(side_path),
            "capability_cards": rel(cards_path),
            "graph_candidates": rel(graph_path),
            "river_rows": rel(river_path),
            "cursor": rel(cursor_path),
        },
        "errors": errors[:20],
        "current_authority": {
            "active": [r for r in parsed_rows if r.get("authority_status") == "active"][:20],
            "superseded": [r for r in parsed_rows if r.get("authority_status") == "legacy"][:20],
            "rejected": [r for r in parsed_rows if r.get("promotion_status") == "quarantine"][:20],
            "conflicts": [],
        },
        "canonical_graph_writes": False,
        "db_writes": False,
        "external_effects": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(f"RECEIPT_PATH={rel(receipt_path)}")
    return 0 if receipt["verdict"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
