#!/usr/bin/env python3
"""TICKLETRUNK proof-hoard scanner and access-layer builder.

Default mode is dry-run. The scanner never deletes, moves, renames, or edits
sovereign toolbox artifacts. Execute mode writes only manifest/access/report files.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()
OUT_DIR = ROOT / "05_OUTPUTS" / "tickletrunk"
MANIFEST_JSON = ROOT / "00_PROJECT_BRAIN" / "TICKLETRUNK.json"
MANIFEST_MD = ROOT / "00_PROJECT_BRAIN" / "TICKLETRUNK.md"
TOOLS_DIR = ROOT / "TOOLS"

CATEGORIES = [
    "ALGOS", "SCRIPTS", "MODELS", "LORAS", "SCHEMAS", "SKILLS", "PLUGINS",
    "SERVICES", "BOOKS", "SURFACES", "SCRAPERS", "WORKFLOWS", "REPOS",
    "RUNTIME", "VAULT", "OTHER",
]
ENTRY_FIELDS = [
    "id", "name", "path", "relative_path", "realpath", "category", "kind",
    "what_it_is", "what_it_does", "one_line_description", "inputs", "outputs",
    "entrypoints", "imports", "depends_on", "known_uses", "known_use_count",
    "status", "sovereignty", "proof_hoard_role", "last_modified", "last_seen",
    "last_used", "size_bytes", "language_or_format", "tags", "combines_well_with",
    "notes", "warnings",
]
PROOF_HOARD_ROLES = {"experiment", "reusable_prior", "candidate_for_promotion", "production_copy", "reference", "unknown"}
HARD_LAW = "TICKLETRUNK / PROOF HOARD SOVEREIGNTY"
OPERATOR_PURPOSE = "Canonical manifest and access layer for the operator's proof hoard: every sovereign tool, algo, model, workflow, book, LoRA, scraper, skill, plugin, service, surface, schema, reusable fragment, and weird experimental instrument in the filesystem."
READ_FIRST_RULE = "Any system touching LUCIDOTA must read this manifest before writing new tools, scripts, workflows, integrations, adapters, schemas, model hooks, scrapers, skills, plugins, or services."

CODE_EXT = {".py", ".sh", ".rs", ".sql", ".js", ".ts", ".tsx", ".jsx", ".go", ".c", ".cpp", ".h", ".hpp"}
TEXT_EXT = CODE_EXT | {".md", ".json", ".toml", ".yaml", ".yml", ".html", ".css", ".txt", ".schema", ".jsonl"}
ASSET_EXT = {".gguf", ".safetensors", ".pt", ".pth", ".onnx", ".bin", ".epub", ".mobi", ".pdf", ".csv", ".sqlite", ".db", ".tl", ".pkl"}
SKIP_DIR_NAMES = {".git", "node_modules", "__pycache__", ".venv", "venv", "env", "target", "build", "dist", ".cache", ".pytest_cache", ".mypy_cache"}
SKIP_FILE_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"(^|/|\\)auth\.json$", r"(^|/|\\)history\.jsonl$", r"(^|/|\\)logs?_?\d*\.sqlite$",
        r"(^|/|\\)\.env(\..*)?$", r"id_rsa$", r"id_ed25519$", r"private[_-]?key",
        r"token", r"credential", r"secret", r"cookie", r"session",
    ]
]
PRIMARY_CANDIDATES: dict[str, list[str]] = {
    "ALGOS": ["ALGOS"],
    "SCRIPTS": ["scripts", "01_REPOS/llama.cpp/scripts", "01_REPOS/doggystyle/scripts"],
    "MODELS": ["03_VAULT/models", "01_REPOS/llama.cpp/models"],
    "LORAS": ["04_RUNTIME/lora_cartridges", "01_REPOS/llama.cpp/tools/export-lora"],
    "SCHEMAS": ["06_SCHEMA", "BOOKS/ontology_packs"],
    "SKILLS": [".codex/skills", str(HOME / ".codex" / "skills")],
    "PLUGINS": [".codex/plugins", str(HOME / ".codex" / "plugins"), str(HOME / ".claw" / "plugins")],
    "SERVICES": ["services"],
    "BOOKS": ["BOOKS"],
    "SURFACES": ["07_SURFACES"],
    "SCRAPERS": ["SCRAPERS", "scripts"],
    "WORKFLOWS": ["WORKFLOWS", "00_PROJECT_BRAIN", "06_SCHEMA/006_workflow_registry.sql"],
    "REPOS": ["01_REPOS"],
    "RUNTIME": ["04_RUNTIME"],
    "VAULT": ["03_VAULT"],
    "OTHER": ["src", "tests"],
}
GUESSED_NAME_MAP = {
    "TOOLS": ["TOOLS", "ALGOS", "scripts", "01_REPOS/llama.cpp/tools", "01_REPOS/doggystyle/scripts"],
    "MODELS": ["03_VAULT/models", "01_REPOS/llama.cpp/models"],
    "WORKFLOWS": ["06_SCHEMA/006_workflow_registry.sql", "00_PROJECT_BRAIN/*WORKFLOW*", "scripts/*workflow*"],
    "LoRAs": ["04_RUNTIME/lora_cartridges", "01_REPOS/llama.cpp/tools/export-lora"],
    "SCRAPERS": ["scripts/lucidota_browser_body_capture.py", "scripts/lucidota_body_capture_evidence.py", "scripts/lucidota_security_scan.py"],
}
SCRAPER_HINTS = ("scrape", "scraper", "survey", "browser", "capture", "crawl", "fetch", "security_scan")
WORKFLOW_HINTS = ("workflow", "absurd", "queue", "dispatch", "watcher", "runbook")
CURATED_FIELDS = {"what_it_is", "what_it_does", "one_line_description", "inputs", "outputs", "known_uses", "status", "sovereignty", "proof_hoard_role", "tags", "combines_well_with", "notes"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def slug(value: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return s[:96] or hashlib.sha256(value.encode()).hexdigest()[:16]


def is_sensitive_path(path: Path) -> bool:
    s = str(path)
    return any(p.search(s) for p in SKIP_FILE_PATTERNS)


def language(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".py":"python", ".sh":"shell", ".rs":"rust", ".sql":"sql", ".md":"markdown",
        ".json":"json", ".jsonl":"jsonl", ".toml":"toml", ".yaml":"yaml", ".yml":"yaml",
        ".gguf":"gguf", ".safetensors":"safetensors", ".epub":"epub", ".mobi":"mobi", ".pdf":"pdf",
        ".html":"html", ".css":"css", ".csv":"csv", ".tl":"treelite", ".pkl":"pickle",
    }.get(ext, ext.lstrip(".") or "directory")


def kind_for(path: Path, category: str) -> str:
    name = path.name.lower()
    if category == "ALGOS": return "algo"
    if category == "SCRIPTS": return "script"
    if category == "MODELS": return "model"
    if category == "LORAS": return "lora"
    if category == "SCHEMAS": return "schema"
    if category == "SKILLS": return "skill"
    if category == "PLUGINS": return "plugin"
    if category == "SERVICES": return "service"
    if category == "BOOKS": return "book"
    if category == "SURFACES": return "surface"
    if category == "SCRAPERS": return "scraper"
    if category == "WORKFLOWS": return "workflow"
    if category == "REPOS": return "repo_tool"
    if category == "RUNTIME": return "runtime_asset"
    if category == "VAULT": return "runtime_asset" if "model" not in name else "model"
    return "other"


def role_for(path: Path, category: str, kind: str) -> str:
    if category == "ALGOS": return "experiment"
    if kind in {"schema", "book", "model", "skill", "plugin"}: return "reference"
    if category in {"SCRIPTS", "WORKFLOWS", "SCRAPERS"}: return "reusable_prior"
    if category in {"SERVICES", "SURFACES", "REPOS"}: return "candidate_for_promotion"
    if category in {"RUNTIME", "VAULT"}: return "reference"
    return "unknown"


def status_for(path: Path, category: str) -> str:
    if category in {"SCHEMAS", "SCRIPTS"}: return "sandbox"
    if category == "REPOS": return "sandbox"
    if category == "ALGOS": return "sandbox"
    return "unknown"


def read_small_text(path: Path, max_bytes: int = 80_000) -> str:
    if not path.is_file() or path.stat().st_size > max_bytes or is_sensitive_path(path):
        return ""
    if path.suffix.lower() not in TEXT_EXT and path.name.upper() not in {"README", "README.MD"}:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def infer_description(path: Path) -> tuple[str, str, list[str], list[str]]:
    warnings: list[str] = []
    imports: list[str] = []
    text = read_small_text(path)
    what = "UNKNOWN — needs operator label"
    if path.is_dir():
        for readme in [path / "README.md", path / "README", path / "ARCHITECTURE.md"]:
            if readme.exists():
                rt = read_small_text(readme, 40_000)
                for line in rt.splitlines():
                    clean = line.strip("# -*\t")
                    if clean and len(clean) > 8:
                        what = clean[:300]
                        break
                break
    elif path.suffix == ".py" and text:
        try:
            node = ast.parse(text)
            doc = ast.get_docstring(node)
            if doc:
                what = " ".join(doc.strip().split())[:300]
            for item in node.body:
                if isinstance(item, (ast.Import, ast.ImportFrom)):
                    if isinstance(item, ast.Import):
                        imports.extend(a.name for a in item.names)
                    else:
                        imports.append(item.module or "")
        except Exception as exc:
            warnings.append(f"python_parse_warning:{exc}")
    elif path.suffix in {".rs", ".sql", ".sh", ".js", ".ts", ".md"} and text:
        for line in text.splitlines()[:40]:
            clean = line.strip().lstrip("#/-*;! ").strip()
            if clean and not clean.lower().startswith(("usr/bin", "env ", "set -", "create ", "insert ", "import ", "from ")):
                what = clean[:300]
                break
    elif path.suffix.lower() in ASSET_EXT or path.parent.name.lower() in {"models", "books"}:
        what = f"Reference asset/file: {path.name}"
    if what == "UNKNOWN — needs operator label":
        stem = path.stem.replace("_", " ").replace("-", " ").strip()
        if stem:
            what = f"UNKNOWN — needs operator label; filename suggests: {stem[:160]}"
    one = what.split(". ")[0][:180]
    return what, one, imports[:20], warnings


def should_index(path: Path, category: str) -> bool:
    if is_sensitive_path(path):
        return False
    if path.is_dir():
        return path.name not in SKIP_DIR_NAMES
    if path.name.startswith(".") and path.name not in {".gitignore", ".editorconfig"}:
        return False
    ext = path.suffix.lower()
    if ext in {".pyc", ".log", ".pid", ".tmp", ".lock"}:
        return False
    if category == "RUNTIME" and ext in {".log", ".pid"}:
        return False
    if category == "VAULT" and ("cas" in path.parts or "secret_quarantine" in path.parts):
        return False
    return ext in TEXT_EXT or ext in ASSET_EXT or os.access(path, os.X_OK)


def discover_roots() -> tuple[dict[str, list[Path]], list[str], list[str]]:
    roots: dict[str, list[Path]] = {c: [] for c in CATEGORIES}
    missing: list[str] = []
    warnings: list[str] = []
    for cat, candidates in PRIMARY_CANDIDATES.items():
        for c in candidates:
            p = Path(c)
            if not p.is_absolute():
                p = ROOT / p
            if p.exists():
                roots[cat].append(p)
            else:
                missing.append(f"{cat}:{c}")
    # extra repo-root one/two-level toolbox-ish directories
    for p in ROOT.iterdir():
        if not p.is_dir() or p.name in SKIP_DIR_NAMES or p.name.startswith("."):
            continue
        lname = p.name.lower()
        if p not in sum(roots.values(), []):
            if any(k in lname for k in ["tool", "algo", "model", "workflow", "book", "lora", "scraper", "service", "surface", "schema", "repo", "runtime", "vault"]):
                roots["OTHER"].append(p)
    return roots, sorted(set(missing)), warnings


def iter_tool_entries(roots: dict[str, list[Path]]) -> tuple[list[Path], list[dict[str, Any]]]:
    skipped: list[Path] = []
    entries_raw: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cat, paths in roots.items():
        for base in paths:
            if is_sensitive_path(base):
                skipped.append(base)
                continue
            if base.is_file():
                candidates = [base]
            else:
                candidates = [base]
                max_depth = 4
                if cat == "REPOS":
                    max_depth = 2
                if cat in {"RUNTIME", "VAULT"}:
                    max_depth = 2
                for dirpath, dirnames, filenames in os.walk(base, topdown=True, followlinks=False):
                    d = Path(dirpath)
                    try:
                        depth = len(d.relative_to(base).parts)
                    except Exception:
                        depth = 0
                    dirnames[:] = [x for x in dirnames if x not in SKIP_DIR_NAMES and not is_sensitive_path(d / x)]
                    if cat == "VAULT":
                        dirnames[:] = [x for x in dirnames if x not in {"cas", "secret_quarantine", "drive_map"}]
                    if depth >= max_depth:
                        dirnames[:] = []
                    for fn in filenames:
                        p = d / fn
                        # REPOS are indexed as repos/toolpacks, not full vendored source trees.
                        if cat == "REPOS" and depth > 1 and fn not in {"README.md", "Cargo.toml", "pyproject.toml", "package.json"}:
                            continue
                        if should_index(p, cat):
                            candidates.append(p)
            for p in candidates:
                key = str(p.resolve())
                if key in seen or not should_index(p, cat):
                    continue
                seen.add(key)
                entries_raw.append({"path": p, "category": cat})
    return skipped, entries_raw


def build_text_index() -> list[tuple[str, str]]:
    dirs = [ROOT / "00_PROJECT_BRAIN", ROOT / "scripts", ROOT]
    files: list[Path] = []
    for d in dirs:
        if not d.exists(): continue
        if d.is_file(): files.append(d); continue
        for dirpath, dirnames, filenames in os.walk(d, topdown=True):
            p = Path(dirpath)
            if len(p.relative_to(d).parts) > 3:
                dirnames[:] = []
            dirnames[:] = [x for x in dirnames if x not in SKIP_DIR_NAMES]
            for fn in filenames:
                f = p / fn
                if f.suffix.lower() in {".md", ".py", ".sh", ".sql", ".json", ".toml"} and f.stat().st_size < 300_000 and not is_sensitive_path(f):
                    files.append(f)
    out: list[tuple[str, str]] = []
    for f in sorted(set(files))[:700]:
        txt = read_small_text(f, 300_000)
        if txt:
            out.append((rel(f), txt.lower()))
    return out


def known_uses_for(path: Path, text_index: list[tuple[str, str]]) -> list[str]:
    if path.is_dir() or path.suffix.lower() in ASSET_EXT:
        return []
    needles = {path.name.lower(), rel(path).lower()}
    if path.stem:
        needles.add(path.stem.lower())
    uses: list[str] = []
    for r, txt in text_index:
        if r == rel(path):
            continue
        if any(n and n in txt for n in needles):
            uses.append(r)
            if len(uses) >= 10:
                break
    return uses


def make_entry(path: Path, category: str, text_index: list[tuple[str, str]]) -> dict[str, Any]:
    st = path.stat()
    if category == "SCRIPTS" and any(h in path.name.lower() for h in SCRAPER_HINTS):
        category = "SCRAPERS"
    kind = kind_for(path, category)
    what, one, imports, warnings = infer_description(path)
    entrypoints: list[str] = []
    if path.is_file() and (os.access(path, os.X_OK) or path.suffix.lower() in {".py", ".sh", ".rs", ".sql"}):
        entrypoints.append(rel(path))
    tags = [category.lower(), kind]
    lname = path.name.lower()
    if any(h in lname for h in SCRAPER_HINTS): tags.append("scraper_hint")
    if any(h in lname for h in WORKFLOW_HINTS): tags.append("workflow_hint")
    uses = known_uses_for(path, text_index)
    return {
        "id": slug(rel(path)),
        "name": path.name,
        "path": str(path),
        "relative_path": rel(path),
        "realpath": str(path.resolve()),
        "category": category,
        "kind": kind,
        "what_it_is": one,
        "what_it_does": what,
        "one_line_description": one,
        "inputs": None,
        "outputs": None,
        "entrypoints": entrypoints,
        "imports": sorted(set(x for x in imports if x))[:25],
        "depends_on": [],
        "known_uses": uses,
        "known_use_count": len(uses),
        "status": status_for(path, category),
        "sovereignty": "sovereign_toolbox",
        "proof_hoard_role": role_for(path, category, kind),
        "last_modified": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z"),
        "last_seen": utc_now(),
        "last_used": None,
        "size_bytes": st.st_size if path.is_file() else 0,
        "language_or_format": language(path),
        "tags": sorted(set(tags)),
        "combines_well_with": [],
        "notes": "",
        "warnings": warnings,
    }


def load_existing() -> dict[str, Any] | None:
    if not MANIFEST_JSON.exists(): return None
    try:
        return json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    except Exception:
        return None


def merge_existing(entries: list[dict[str, Any]], append: bool) -> list[dict[str, Any]]:
    old = load_existing()
    if not old: return entries
    by_key: dict[str, dict[str, Any]] = {}
    for cat_entries in old.get("toolboxes", {}).values():
        for e in cat_entries:
            key = e.get("realpath") or e.get("relative_path") or e.get("id")
            if key: by_key[str(key)] = e
    current_keys=set()
    merged=[]
    for e in entries:
        key = e.get("realpath") or e.get("relative_path") or e.get("id")
        current_keys.add(str(key))
        olde = by_key.get(str(key))
        if olde:
            for f in CURATED_FIELDS:
                oldv = olde.get(f)
                if oldv not in (None, "", [], {}, "UNKNOWN — needs operator label", "unknown"):
                    if f in {"tags", "combines_well_with", "known_uses"} and isinstance(oldv, list):
                        e[f] = sorted(set(oldv) | set(e.get(f, [])))
                    elif f in {"what_it_is", "what_it_does", "one_line_description"}:
                        if str(e.get(f, "")).startswith("UNKNOWN") or not e.get(f):
                            e[f] = oldv
                    else:
                        e[f] = oldv
            if olde.get("not_seen_this_scan"):
                e.pop("not_seen_this_scan", None)
        merged.append(e)
    if append:
        for key, olde in by_key.items():
            if key not in current_keys:
                olde["not_seen_this_scan"] = True
                merged.append(olde)
    return merged


def build_manifest(entries: list[dict[str, Any]], missing: list[str], skipped: list[Path], warnings: list[str]) -> dict[str, Any]:
    boxes = {c: [] for c in CATEGORIES}
    for e in sorted(entries, key=lambda x: (x["category"], x["relative_path"])):
        boxes.setdefault(e["category"], []).append(e)
    return {
        "schema_version": "1.0",
        "generated_at": utc_now(),
        "generator": "scripts/tickletrunk_scan.py",
        "repo_root": str(ROOT),
        "total_tools": len(entries),
        "hard_law": "TICKLETRUNK",
        "operator_purpose": OPERATOR_PURPOSE,
        "read_first_rule": READ_FIRST_RULE,
        "proof_hoard_doctrine": "Index the jungle; do not pave it. The proof hoard is made findable, not respectable. Weird, disconnected, experimental, and useful-later artifacts are allowed.",
        "toolboxes": boxes,
        "guessed_name_map": GUESSED_NAME_MAP,
        "skipped": [str(p) for p in skipped] + missing,
        "warnings": warnings,
    }


def render_md(manifest: dict[str, Any]) -> str:
    lines = [
        "# TICKLETRUNK — LUCIDOTA PROOF HOARD", "",
        manifest["operator_purpose"], "",
        "## Hard Law Summary", "",
        "- TICKLETRUNK makes the proof hoard findable, not respectable.",
        "- The scattered folders are not junk, dead code, or production dependencies merely because they exist.",
        "- Before writing new tools, search TICKLETRUNK first and copy/adapt from existing work when useful.",
        "- Never delete, rename, move, normalize, or production-gate sovereign toolbox artifacts without explicit operator instruction naming the exact target.",
        "- If an artifact graduates, copy/adapt it into production and harden the copy; the original remains sovereign.", "",
        "## Regenerate", "", "```bash", "python3 scripts/tickletrunk_scan.py --execute", "```", "",
        "## Query Examples", "", "```bash", "python3 scripts/tickletrunk_scan.py --query krampus", "python3 scripts/tickletrunk_scan.py --category ALGOS", "python3 scripts/tickletrunk_scan.py --list", "```", "",
        "## Category Table", "", "| Category | Count |", "|---|---:|",
    ]
    for cat in CATEGORIES:
        lines.append(f"| {cat} | {len(manifest['toolboxes'].get(cat, []))} |")
    unknowns=[]
    for cat in CATEGORIES:
        entries=manifest["toolboxes"].get(cat, [])
        lines += ["", f"## {cat}", "", "| name | kind | status | proof_hoard_role | path | what_it_does |", "|---|---|---|---|---|---|"]
        for e in entries:
            what=str(e.get("what_it_does", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {e.get('name','')} | {e.get('kind','')} | {e.get('status','')} | {e.get('proof_hoard_role','')} | `{e.get('relative_path','')}` | {what[:240]} |")
            if str(e.get("what_it_does", "")).startswith("UNKNOWN"):
                unknowns.append(e)
    lines += ["", "## NEEDS OPERATOR LABEL", "", "| name | category | path |", "|---|---|---|"]
    for e in unknowns[:500]:
        lines.append(f"| {e.get('name','')} | {e.get('category','')} | `{e.get('relative_path','')}` |")
    return "\n".join(lines).rstrip()+"\n"


def build_tools_access(manifest: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    symlinks=[]; refs=[]; skipped=[]
    TOOLS_DIR.mkdir(exist_ok=True)
    for cat in CATEGORIES:
        d=TOOLS_DIR/cat
        d.mkdir(parents=True, exist_ok=True)
        real_roots=[]
        for c in PRIMARY_CANDIDATES.get(cat, []):
            p=Path(c); p=p if p.is_absolute() else ROOT/p
            if p.exists() and not is_sensitive_path(p):
                real_roots.append(p)
        lines=[f"# TOOLS/{cat}", "", "TICKLETRUNK access layer category.", "", "Original sandbox/toolbox artifacts remain sovereign. Copy/adapt into production; do not mutate originals unless explicitly instructed.", "", "## Real paths"]
        for p in real_roots:
            lines.append(f"- `{p}`")
        lines += ["", "## Use", "", "```bash", f"python3 scripts/tickletrunk_scan.py --category {cat}", "```", ""]
        (d/"README.md").write_text("\n".join(lines), encoding="utf-8")
        refs.append(str((d/"README.md").relative_to(ROOT)))
        for p in real_roots:
            # Symlink only repo-internal top-level-ish dirs/files. External hidden dirs stay references only.
            if not str(p).startswith(str(ROOT)) or p.name.startswith('.'):
                skipped.append(str(p))
                continue
            name = p.name if p.is_dir() else p.stem
            target=d/name
            if target.exists() or target.is_symlink():
                continue
            try:
                target.symlink_to(os.path.relpath(p, target.parent), target_is_directory=p.is_dir())
                symlinks.append(str(target.relative_to(ROOT)))
            except Exception:
                skipped.append(str(p))
    readme = f"""# TICKLETRUNK TOOLS ACCESS LAYER

TICKLETRUNK is the canonical manifest and access layer for the operator's proof hoard: every sovereign tool, algo, model, workflow, book, LoRA, scraper, skill, plugin, service, surface, schema, reusable fragment, and weird experimental instrument in the filesystem.

Machine manifest: `00_PROJECT_BRAIN/TICKLETRUNK.json`
Human manifest: `00_PROJECT_BRAIN/TICKLETRUNK.md`
Regenerate: `python3 scripts/tickletrunk_scan.py --execute`

Hard laws:
- Check TICKLETRUNK before writing new tools.
- Copy/adapt from the proof hoard when useful.
- Do not mutate sandbox originals without explicit operator instruction.
- Missing imports are not evidence of uselessness.
- Production gates apply to production copies, not sovereign proof-hoard originals.

This directory is for navigation. It uses symlinks when safe and README references otherwise.
"""
    (TOOLS_DIR/"README.md").write_text(readme, encoding="utf-8")
    refs.append(str((TOOLS_DIR/"README.md").relative_to(ROOT)))
    return symlinks, refs, skipped


def validate_manifest() -> tuple[bool, list[str]]:
    errors=[]
    if not MANIFEST_JSON.exists(): errors.append("missing TICKLETRUNK.json")
    if not MANIFEST_MD.exists(): errors.append("missing TICKLETRUNK.md")
    if not TOOLS_DIR.exists(): errors.append("missing TOOLS")
    if errors: return False, errors
    try:
        data=json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"json_parse_failed:{exc}"]
    for k in ["schema_version","generated_at","generator","repo_root","total_tools","toolboxes","guessed_name_map","skipped","warnings"]:
        if k not in data: errors.append(f"missing_top_level:{k}")
    for cat, entries in data.get("toolboxes", {}).items():
        for i,e in enumerate(entries):
            for f in ENTRY_FIELDS:
                if f not in e: errors.append(f"{cat}[{i}] missing {f}")
            if e.get("proof_hoard_role") not in PROOF_HOARD_ROLES:
                errors.append(f"{e.get('relative_path')} invalid proof_hoard_role")
            if is_sensitive_path(Path(str(e.get("path", "")))):
                errors.append(f"sensitive_path_indexed:{e.get('path')}")
    sl=ROOT/"00_PROJECT_BRAIN"/"STATUS_LEDGER.md"
    if sl.exists() and "TICKLETRUNK / TOOLBOX SOVEREIGNTY" not in sl.read_text(errors="ignore"):
        errors.append("status_ledger_missing_tickletrunk_hard_law")
    ag=ROOT/"AGENTS.md"
    if ag.exists() and "TICKLETRUNK" not in ag.read_text(errors="ignore"):
        errors.append("startup_reference_missing")
    elif not ag.exists():
        errors.append("AGENTS.md_missing")
    return not errors, errors


def scan(append: bool=False) -> tuple[dict[str, Any], dict[str, Any]]:
    roots, missing, warnings = discover_roots()
    skipped, raw = iter_tool_entries(roots)
    text_index = build_text_index()
    entries = [make_entry(item["path"], item["category"], text_index) for item in raw]
    entries = merge_existing(entries, append=append)
    manifest = build_manifest(entries, missing, skipped, warnings)
    categories = {cat: len(manifest["toolboxes"].get(cat, [])) for cat in CATEGORIES}
    report = {"total_entries": len(entries), "categories": categories, "missing_candidate_paths": missing, "skipped_paths": [str(p) for p in skipped], "warnings": warnings}
    return manifest, report


def write_run_report(mode: str, started: str, scan_report: dict[str, Any], files_written: list[str], validation: dict[str, Any]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"schema":"lucidota.tickletrunk.scan_report.v1","mode":mode,"started_at":started,"completed_at":utc_now(),**scan_report,"files_written":files_written,"validation_result":validation}
    out=OUT_DIR/f"tickletrunk_scan_{stamp()}.json"
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"REPORT_PATH={out.relative_to(ROOT)}")
    return out


def print_matches(entries: list[dict[str, Any]], limit: int=80) -> None:
    for e in entries[:limit]:
        print(f"{e['category']} | {e['kind']} | {e['relative_path']} | {e['one_line_description']}")


def main() -> int:
    ap=argparse.ArgumentParser(description="Scan/query TICKLETRUNK proof-hoard manifest")
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--append", action="store_true")
    mode.add_argument("--query")
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--category")
    mode.add_argument("--check", action="store_true")
    args=ap.parse_args()
    started=utc_now(); files_written=[]
    if args.check:
        ok, errs = validate_manifest()
        validation={"ok":ok,"errors":errs}
        write_run_report("check", started, {"total_entries":0,"categories":{},"missing_candidate_paths":[],"skipped_paths":[],"warnings":[]}, [], validation)
        print("CHECK_OK TICKLETRUNK valid" if ok else "CHECK_FAIL " + ";".join(errs))
        return 0 if ok else 1
    if args.query or args.list or args.category:
        existing_manifest = load_existing()
        if existing_manifest:
            manifest = existing_manifest
            scan_report = {
                "total_entries": int(existing_manifest.get("total_tools", 0)),
                "categories": {cat: len(existing_manifest.get("toolboxes", {}).get(cat, [])) for cat in CATEGORIES},
                "missing_candidate_paths": [],
                "skipped_paths": [],
                "warnings": ["query_used_existing_manifest_no_rescan"],
            }
        else:
            manifest, scan_report = scan(append=args.append)
        all_entries=[e for entries in manifest["toolboxes"].values() for e in entries]
        if args.query:
            q=args.query.lower()
            fields=("name","path","relative_path","category","kind","what_it_does","one_line_description","notes")
            matched=[e for e in all_entries if any(q in str(e.get(f,"")).lower() for f in fields) or any(q in str(t).lower() for t in e.get("tags", []))]
            print_matches(matched)
            mode_name=f"query:{args.query}"
        elif args.category:
            matched=manifest["toolboxes"].get(args.category.upper(), [])
            print_matches(matched, limit=300)
            mode_name=f"category:{args.category.upper()}"
        else:
            for cat in CATEGORIES:
                print(f"{cat}: {len(manifest['toolboxes'].get(cat, []))}")
            mode_name="list"
        write_run_report(mode_name, started, scan_report, [], {"ok":True,"errors":[]})
        return 0
    manifest, scan_report = scan(append=args.append)
    mode_name = "execute" if args.execute else "append" if args.append else "dry_run"
    if args.execute or args.append:
        MANIFEST_JSON.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
        MANIFEST_MD.write_text(render_md(manifest), encoding="utf-8")
        files_written += [str(MANIFEST_JSON.relative_to(ROOT)), str(MANIFEST_MD.relative_to(ROOT))]
        symlinks, refs, access_skips = build_tools_access(manifest)
        files_written += refs + symlinks
        scan_report["symlinks_created"] = symlinks
        scan_report["references_created"] = refs
        scan_report["access_layer_skipped"] = access_skips
    ok, errs = validate_manifest() if (args.execute or args.append) else (True, [])
    validation={"ok":ok,"errors":errs}
    write_run_report(mode_name, started, scan_report, files_written, validation)
    print(f"TOTAL_TOOLS={manifest['total_tools']}")
    print("CATEGORY_COUNTS=" + json.dumps(scan_report["categories"], sort_keys=True))
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
