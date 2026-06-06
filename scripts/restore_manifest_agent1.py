#!/usr/bin/env python3
"""
Agent 1 — Manifest / File Enumeration for LUCIDOTA restore.
Enumerates every eligible file in the repo, computes metadata, writes manifest.
"""

import os
import sys
import json
import hashlib
import mimetypes
import subprocess
import time
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────
ROOT = Path("/home/mfspx/LUCIDOTA")
OUTPUT_DIR = ROOT / "05_OUTPUTS/restore/restore_ingest_20260606_093000Z"
MANIFEST_PATH = OUTPUT_DIR / "manifest_all_files.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "manifest_summary.json"
RECEIPT_PATH = OUTPUT_DIR / "agent1_manifest_receipt.json"
RUN_ID = "restore_ingest_20260606_093000Z"

# Exclusion directories (relative to ROOT or any sub-directory)
EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    "target",
    "dist",
    "build",
    ".venv",
    "venv",
    ".cache",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

# Full-path prefix exclusions
EXCLUDE_PATH_PREFIXES = [
    str(ROOT / "05_OUTPUTS"),
    str(ROOT / "09_STORAGE"),
    str(ROOT / "03_VAULT/models"),
    str(ROOT / "04_RUNTIME/models"),
    str(ROOT / "04_RUNTIME/needle_swarm"),
    str(ROOT / "04_RUNTIME/inference_os"),
]

# Known text extensions (for language guess and binary detection fallback)
TEXT_EXTENSIONS = {
    ".py": "Python", ".pyi": "Python", ".pyx": "Python",
    ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript",
    ".rs": "Rust",
    ".go": "Go",
    ".c": "C", ".h": "C",
    ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".hpp": "C++", ".hxx": "C++",
    ".java": "Java",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell",
    ".json": "JSON", ".jsonl": "JSON Lines",
    ".yaml": "YAML", ".yml": "YAML",
    ".toml": "TOML",
    ".md": "Markdown", ".mdx": "Markdown",
    ".txt": "Text", ".text": "Text",
    ".rst": "reStructuredText",
    ".cfg": "INI", ".ini": "INI",
    ".xml": "XML", ".html": "HTML", ".htm": "HTML",
    ".css": "CSS", ".scss": "SCSS", ".sass": "Sass", ".less": "Less",
    ".sql": "SQL",
    ".csv": "CSV", ".tsv": "TSV",
    ".env": "Shell", ".envrc": "Shell",
    ".dockerfile": "Dockerfile", ".dockerignore": "Ignore",
    ".gitignore": "Ignore", ".gitattributes": "Git Attributes", ".gitmodules": "Git Config",
    ".cmake": "CMake", ".cmake.in": "CMake",
    ".makefile": "Makefile", ".mk": "Makefile",
    ".proto": "Protobuf",
    ".svg": "SVG",
    ".lua": "Lua",
    ".rb": "Ruby",
    ".pl": "Perl", ".pm": "Perl",
    ".php": "PHP",
    ".r": "R", ".rmd": "R Markdown",
    ".swift": "Swift",
    ".kt": "Kotlin",
    ".scala": "Scala",
    ".el": "Emacs Lisp",
    ".clj": "Clojure", ".cljs": "ClojureScript",
    ".ex": "Elixir", ".exs": "Elixir",
    ".hs": "Haskell",
    ".ml": "OCaml", ".mli": "OCaml",
    ".vim": "Vim Script",
    ".ps1": "PowerShell",
    ".bat": "Batch", ".cmd": "Batch",
    ".patch": "Patch", ".diff": "Diff",
    ".lock": "Lockfile",
    ".nix": "Nix",
    ".tf": "Terraform",
    ".ncl": "Nickel",
    ".cue": "CUE",
    ".wgsl": "WGSL",
    ".glsl": "GLSL", ".vert": "GLSL", ".frag": "GLSL",
    ".ipynb": "Jupyter Notebook",
    ".puml": "PlantUML",
}

KNOWN_BINARY_EXTENSIONS = {
    ".bin", ".exe", ".dll", ".so", ".dylib", ".a", ".o", ".obj", ".lib",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tiff", ".tif",
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".opus",
    ".mp4", ".avi", ".mkv", ".mov", ".webm", ".wmv",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".zst",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".ttf", ".otf", ".woff", ".woff2",
    ".pyc", ".pyo", ".pyd",
    ".class", ".jar", ".war",
    ".wasm",
    ".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt", ".h5", ".onnx",
    ".pickle", ".pkl",
    ".db", ".sqlite", ".sqlite3",
    ".ipch", ".pch",
    ".DS_Store",
}

def is_path_excluded(abs_path_str):
    """Return True if the file should be excluded."""
    for prefix in EXCLUDE_PATH_PREFIXES:
        if abs_path_str.startswith(prefix):
            return True
    return False

def find_eligible_files():
    """Find all eligible files using find, filtering out excluded directories."""
    # Run a single find command that covers the whole tree
    exclude_args = []
    for d in EXCLUDE_DIRS:
        exclude_args.extend(["-name", d, "-prune", "-o"])
    # Also prune the explicit exclusion paths at the top level
    exclude_patterns = [
        "-path", f"{ROOT}/05_OUTPUTS", "-prune", "-o",
        "-path", f"{ROOT}/09_STORAGE", "-prune", "-o",
        "-path", f"{ROOT}/03_VAULT/models", "-prune", "-o",
        "-path", f"{ROOT}/04_RUNTIME/models", "-prune", "-o",
        "-path", f"{ROOT}/04_RUNTIME/needle_swarm", "-prune", "-o",
        "-path", f"{ROOT}/04_RUNTIME/inference_os", "-prune", "-o",
    ]

    cmd = ["find", str(ROOT)] + exclude_patterns + exclude_args + ["-type", "f", "-print0"]

    print(f"[agent1] Running find...", flush=True)
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"[agent1] find failed: {result.stderr.decode()}", flush=True)
        sys.exit(1)

    files = result.stdout.split(b'\x00')
    files = [f.decode('utf-8', errors='replace') for f in files if f]
    print(f"[agent1] Found {len(files)} raw file paths", flush=True)

    # Filter out any remaining excluded paths (belt and suspenders)
    eligible = []
    excluded_count = 0
    for f in files:
        if is_path_excluded(f):
            excluded_count += 1
            continue
        eligible.append(f)

    print(f"[agent1] Excluded {excluded_count} files by path prefix, {len(eligible)} eligible", flush=True)
    return eligible

def guess_mime_type(filepath_str, extension):
    """Guess MIME type using extension-based heuristics and python mimetypes."""
    # Try mimetypes first
    mime, _ = mimetypes.guess_type(filepath_str)
    if mime:
        return mime
    # Fallback heuristics for common unregistered types
    ext = extension.lower()
    mime_map = {
        ".rs": "text/x-rust",
        ".go": "text/x-go",
        ".py": "text/x-python",
        ".pyi": "text/x-python",
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".ts": "text/typescript",
        ".tsx": "text/typescript",
        ".sh": "text/x-shellscript",
        ".bash": "text/x-shellscript",
        ".toml": "application/toml",
        ".yaml": "text/yaml",
        ".yml": "text/yaml",
        ".md": "text/markdown",
        ".mdx": "text/markdown",
        ".jsonl": "application/jsonl",
        ".sql": "text/x-sql",
        ".proto": "text/x-protobuf",
        ".cmake": "text/x-cmake",
        ".dockerfile": "text/x-dockerfile",
        ".gitignore": "text/plain",
        ".gitattributes": "text/plain",
        ".env": "text/plain",
        ".lock": "application/json",
        ".nix": "text/x-nix",
        ".tf": "text/x-terraform",
        ".cue": "text/x-cue",
        ".wgsl": "text/x-wgsl",
        ".glsl": "text/x-glsl",
        ".puml": "text/x-plantuml",
        ".ipynb": "application/x-ipynb+json",
        ".patch": "text/x-diff",
        ".diff": "text/x-diff",
        ".csv": "text/csv",
        ".tsv": "text/tab-separated-values",
        ".rst": "text/x-rst",
        ".svg": "image/svg+xml",
        ".cfg": "text/plain",
        ".ini": "text/plain",
        ".lua": "text/x-lua",
        ".hs": "text/x-haskell",
        ".ml": "text/x-ocaml",
        ".mli": "text/x-ocaml",
        ".elm": "text/x-elm",
        ".kt": "text/x-kotlin",
        ".swift": "text/x-swift",
        ".scala": "text/x-scala",
        ".ex": "text/x-elixir",
        ".exs": "text/x-elixir",
        ".r": "text/x-r",
        ".rmd": "text/x-r-markdown",
        ".clj": "text/x-clojure",
        ".cljs": "text/x-clojure",
        ".pl": "text/x-perl",
        ".pm": "text/x-perl",
        ".ps1": "text/x-powershell",
        ".ncl": "text/x-nickel",
        ".vert": "text/x-glsl",
        ".frag": "text/x-glsl",
    }
    return mime_map.get(ext, "application/octet-stream")

def is_binary_by_content(filepath_str, extension):
    """Determine if a file is binary. Check extension first, then try reading."""
    ext = extension.lower()
    if ext in KNOWN_BINARY_EXTENSIONS:
        return True
    if ext in TEXT_EXTENSIONS:
        return False
    # Try to read first 8KB and check for null bytes
    try:
        with open(filepath_str, 'rb') as f:
            chunk = f.read(8192)
        return b'\x00' in chunk
    except (OSError, PermissionError):
        return True

def count_lines(filepath_str, is_binary):
    """Count lines in a text file. Returns None for binary files."""
    if is_binary:
        return None
    try:
        with open(filepath_str, 'r', encoding='utf-8', errors='replace') as f:
            return sum(1 for _ in f)
    except (OSError, PermissionError):
        return None

def guess_language(extension, filename, is_binary):
    """Guess programming language based on extension and filename."""
    if is_binary:
        return None
    ext = extension.lower()
    if ext in TEXT_EXTENSIONS:
        return TEXT_EXTENSIONS[ext]
    # Special filenames
    basename = filename.lower() if filename else ""
    if basename == "makefile" or basename == "gnumakefile":
        return "Makefile"
    if basename == "dockerfile":
        return "Dockerfile"
    if basename == "cmakelists.txt":
        return "CMake"
    if basename.startswith("."):
        return "Config"
    return None

def get_repo_root(filepath_str):
    """Determine which repo root a file belongs to."""
    # Walk up from the file to find the nearest git-managed directory
    path = Path(filepath_str)
    # Check if it's under 01_REPOS
    repos_base = ROOT / "01_REPOS"
    try:
        rel = path.relative_to(repos_base)
        parts = rel.parts
        if len(parts) >= 1:
            return parts[0]  # The repo name under 01_REPOS
    except ValueError:
        pass
    # Check odysseus at root level
    try:
        path.relative_to(ROOT / "odysseus")
        return "odysseus"
    except ValueError:
        pass
    # Everything else is LUCIDOTA
    return "LUCIDOTA"

def sha256_file(filepath_str):
    """Compute SHA-256 hash of a file."""
    try:
        h = hashlib.sha256()
        with open(filepath_str, 'rb') as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except (OSError, PermissionError) as e:
        return f"ERROR:{e}"

def stat_file(filepath_str):
    """Get size and mtime for a file."""
    try:
        s = os.stat(filepath_str)
        return s.st_size, s.st_mtime
    except (OSError, PermissionError) as e:
        return -1, 0

def build_git_tracked_sets():
    """For each git repo, build a set of tracked files (relative paths)."""
    git_repos = {}

    # Find all git repos
    to_check = [ROOT] + sorted([ROOT / "01_REPOS" / d for d in os.listdir(ROOT / "01_REPOS") if (ROOT / "01_REPOS" / d).is_dir()])

    # Also check odysseus
    odysseus_path = ROOT / "odysseus"
    if odysseus_path.is_dir():
        to_check.append(odysseus_path)

    for repo_path in to_check:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_path), "rev-parse", "--git-dir"],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                # Get all tracked files (cached/HEAD)
                tracked_result = subprocess.run(
                    ["git", "-C", str(repo_path), "ls-files", "--cached", "-z"],
                    capture_output=True, timeout=30
                )
                tracked = set()
                if tracked_result.returncode == 0:
                    for f in tracked_result.stdout.split(b'\x00'):
                        if f:
                            tracked.add(f.decode('utf-8', errors='replace'))

                # Get untracked non-ignored
                other_result = subprocess.run(
                    ["git", "-C", str(repo_path), "ls-files", "--others", "--exclude-standard", "-z"],
                    capture_output=True, timeout=30
                )
                untracked = set()
                if other_result.returncode == 0:
                    for f in other_result.stdout.split(b'\x00'):
                        if f:
                            untracked.add(f.decode('utf-8', errors='replace'))

                repo_name = repo_path.name
                # For root LUCIDOTA, map it specially
                if repo_path == ROOT:
                    repo_name = "LUCIDOTA"

                git_repos[str(repo_path)] = {
                    "name": repo_name,
                    "tracked": tracked,
                    "untracked": untracked,
                }
                print(f"[agent1] Git repo: {repo_name} -> {len(tracked)} tracked, {len(untracked)} untracked", flush=True)
        except (subprocess.TimeoutExpired, OSError) as e:
            print(f"[agent1] Git check failed for {repo_path}: {e}", flush=True)

    return git_repos

def check_git_tracked(filepath_str, git_repos):
    """Determine if a file is git-tracked. Returns True/False/None."""
    abs_path = str(filepath_str)

    # Find the matching git repo
    best_match = None
    best_len = 0
    for repo_path, info in git_repos.items():
        if abs_path.startswith(repo_path + "/") or abs_path == repo_path:
            if len(repo_path) > best_len:
                best_len = len(repo_path)
                best_match = (repo_path, info)

    if best_match is None:
        return None

    repo_path, info = best_match
    rel_path = abs_path[len(repo_path):].lstrip("/")

    if rel_path in info["tracked"]:
        return True
    elif rel_path in info["untracked"]:
        return False
    else:
        # Could be ignored - treat as None (can't determine)
        return None

def main():
    start_time = time.time()
    print(f"[agent1] Starting manifest enumeration at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start_time))}", flush=True)

    # Step 1: Find all eligible files
    files = find_eligible_files()
    total_files = len(files)

    # Step 2: Build git tracked sets for all repos
    print("[agent1] Building git tracked sets...", flush=True)
    git_repos = build_git_tracked_sets()

    # Step 3: Process each file
    print(f"[agent1] Processing {total_files} files...", flush=True)

    summary = {
        "schema": "lucidota.restore.manifest_summary.v1",
        "run_id": RUN_ID,
        "total_files": 0,
        "by_extension": {},
        "by_repo_root": {},
        "total_size_bytes": 0,
        "binary_count": 0,
        "text_count": 0,
        "git_tracked_count": 0,
        "git_untracked_count": 0,
        "git_unknown_count": 0,
    }

    bytes_written = 0
    file_counter = 0

    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        for i, filepath_str in enumerate(files):
            if i > 0 and i % 5000 == 0:
                elapsed = time.time() - start_time
                print(f"[agent1] Progress: {i}/{total_files} files ({i*100//total_files}%), {elapsed:.1f}s elapsed", flush=True)

            path = Path(filepath_str)
            filename = path.name
            extension = path.suffix.lower()

            # Compute file-level metadata
            size_bytes, mtime = stat_file(filepath_str)
            sha256 = sha256_file(filepath_str)
            mime_type = guess_mime_type(filepath_str, extension)
            binary = is_binary_by_content(filepath_str, extension)
            line_count = count_lines(filepath_str, binary)
            language = guess_language(extension, filename, binary)
            repo_root = get_repo_root(filepath_str)
            git_tracked = check_git_tracked(filepath_str, git_repos)

            # Relative path from LUCIDOTA root
            try:
                rel_path = str(path.relative_to(ROOT))
            except ValueError:
                rel_path = filepath_str

            # Mtime ISO format
            if mtime > 0:
                mtime_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(mtime))
            else:
                mtime_iso = None

            record = {
                "absolute_path": filepath_str,
                "relative_path": rel_path,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "mtime_iso": mtime_iso,
                "extension": extension if extension else None,
                "mime_guess": mime_type,
                "is_binary": binary,
                "line_count": line_count,
                "language_guess": language,
                "git_tracked": git_tracked,
                "repo_root": repo_root,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            bytes_written += size_bytes if size_bytes > 0 else 0
            file_counter += 1

            # Aggregate summary
            ext_key = extension.lstrip(".") if extension else "no_extension"
            if ext_key in TEXT_EXTENSIONS or ext_key in {"no_extension", "Makefile", "Dockerfile"}:
                pass  # keep as-is
            elif ext_key in KNOWN_BINARY_EXTENSIONS:
                pass  # keep as-is

            summary["by_extension"][ext_key] = summary["by_extension"].get(ext_key, 0) + 1
            summary["by_repo_root"][repo_root] = summary["by_repo_root"].get(repo_root, 0) + 1
            if binary:
                summary["binary_count"] += 1
            else:
                summary["text_count"] += 1
            if git_tracked is True:
                summary["git_tracked_count"] += 1
            elif git_tracked is False:
                summary["git_untracked_count"] += 1
            else:
                summary["git_unknown_count"] += 1

    summary["total_files"] = file_counter
    summary["total_size_bytes"] = bytes_written

    # Sort extension and repo counts for cleaner output
    summary["by_extension"] = dict(sorted(summary["by_extension"].items(), key=lambda x: -x[1]))
    summary["by_repo_root"] = dict(sorted(summary["by_repo_root"].items(), key=lambda x: -x[1]))

    # Write summary
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time

    # Write receipt
    receipt = {
        "agent": "agent1",
        "task": "manifest_file_enumeration",
        "run_id": RUN_ID,
        "status": "complete",
        "outputs": {
            "manifest": str(MANIFEST_PATH),
            "summary": str(SUMMARY_PATH),
        },
        "stats": {
            "total_files": file_counter,
            "total_size_bytes": bytes_written,
            "total_size_human": f"{bytes_written / (1024**3):.2f} GB",
            "binary_count": summary["binary_count"],
            "text_count": summary["text_count"],
            "git_tracked_count": summary["git_tracked_count"],
            "git_untracked_count": summary["git_untracked_count"],
            "repos_enumerated": len(summary["by_repo_root"]),
            "elapsed_seconds": round(elapsed, 2),
        },
        "completed_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    with open(RECEIPT_PATH, 'w', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)

    print(f"\n[agent1] COMPLETE in {elapsed:.1f}s", flush=True)
    print(f"[agent1] Total files: {file_counter}", flush=True)
    print(f"[agent1] Total size: {bytes_written / (1024**3):.2f} GB", flush=True)
    print(f"[agent1] Binary: {summary['binary_count']}, Text: {summary['text_count']}", flush=True)
    print(f"[agent1] Git tracked: {summary['git_tracked_count']}, Untracked: {summary['git_untracked_count']}, Unknown: {summary['git_unknown_count']}", flush=True)
    print(f"[agent1] Repos: {list(summary['by_repo_root'].keys())}", flush=True)
    print(f"[agent1] Top extensions: {list(summary['by_extension'].items())[:15]}", flush=True)
    print(f"[agent1] Manifest: {MANIFEST_PATH}", flush=True)
    print(f"[agent1] Summary: {SUMMARY_PATH}", flush=True)
    print(f"[agent1] Receipt: {RECEIPT_PATH}", flush=True)

if __name__ == "__main__":
    main()
