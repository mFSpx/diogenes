#!/usr/bin/env python3
"""
GROQ REPO SCANNER — Sliding Window Async Cannon
================================================
Aims Groq's LPU inference at any repository for adversarial audit,
code review, or custom analysis. Zero-fucks ingester: reads everything
in binary mode, builds a spatial geometry map, blasts overlapping chunks
at Groq with asyncio concurrency, reconstructs findings to file:line.

Usage:
  python3 scripts/groq_repo_scanner.py --repo /path/to/any/repo
  python3 scripts/groq_repo_scanner.py --repo . --mode review --max-concurrent 40
  python3 scripts/groq_repo_scanner.py --repo /path --system-prompt my_prompt.txt
  python3 scripts/groq_repo_scanner.py --repo /path --reconstruct-only  (post-process existing DB)

Modes: security | review | algo | custom (requires --system-prompt)

Architecture:
  Phase 1: Ingest entire repo as one continuous string + spatial geometry map
  Phase 2: Sliding window chunks (configurable size + overlap)
  Phase 3: Async Groq blast with semaphore concurrency control
  Phase 4: Reconstruct findings back to original file:line
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_WINDOW = 12000   # chars per chunk (~3k tokens)
DEFAULT_STEP   = 8000    # step forward per chunk (4k overlap)
DEFAULT_CONCURRENT = 40

SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".lib", ".dll", ".exe",
    ".bin", ".pkl", ".pt", ".gguf", ".safetensors", ".onnx",
    ".jpg", ".jpeg", ".png", ".gif", ".ico", ".svg", ".woff",
    ".ttf", ".eot", ".mp4", ".mp3", ".wav", ".zip", ".tar",
    ".gz", ".bz2", ".7z", ".pdf", ".db", ".sqlite", ".sqlite3",
}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".env", "dist", "build", ".cache", ".mypy_cache",
}

SYSTEM_PROMPTS = {
    "security": """\
You are a hostile, state-level adversarial security auditor. You are analyzing a \
raw overlapping geometric chunk of a codebase. File boundaries may be mid-chunk. \
Ignore formatting. Your ONLY goal: identify critical vulnerabilities, logic flaws, \
race conditions, injection points, authentication bypasses, hardcoded secrets, \
insecure deserialization, and cryptographic weaknesses.

If you find NOTHING: output exactly the single word: CLEAN

If you find an issue, output ONLY this format (no prose, no markdown):
EXPLOIT: <severity:CRITICAL/HIGH/MEDIUM> <one-line description>
CONTEXT: <exact function name or variable or line excerpt>
VECTOR: <specific attack vector in one sentence>""",

    "review": """\
You are a brutal senior engineer doing a code review. You are reading a raw \
overlapping chunk of a codebase. File boundaries may be mid-chunk.

If the code is fine: output exactly the single word: CLEAN

If you find issues, output ONLY this format:
ISSUE: <severity:BLOCKING/MAJOR/MINOR> <one-line description>
CONTEXT: <exact code excerpt>
FIX: <specific fix in one sentence>""",

    "algo": """\
You are a computational mathematician scanning code for interesting algorithms, \
novel mathematical structures, and non-obvious patterns. You are reading a raw \
overlapping chunk of a codebase.

If nothing interesting: output exactly the single word: CLEAN

If you find something, output ONLY this format:
FIND: <type:NOVEL/INTERESTING/REUSE> <one-line description>
CONTEXT: <exact code excerpt>
NOTE: <why this is mathematically interesting>""",

    "bullshit": """\
You are a ruthless bullshit-detector auditing a self-built AI system. You read a raw \
overlapping chunk of the codebase (code/SQL/docs; file boundaries may be mid-chunk). \
Your ONLY target is THEATER: things that pretend to work or pretend to be done but \
are not. Hunt: stubbed/fake functions that return success without doing the work; \
hardcoded or mocked results dressed up as real; dead code that is never called; \
claims/comments/docstrings that contradict the actual code; receipts or status that \
assert "done"/"green" with no real effect; swallowed exceptions hiding failure \
(except: pass / except: return None); TODO/FIXME sitting in load-bearing paths; \
forked/duplicated logic that has drifted out of sync; config/paths that point at \
nothing.

If the chunk is honest and real: output exactly the single word: CLEAN

If you find bullshit, output ONLY this format (no prose, no markdown):
BS: <severity:ROT/STENCH/WHIFF> <one-line what the bullshit is>
CONTEXT: <exact function/variable/line excerpt that proves it>
WHY: <why it is theater, one sentence>""",
}


# ---------------------------------------------------------------------------
# Phase 1: Zero-fucks geometric ingester
# ---------------------------------------------------------------------------

def ingest_repo(repo_path: Path, verbose: bool = False) -> tuple[str, list[dict]]:
    """
    Walk repo, concatenate ALL files as text (binary decode, errors=ignore).
    Returns (raw_text, geometry_map).
    geometry_map: list of {file, char_start, char_end, line_start} dicts,
    sorted by char_start. Use for reconstructing file:line from char offset.
    """
    parts: list[str] = []
    geometry: list[dict] = []
    cursor = 0
    file_count = 0

    for root, dirs, files in os.walk(repo_path):
        # Prune skip dirs in-place
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        root_path = Path(root)

        for fname in sorted(files):
            fpath = root_path / fname
            if fpath.suffix.lower() in SKIP_EXTENSIONS:
                continue

            try:
                raw = fpath.read_bytes()
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                continue

            if not text.strip():
                continue

            # File separator so model sees filename mid-chunk
            rel = str(fpath.relative_to(repo_path))
            header = f"\n\n# === FILE: {rel} ===\n"
            block = header + text

            geometry.append({
                "file": rel,
                "char_start": cursor + len(header),
                "char_end": cursor + len(block),
                "line_start": 1,
                "line_count": text.count("\n") + 1,
            })

            parts.append(block)
            cursor += len(block)
            file_count += 1

    raw_text = "".join(parts)
    if verbose:
        print(f"Ingested {file_count} files → {len(raw_text):,} chars")
    return raw_text, geometry


def char_to_file_line(char_pos: int, geometry: list[dict]) -> tuple[str, int]:
    """Binary search geometry map to find file + approximate line for a char offset."""
    lo, hi = 0, len(geometry) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        g = geometry[mid]
        if char_pos < g["char_start"]:
            hi = mid - 1
        elif char_pos > g["char_end"]:
            lo = mid + 1
        else:
            # Approximate line within file
            offset_in_file = char_pos - g["char_start"]
            # Can't easily get exact line without re-reading, approximate
            approx_line = int(offset_in_file / max(g["char_end"] - g["char_start"], 1) * g["line_count"]) + 1
            return g["file"], approx_line
    return "unknown", 0


# ---------------------------------------------------------------------------
# Phase 2: Sliding window chunker
# ---------------------------------------------------------------------------

def sliding_window(text: str, window: int, step: int) -> list[tuple[int, int, str]]:
    """
    Yield (chunk_id, char_start, text_slice) tuples.
    window: chars per chunk. step: chars to advance (window-step = overlap).
    """
    chunks = []
    pos = 0
    cid = 0
    total = len(text)
    seen: set[int] = set()
    skipped = 0
    while pos < total:
        end = min(pos + window, total)
        slice_ = text[pos:end]
        key = hash(" ".join(slice_.split()))  # whitespace-normalized dedup ("bloom-lite")
        if key in seen:
            skipped += 1
            if end == total:
                break
            pos += step
            continue
        seen.add(key)
        chunks.append((cid, pos, slice_))
        cid += 1
        if end == total:
            break
        pos += step
    if skipped:
        print(f"         dedup skipped {skipped} duplicate chunks")
    return chunks


# ---------------------------------------------------------------------------
# Phase 3: Async Groq cannon
# ---------------------------------------------------------------------------

def load_key() -> str:
    k = os.environ.get("GROQ_API_KEY", "")
    if k:
        return k
    p = Path("/tmp/lucidota_groq_key")
    if p.exists():
        return p.read_text().strip()
    sys.exit("No Groq key. Set GROQ_API_KEY or write to /tmp/lucidota_groq_key")


def _groq_call_sync(key: str, model: str, system_prompt: str, chunk_text: str, timeout: int = 60) -> dict | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk_text},
        ],
        "temperature": 0.1,
        "max_tokens": 400,
    }
    req = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "groq-python/0.28.0",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            time.sleep(2)
            return {"_rate_limited": True}
        return None
    except Exception as ex:
        return {"_error": str(ex)}


def init_db(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.execute("""CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY,
        char_start INTEGER,
        processed INTEGER DEFAULT 0,
        response TEXT,
        finding TEXT,
        ts TEXT
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS findings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chunk_id INTEGER,
        char_start INTEGER,
        file TEXT,
        approx_line INTEGER,
        finding_type TEXT,
        raw TEXT,
        ts TEXT
    )""")
    con.commit()
    return con


def get_processed_ids(con: sqlite3.Connection) -> set[int]:
    return {r[0] for r in con.execute("SELECT chunk_id FROM chunks WHERE processed=1")}


async def blast_groq(
    chunks: list[tuple[int, int, str]],
    geometry: list[dict],
    key: str,
    model: str,
    system_prompt: str,
    db_path: Path,
    max_concurrent: int,
    resume_ids: set[int],
):
    semaphore = asyncio.Semaphore(max_concurrent)
    con = init_db(db_path)
    total = len(chunks)
    done = [0]
    findings_count = [0]

    async def process(chunk_id: int, char_start: int, text: str):
        if chunk_id in resume_ids:
            done[0] += 1
            return

        async with semaphore:
            result = await asyncio.to_thread(
                _groq_call_sync, key, model, system_prompt, text
            )

        ts = datetime.now(timezone.utc).isoformat()

        if result is None or result.get("_error"):
            con.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?)",
                (chunk_id, char_start, 0, json.dumps(result), None, ts),
            )
            con.commit()
            done[0] += 1
            return

        if result.get("_rate_limited"):
            # Re-queue by not marking as processed
            done[0] += 1
            return

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        is_clean = content.upper().startswith("CLEAN") or content.upper() == "CLEAN"
        finding = None if is_clean else content

        con.execute(
            "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?)",
            (chunk_id, char_start, 1, content[:2000], finding, ts),
        )

        if finding:
            file, line = char_to_file_line(char_start, geometry)
            con.execute(
                "INSERT INTO findings VALUES (NULL,?,?,?,?,?,?,?)",
                (chunk_id, char_start, file, line, "FINDING", finding[:2000], ts),
            )
            findings_count[0] += 1
            print(f"\r  [{done[0]+1}/{total}] FINDING in {file}:{line} → {finding[:60]}...")

        con.commit()
        done[0] += 1

        if done[0] % 50 == 0:
            pct = 100 * done[0] / total
            print(f"\r  {done[0]}/{total} ({pct:.1f}%) | findings: {findings_count[0]}   ", end="", flush=True)

    tasks = [process(cid, cs, txt) for cid, cs, txt in chunks]
    await asyncio.gather(*tasks)
    print(f"\n  Done. {done[0]} chunks, {findings_count[0]} findings.")
    con.close()


# ---------------------------------------------------------------------------
# Phase 4: Reconstruction report
# ---------------------------------------------------------------------------

def reconstruct(db_path: Path, geometry: list[dict], output_dir: Path):
    con = sqlite3.connect(db_path)
    findings = con.execute(
        "SELECT chunk_id, char_start, file, approx_line, raw FROM findings ORDER BY char_start"
    ).fetchall()
    con.close()

    if not findings:
        print("No findings. Everything is CLEAN.")
        return

    # Group by file
    by_file: dict[str, list] = {}
    for chunk_id, char_start, file, line, raw in findings:
        if not file or file == "unknown":
            file, line = char_to_file_line(char_start, geometry)
        by_file.setdefault(file, []).append((line, raw))

    report_path = output_dir / "findings_report.md"
    with report_path.open("w") as f:
        f.write(f"# Groq Scanner Findings\nGenerated: {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write(f"**Total findings:** {len(findings)} across {len(by_file)} files\n\n---\n\n")
        for file in sorted(by_file.keys()):
            f.write(f"## {file}\n\n")
            for line, raw in sorted(by_file[file]):
                f.write(f"**~Line {line}**\n```\n{raw}\n```\n\n")

    # Also write JSONL
    jsonl_path = output_dir / "findings.jsonl"
    with jsonl_path.open("w") as f:
        for chunk_id, char_start, file, line, raw in findings:
            f.write(json.dumps({
                "chunk_id": chunk_id, "file": file,
                "approx_line": line, "finding": raw,
            }) + "\n")

    print(f"\nFindings report: {report_path}")
    print(f"Findings JSONL:  {jsonl_path}")
    print(f"\nTop files by finding count:")
    for file, items in sorted(by_file.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  {len(items):3d}  {file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Groq Repo Scanner — Sliding Window Async Cannon")
    ap.add_argument("--repo", required=True, help="Path to repo to scan")
    ap.add_argument("--mode", default="security", choices=["security", "review", "algo", "bullshit", "custom"])
    ap.add_argument("--system-prompt", help="Path to custom system prompt file (mode=custom)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-concurrent", type=int, default=DEFAULT_CONCURRENT)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Chars per chunk")
    ap.add_argument("--step", type=int, default=DEFAULT_STEP, help="Step between chunks")
    ap.add_argument("--output-dir", default="05_OUTPUTS/groq_scanner")
    ap.add_argument("--extra-skip-dirs", default="", help="comma-separated extra directory names to skip")
    ap.add_argument("--resume", action="store_true", help="Resume from existing DB")
    ap.add_argument("--reconstruct-only", action="store_true", help="Only run reconstruction pass")
    ap.add_argument("--dry-run", action="store_true", help="Ingest and chunk only, no API calls")
    args = ap.parse_args()
    if args.extra_skip_dirs:
        SKIP_DIRS.update(d.strip() for d in args.extra_skip_dirs.split(",") if d.strip())

    repo = Path(args.repo).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / f"scan_{repo.name}.db"

    # Load system prompt
    if args.mode == "custom":
        if not args.system_prompt:
            sys.exit("--system-prompt required for mode=custom")
        system_prompt = Path(args.system_prompt).read_text()
    else:
        system_prompt = SYSTEM_PROMPTS[args.mode]

    print(f"\n=== GROQ REPO SCANNER ===")
    print(f"Repo     : {repo}")
    print(f"Mode     : {args.mode}")
    print(f"Model    : {args.model}")
    print(f"Window   : {args.window} chars | Step: {args.step} chars | Overlap: {args.window - args.step} chars")
    print(f"Concurrent: {args.max_concurrent}")
    print(f"DB       : {db_path}")
    print()

    # Phase 1: Ingest
    print("Phase 1: Ingesting repo...")
    raw_text, geometry = ingest_repo(repo, verbose=True)
    print(f"         {len(geometry)} files mapped")

    if args.reconstruct_only:
        print("\nPhase 4: Reconstruction only...")
        reconstruct(db_path, geometry, output_dir)
        return

    # Phase 2: Chunk
    print(f"\nPhase 2: Sliding window chunking...")
    chunks = sliding_window(raw_text, args.window, args.step)
    print(f"         {len(chunks)} chunks")

    if args.dry_run:
        print(f"\nDry run complete. Would send {len(chunks)} chunks to Groq.")
        print(f"Estimated tokens: ~{len(chunks) * args.window // 4:,}")
        return

    # Phase 3: Blast
    key = load_key()
    resume_ids: set[int] = set()
    if args.resume and db_path.exists():
        con = init_db(db_path)
        resume_ids = get_processed_ids(con)
        con.close()
        print(f"\nPhase 3: Resuming — {len(resume_ids)} already done, {len(chunks)-len(resume_ids)} remaining")
    else:
        print(f"\nPhase 3: Blasting {len(chunks)} chunks at Groq...")

    init_db(db_path)
    asyncio.run(blast_groq(
        chunks, geometry, key, args.model, system_prompt,
        db_path, args.max_concurrent, resume_ids,
    ))

    # Phase 4: Reconstruct
    print("\nPhase 4: Reconstructing findings...")
    reconstruct(db_path, geometry, output_dir)


if __name__ == "__main__":
    main()
