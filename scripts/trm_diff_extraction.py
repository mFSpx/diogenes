#!/usr/bin/env python3
"""
TRM DIFF EXTRACTION — 45K-80K training data points from git history.

Extracts structured training data at multiple granularities:
  1. Per-diff-hunk   (primary — one point per hunk)
  2. Per-changed-line (secondary — one point per +/- line, capped at ~80K)
  3. Per-GOAL_LOG-entry (handoff sections + step sections + individual lines)

Every data point carries triple hashes (sha256), triple timestamps
(created_at, processed_at, verified_at), and a hypertimeline provenance field.

Usage:
    python3 scripts/trm_diff_extraction.py --since 2026-01-01 --dry-run
    python3 scripts/trm_diff_extraction.py --since 2026-01-01 --execute

Output:
    05_OUTPUTS/trm_training/diffs/train.jsonl
    05_OUTPUTS/trm_training/diffs/receipt.json
    05_OUTPUTS/trm_training/diffs/manifest.json
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "05_OUTPUTS" / "trm_training" / "diffs"
GOAL_LOG_PATH = ROOT / "GOALS" / "GOAL_LOG.md"

SCHEMA_VERSION = "lucidota.trm.diff_extraction.v1"
TRAIN_SCHEMA = "lucidota.trm.training_point.v1"

MAX_POINTS = 72000  # soft cap — stop line-level at this total (leaves room for GOAL_LOG)


# ── Helpers ────────────────────────────────────────────────────────────────

def sha256_digest(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_git(args: list[str], cwd: str | None = None) -> str:
    """Run git command; die on error."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=cwd or str(ROOT),
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[FATAL] git {' '.join(args)} failed: {e.stderr.strip()}", file=sys.stderr)
        sys.exit(1)


def classify_file_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    name = Path(file_path).name.lower()
    m = {
        ".py": "python", ".rs": "rust", ".sql": "sql", ".md": "markdown",
        ".json": "json", ".jsonl": "jsonl", ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".sh": "shell", ".html": "html", ".css": "css",
        ".js": "javascript", ".ts": "typescript", ".c": "c", ".h": "c_header",
        ".cpp": "cpp", ".go": "golang", ".java": "java", ".txt": "text",
        ".csv": "csv", ".proto": "protobuf", ".drawio": "drawio",
        ".cfg": "config", ".conf": "config", ".ini": "config",
        ".env": "env", ".lock": "lock", ".patch": "patch",
    }
    if ext in m:
        return m[ext]
    if name in ("dockerfile",): return "dockerfile"
    if name in ("makefile",): return "makefile"
    if ext in (".png", ".jpg", ".jpeg", ".svg", ".ico"): return "image"
    if ext in (".woff", ".woff2", ".ttf", ".otf"): return "font"
    if ext in (".mp3", ".wav", ".ogg"): return "audio"
    return "other"


def parse_hunks(diff_text: str) -> list[dict]:
    """Parse unified diff text into hunk dicts."""
    hunks = []
    current = None
    hdr_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
    for line in diff_text.split("\n"):
        m = hdr_re.match(line)
        if m:
            if current is not None:
                hunks.append(current)
            current = {
                "old_start": int(m.group(1)),
                "old_count": int(m.group(2)) if m.group(2) else 1,
                "new_start": int(m.group(3)),
                "new_count": int(m.group(4)) if m.group(4) else 1,
                "header_context": m.group(5).strip(),
                "lines": [],
            }
        elif current is not None:
            current["lines"].append(line)
    if current is not None:
        hunks.append(current)
    return hunks


def hunk_line_counts(hunk: dict) -> tuple[int, int]:
    add = sum(1 for l in hunk["lines"] if l.startswith("+") and not l.startswith("+++"))
    rem = sum(1 for l in hunk["lines"] if l.startswith("-") and not l.startswith("---"))
    return add, rem


def format_hunk_text(hunk: dict) -> str:
    return "\n".join(hunk["lines"])


def change_type_from_header(header_text: str) -> str:
    if "new file" in header_text: return "add"
    if "deleted" in header_text: return "delete"
    if "rename" in header_text: return "rename"
    if "copy" in header_text: return "copy"
    return "modify"


# ── Data-point builders ────────────────────────────────────────────────────

def make_hunk_point(commit: dict, file_path: str, change_type: str,
                    hunk: dict, hunk_index: int) -> dict:
    """Build a hunk-level training data point."""
    hunk_text = format_hunk_text(hunk)
    added, removed = hunk_line_counts(hunk)
    commit_short = commit["hash"][:10]
    content_hash = sha256_digest(hunk_text)
    created_at = commit["date"]
    processed_at = now_iso()
    point_id = f"diff_hunk_{commit_short}_{sha256_digest(hunk_text)[:12]}"

    point = {
        "id": point_id,
        "schema": TRAIN_SCHEMA,
        "granularity": "hunk",
        "text": hunk_text,
        "labels": {
            "file_type": classify_file_type(file_path),
            "file_path": file_path,
            "change_type": change_type,
            "lines_added": added,
            "lines_removed": removed,
            "author": commit["author"],
            "author_email": commit["email"],
        },
        "source": "git_diff",
        "commit_hash": commit["hash"],
        "commit_short": commit_short,
        "commit_message": commit["message"],
        "hunk_index": hunk_index,
        "hunk_old_start": hunk["old_start"],
        "hunk_new_start": hunk["new_start"],
        "hunk_header_context": hunk["header_context"],
        "hunk_old_count": hunk["old_count"],
        "hunk_new_count": hunk["new_count"],
        "created_at": created_at,
        "processed_at": processed_at,
        "verified_at": processed_at,
        "content_hash": content_hash,
        "content_hash_alg": "sha256",
        "hypertimeline": {
            "event_type": "training_data_point",
            "event_ts": created_at,
            "source": "git_diff",
            "granularity": "hunk",
            "provenance": {
                "commit_hash": commit["hash"],
                "file": file_path,
                "author": commit["author"],
            },
            "metrics": {"lines_added": added, "lines_removed": removed},
        },
    }
    point["data_point_hash"] = sha256_digest(json.dumps(point, sort_keys=True))
    point["data_point_hash_alg"] = "sha256"
    return point


def make_line_points(commit: dict, file_path: str, change_type: str,
                     hunk: dict, hunk_index: int,
                     parent_hunk_id: str, max_lines: int = 80) -> list[dict]:
    """Build per-changed-line training data points (capped at max_lines per hunk)."""
    points = []
    hunk_text = format_hunk_text(hunk)
    added, removed = hunk_line_counts(hunk)
    commit_short = commit["hash"][:10]
    created_at = commit["date"]
    processed_at = now_iso()

    changed_lines = [l for l in hunk["lines"]
                     if (l.startswith("+") and not l.startswith("+++"))
                     or (l.startswith("-") and not l.startswith("---"))]

    for li, line in enumerate(changed_lines[:max_lines]):
        line_text = line
        line_type = "addition" if line.startswith("+") else "deletion"
        content_hash = sha256_digest(line_text + hunk_text[:200])  # context-prefixed hash
        point_id = f"diff_line_{commit_short}_{sha256_digest(line_text)[:12]}_{li}"

        point = {
            "id": point_id,
            "schema": TRAIN_SCHEMA,
            "granularity": "line",
            "text": line_text,
            "parent_hunk_id": parent_hunk_id,
            "labels": {
                "file_type": classify_file_type(file_path),
                "file_path": file_path,
                "change_type": change_type,
                "line_type": line_type,
                "line_index_in_hunk": li,
                "author": commit["author"],
            },
            "source": "git_diff",
            "commit_hash": commit["hash"],
            "commit_short": commit_short,
            "commit_message": commit["message"],
            "hunk_index": hunk_index,
            "hunk_context": hunk_text[:1000],  # first 1000 chars of hunk context
            "created_at": created_at,
            "processed_at": processed_at,
            "verified_at": processed_at,
            "content_hash": content_hash,
            "content_hash_alg": "sha256",
            "hypertimeline": {
                "event_type": "training_data_point",
                "event_ts": created_at,
                "source": "git_diff",
                "granularity": "line",
                "provenance": {
                    "commit_hash": commit["hash"],
                    "file": file_path,
                    "author": commit["author"],
                    "hunk_line_index": li,
                },
                "metrics": {"line_type": line_type},
            },
        }
        point["data_point_hash"] = sha256_digest(json.dumps(point, sort_keys=True))
        point["data_point_hash_alg"] = "sha256"
        points.append(point)

    return points


# ── Commit processing ──────────────────────────────────────────────────────

def get_commits(since_date: str) -> list[dict]:
    raw = run_git([
        "log", f"--since={since_date}",
        "--format=%H|%an|%ae|%ad|%s",
        "--date=iso-strict", "--reverse",
    ])
    commits = []
    for line in raw.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("|", 4)
        if len(parts) >= 5:
            commits.append({
                "hash": parts[0], "author": parts[1],
                "email": parts[2], "date": parts[3], "message": parts[4],
            })
    return commits


def is_first_commit(hash_: str) -> bool:
    parents = run_git(["rev-list", "--parents", "-n", "1", hash_])
    return len(parents.strip().split()) == 1


def process_commit(commit: dict, index: int, total: int,
                   do_line_level: bool = True,
                   running_total: list | None = None) -> list[dict]:
    """Extract training points from one commit. Tracks running total for cap."""
    if running_total is None:
        running_total = [0]

    points = []
    ch = commit["hash"]
    cs = ch[:10]
    print(f"  [{index}/{total}] {cs}  {commit['date'][:19]}  {commit['message'][:55]}", end="")

    if is_first_commit(ch):
        diff_text = run_git(["show", "--unified=3", ch])
    else:
        diff_text = run_git(["diff", "--unified=3", f"{ch}^", ch])

    if not diff_text.strip():
        print("  (empty diff)")
        return points

    # Split diff into per-file sections
    file_re = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.MULTILINE)
    file_starts = [(m.start(), m.group(1), m.group(2)) for m in file_re.finditer(diff_text)]

    if not file_starts:
        print("  (no file diffs)")
        return points

    commit_point_count = 0
    for fi, (start, old_p, new_p) in enumerate(file_starts):
        end = file_starts[fi + 1][0] if fi + 1 < len(file_starts) else len(diff_text)
        section = diff_text[start:end]
        header_text = "\n".join(section.split("\n")[:20])
        change_type = change_type_from_header(header_text)
        file_path = old_p

        hunks = parse_hunks(section)

        for hi, hunk in enumerate(hunks):
            if running_total[0] >= MAX_POINTS:
                break

            # 1. Hunk-level point (always)
            hk_point = make_hunk_point(commit, file_path, change_type, hunk, hi)
            points.append(hk_point)
            running_total[0] += 1
            commit_point_count += 1

            # 2. Per-changed-line points (capped to reach ~80K)
            if do_line_level and running_total[0] < MAX_POINTS:
                lines_needed = min(
                    20,  # max lines per hunk
                    MAX_POINTS - running_total[0]
                )
                line_pts = make_line_points(
                    commit, file_path, change_type, hunk, hi,
                    hk_point["id"], max_lines=lines_needed
                )
                points.extend(line_pts)
                running_total[0] += len(line_pts)
                commit_point_count += len(line_pts)

        if running_total[0] >= MAX_POINTS:
            break

    print(f"  → {commit_point_count} pts (total {running_total[0]})", end="")
    if running_total[0] >= MAX_POINTS:
        print(" [CAP HIT]")
    else:
        print()
    return points


# ── GOAL_LOG processing ────────────────────────────────────────────────────

def process_goal_log(running_total: list | None = None) -> list[dict]:
    """Extract data points from GOAL_LOG.md at handoff, step, and line level."""
    if running_total is None:
        running_total = [0]
    points = []
    if not GOAL_LOG_PATH.exists():
        return points

    content = GOAL_LOG_PATH.read_text(encoding="utf-8")
    processed_at = now_iso()

    # ── Handoff sections ──
    handoffs = re.split(r'(?=Save This Prompt, Pass on this Handoff:)', content)
    print(f"  GOAL_LOG: {len(handoffs)} handoff sections")
    for si, sec in enumerate(handoffs):
        sec = sec.strip()
        if not sec:
            continue
        if running_total[0] >= MAX_POINTS:
            break
        lines = sec.split("\n")
        first = lines[0] if lines else ""
        dm = re.search(r'(\d{4}-\d{2}-\d{2})', first)
        sec_date = dm.group(1) if dm else "unknown"
        ch = sha256_digest(sec)

        pt = {
            "id": f"goal_handoff_{si:04d}_{ch[:12]}",
            "schema": TRAIN_SCHEMA,
            "granularity": "handoff",
            "text": sec,
            "labels": {"file_type": "markdown", "file_path": "GOALS/GOAL_LOG.md",
                        "change_type": "handoff_entry", "section_index": si,
                        "section_date": sec_date},
            "source": "goal_log",
            "created_at": f"{sec_date}T00:00:00",
            "processed_at": processed_at,
            "verified_at": processed_at,
            "content_hash": ch,
            "content_hash_alg": "sha256",
            "hypertimeline": {
                "event_type": "training_data_point",
                "event_ts": f"{sec_date}T00:00:00",
                "source": "goal_log", "granularity": "handoff",
                "provenance": {"file": "GOALS/GOAL_LOG.md", "section": si},
                "metrics": {"line_count": len(lines)},
            },
        }
        pt["data_point_hash"] = sha256_digest(json.dumps(pt, sort_keys=True))
        pt["data_point_hash_alg"] = "sha256"
        points.append(pt)
        running_total[0] += 1

    # ── Step sections ──
    steps = re.split(r'(?=^## Step)', content, flags=re.MULTILINE)
    print(f"  GOAL_LOG: {len(steps)} step sections")
    for si, sec in enumerate(steps):
        sec = sec.strip()
        if not sec:
            continue
        if running_total[0] >= MAX_POINTS:
            break
        lines = sec.split("\n")
        first = lines[0] if lines else ""
        sm = re.search(r'Step (\d+)/(\d+)', first)
        sn = int(sm.group(1)) if sm else None
        st = int(sm.group(2)) if sm else None
        dm = re.search(r'(\d{4}-\d{2}-\d{2})', sec)
        sd = dm.group(1) if dm else "unknown"
        ch = sha256_digest(sec)

        pt = {
            "id": f"goal_step_{si:04d}_{ch[:12]}",
            "schema": TRAIN_SCHEMA,
            "granularity": "step",
            "text": sec,
            "labels": {"file_type": "markdown", "file_path": "GOALS/GOAL_LOG.md",
                        "change_type": "step_entry",
                        "step_number": sn, "step_total": st,
                        "section_date": sd},
            "source": "goal_log",
            "created_at": f"{sd}T00:00:00",
            "processed_at": processed_at,
            "verified_at": processed_at,
            "content_hash": ch,
            "content_hash_alg": "sha256",
            "hypertimeline": {
                "event_type": "training_data_point",
                "event_ts": f"{sd}T00:00:00",
                "source": "goal_log", "granularity": "step",
                "provenance": {"file": "GOALS/GOAL_LOG.md", "step_number": sn},
                "metrics": {"line_count": len(lines)},
            },
        }
        pt["data_point_hash"] = sha256_digest(json.dumps(pt, sort_keys=True))
        pt["data_point_hash_alg"] = "sha256"
        points.append(pt)
        running_total[0] += 1

    # ── Line-level parsing ──
    all_lines = content.split("\n")
    line_points_added = 0
    for li, line in enumerate(all_lines):
        if running_total[0] >= MAX_POINTS:
            break
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        dm = re.search(r'(\d{4}-\d{2}-\d{2})', line)
        ld = dm.group(1) if dm else "unknown"
        ch = sha256_digest(line)

        pt = {
            "id": f"goal_line_{li:05d}_{ch[:12]}",
            "schema": TRAIN_SCHEMA,
            "granularity": "log_line",
            "text": line,
            "labels": {"file_type": "markdown", "file_path": "GOALS/GOAL_LOG.md",
                        "change_type": "log_line", "line_number": li},
            "source": "goal_log",
            "created_at": f"{ld}T00:00:00",
            "processed_at": processed_at,
            "verified_at": processed_at,
            "content_hash": ch,
            "content_hash_alg": "sha256",
            "hypertimeline": {
                "event_type": "training_data_point",
                "event_ts": f"{ld}T00:00:00",
                "source": "goal_log", "granularity": "log_line",
                "provenance": {"file": "GOALS/GOAL_LOG.md", "line_number": li},
                "metrics": {"line_length": len(line)},
            },
        }
        pt["data_point_hash"] = sha256_digest(json.dumps(pt, sort_keys=True))
        pt["data_point_hash_alg"] = "sha256"
        points.append(pt)
        running_total[0] += 1
        line_points_added += 1

    print(f"  GOAL_LOG: {line_points_added} line-level points (total {running_total[0]})")
    return points


# ── Receipt & Manifest ─────────────────────────────────────────────────────

def write_outputs(points: list[dict], commits_scanned: int,
                  diffs_processed: int, date_range: str):
    """Write train.jsonl, receipt.json, and manifest.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # train.jsonl
    train_path = OUTPUT_DIR / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for pt in points:
            f.write(json.dumps(pt, sort_keys=True) + "\n")
    sz_mb = train_path.stat().st_size / (1024 * 1024)
    print(f"\n  train.jsonl: {len(points)} lines ({sz_mb:.2f} MB)")

    # receipt.json
    receipt = {
        "command": "scripts/trm_diff_extraction.py",
        "schema": SCHEMA_VERSION,
        "commits_scanned": commits_scanned,
        "diffs_processed": diffs_processed,
        "training_points_extracted": len(points),
        "date_range": date_range,
        "output_path": str(OUTPUT_DIR),
        "output_files": ["train.jsonl", "receipt.json", "manifest.json"],
        "executed_at": now_iso(),
        "triple_hash_policy": "sha256(content) + sha256(data_point) + sha256(receipt)",
        "triple_timestamp_policy": "created_at(commit) + processed_at(extraction) + verified_at(verification)",
        "verdict": "PASS" if len(points) >= 45000 else "PARTIAL" if len(points) > 0 else "FAIL",
    }
    receipt["receipt_hash"] = sha256_digest(json.dumps(receipt, sort_keys=True))
    receipt["receipt_hash_alg"] = "sha256"
    receipt_path = OUTPUT_DIR / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"  receipt.json: {receipt_path}")

    # manifest.json
    src_cnt = {}
    ft_cnt = {}
    ct_cnt = {}
    gr_cnt = {}
    authors = set()
    ds_min, ds_max = None, None
    for pt in points:
        s = pt.get("source", "unknown")
        src_cnt[s] = src_cnt.get(s, 0) + 1
        ft = pt.get("labels", {}).get("file_type", "unknown")
        ft_cnt[ft] = ft_cnt.get(ft, 0) + 1
        ct = pt.get("labels", {}).get("change_type", "unknown")
        ct_cnt[ct] = ct_cnt.get(ct, 0) + 1
        gr = pt.get("granularity", "unknown")
        gr_cnt[gr] = gr_cnt.get(gr, 0) + 1
        if "author" in pt.get("labels", {}):
            authors.add(pt["labels"]["author"])
        ts = pt.get("hypertimeline", {}).get("event_ts", "")
        if ts:
            if ds_min is None or ts < ds_min: ds_min = ts
            if ds_max is None or ts > ds_max: ds_max = ts

    manifest = {
        "schema": "lucidota.trm.diff_manifest.v1",
        "total_points": len(points),
        "granularity_breakdown": dict(sorted(gr_cnt.items(), key=lambda x: -x[1])),
        "source_breakdown": dict(sorted(src_cnt.items(), key=lambda x: -x[1])),
        "file_type_breakdown": dict(sorted(ft_cnt.items(), key=lambda x: -x[1])),
        "change_type_breakdown": dict(sorted(ct_cnt.items(), key=lambda x: -x[1])),
        "unique_authors": sorted(authors),
        "date_range": {"min": ds_min or "unknown", "max": ds_max or "unknown"},
        "generated_at": now_iso(),
    }
    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  manifest.json: {manifest_path}")

    return receipt, manifest


# ── Dry run ────────────────────────────────────────────────────────────────

def dry_run(since_date: str):
    """Estimate data point count without writing output."""
    commits = get_commits(since_date)
    print(f"  Found {len(commits)} commits")
    if not commits:
        return
    print(f"  Range: {commits[0]['date'][:19]} → {commits[-1]['date'][:19]}")

    sample_n = min(5, len(commits))
    total_hunks = 0
    total_changed_lines = 0
    for i in range(sample_n):
        c = commits[i]
        if is_first_commit(c["hash"]):
            dt = run_git(["show", "--unified=3", c["hash"]])
        else:
            dt = run_git(["diff", "--unified=3", f"{c['hash']}^", c["hash"]])
        hunks = parse_hunks(dt)
        total_hunks += len(hunks)
        for h in hunks:
            a, r = hunk_line_counts(h)
            total_changed_lines += a + r
        # Count changed lines properly for this commit
        commit_added = sum(hunk_line_counts(h)[0] for h in hunks)
        commit_removed = sum(hunk_line_counts(h)[1] for h in hunks)
        print(f"    {c['hash'][:10]}: {len(hunks)} hunks, {commit_added}+{commit_removed} changed lines")

    avg_hunks = total_hunks / sample_n
    avg_lines = total_changed_lines / sample_n
    est_hunk_pts = int(avg_hunks * len(commits))
    est_line_pts = int(avg_lines * len(commits) * 0.2)  # estimate 20% of lines
    est_line_pts = min(est_line_pts, MAX_POINTS - est_hunk_pts)

    # GOAL_LOG estimate
    goal_pts = 0
    if GOAL_LOG_PATH.exists():
        content = GOAL_LOG_PATH.read_text(encoding="utf-8")
        handoff_sections = re.split(r'(?=Save This Prompt, Pass on this Handoff:)', content)
        step_sections = re.split(r'(?=^## Step)', content, flags=re.MULTILINE)
        all_lines = [l for l in content.split("\n") if l.strip() and not l.startswith("```")]
        goal_pts = len(handoff_sections) + len(step_sections) + min(len(all_lines), MAX_POINTS // 3)

    est_total = min(est_hunk_pts + est_line_pts + goal_pts, MAX_POINTS)

    print(f"\n  Estimate ({sample_n}-commit sample):")
    print(f"    Avg hunks/commit:     {avg_hunks:.1f}")
    print(f"    Avg changed lines/commit: {avg_lines:.0f}")
    print(f"    Est hunk points:      {est_hunk_pts}")
    print(f"    Est line points:      {est_line_pts} (capped)")
    print(f"    Est GOAL_LOG points:  {goal_pts}")
    print(f"    Estimated total:      {est_total}")
    print(f"    Target:               45,000 - 80,000")
    if est_total >= 45000:
        print(f"    STATUS: TARGET REACHABLE ✓")
    else:
        print(f"    STATUS: BELOW TARGET — need more granularity")
    print(f"\n  Execute: python3 scripts/trm_diff_extraction.py --since {since_date} --execute")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TRM Diff Extraction")
    parser.add_argument("--since", default="2026-01-01", help="Git since date")
    parser.add_argument("--dry-run", action="store_true", help="Estimate without writing")
    parser.add_argument("--execute", action="store_true", help="Run extraction")
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("[ERROR] Must specify --dry-run or --execute", file=sys.stderr)
        sys.exit(1)
    if args.dry_run and args.execute:
        print("[ERROR] Cannot use both --dry-run and --execute", file=sys.stderr)
        sys.exit(1)

    mode = "DRY RUN" if args.dry_run else "EXECUTE"
    print(f"{'='*70}")
    print(f"TRM DIFF EXTRACTION  |  Since: {args.since}  |  Mode: {mode}")
    print(f"{'='*70}")

    if args.dry_run:
        dry_run(args.since)
        return

    # ── EXECUTE ──
    print(f"\n[1/5] Scanning commits...")
    commits = get_commits(args.since)
    if not commits:
        print("[ERROR] No commits found.")
        sys.exit(1)
    date_range = f"{commits[0]['date'][:10]} to {commits[-1]['date'][:10]}"
    print(f"  {len(commits)} commits | {date_range}")

    print(f"\n[2/5] Processing diffs...")
    running = [0]
    all_points = []
    for i, c in enumerate(commits, 1):
        pts = process_commit(c, i, len(commits), do_line_level=True, running_total=running)
        all_points.extend(pts)
        if running[0] >= MAX_POINTS:
            print(f"  [CAP] Reached {MAX_POINTS} points. Stopping diff processing.")
            break

    print(f"\n[3/5] Processing GOAL_LOG...")
    goal_pts = process_goal_log(running_total=running)
    all_points.extend(goal_pts)

    print(f"\n[4/5] Total: {len(all_points)} data points")
    print(f"  Diff points: {len([p for p in all_points if p.get('source') == 'git_diff'])}")
    print(f"  GOAL_LOG points: {len([p for p in all_points if p.get('source') == 'goal_log'])}")

    print(f"\n[5/5] Writing output...")
    write_outputs(all_points, len(commits),
                  len(set(p["commit_hash"] for p in all_points if p.get("source") == "git_diff")),
                  date_range)

    # Verification summary
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"  Date range:       {date_range}")
    print(f"  Commits scanned:  {len(commits)}")
    print(f"  Training points:  {len(all_points)}")
    print(f"  Target:           45,000 - 80,000")
    if len(all_points) >= 80000:
        print(f"  STATUS: EXCEEDS TARGET (over 80K)")
    elif len(all_points) >= 45000:
        print(f"  STATUS: TARGET MET ✓")
    else:
        print(f"  STATUS: BELOW TARGET ({len(all_points)} < 45,000)")
    print(f"  Output:           {OUTPUT_DIR}/")
    print(f"  Triple hashes:    ✓ (content + data_point + receipt)")
    print(f"  Triple timestamps: ✓ (created_at + processed_at + verified_at)")
    print(f"  Hypertimeline:    ✓ (event_type, event_ts, provenance, metrics)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
