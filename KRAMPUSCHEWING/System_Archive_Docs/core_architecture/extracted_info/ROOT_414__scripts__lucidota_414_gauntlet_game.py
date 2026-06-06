#!/usr/bin/env python3
"""ROOT-414 Gauntlet Game.

Turns benchmark cases or book pages into judgeable rounds. Supports .txt/.md/.pdf
page extraction and .mobi best-effort extraction through calibre's ebook-convert
when available, falling back to strings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "00_PROJECT_BRAIN" / "414_PRIMITIVE_CRIES" / "benchmarks"
CASES = BENCH / "cases"
SUBS = BENCH / "submissions"
JUDGMENTS = BENCH / "judgments"
REPORTS = BENCH / "reports"
BOOKS = ROOT / "BOOKS"
EXTRACTED_BOOKS = BENCH / "books"
PAGES = BENCH / "pages"

CANONICAL_BPS = [0, 2, 4, 6, 10, 50, 69, 150]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_").lower()[:90] or "book"


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def extract_pdf_page(path: Path, page: int) -> str:
    if not shutil.which("pdftotext"):
        raise SystemExit("pdftotext not installed; cannot extract PDF page.")
    cp = run(["pdftotext", "-f", str(page), "-l", str(page), str(path), "-"])
    if cp.returncode != 0:
        raise SystemExit(cp.stderr.strip() or "pdftotext failed")
    return cp.stdout.strip()


def split_text_pages(text: str, chars_per_page: int = 2500) -> list[str]:
    chunks: list[str] = []
    paras = re.split(r"\n\s*\n", text)
    cur = ""
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if cur and len(cur) + len(para) + 2 > chars_per_page:
            chunks.append(cur.strip())
            cur = para
        else:
            cur = (cur + "\n\n" + para).strip() if cur else para
    if cur:
        chunks.append(cur.strip())
    return chunks or [text[:chars_per_page]]


def strip_html(s: str) -> str:
    s = re.sub(r"<script.*?</script>", " ", s, flags=re.I | re.S)
    s = re.sub(r"<style.*?</style>", " ", s, flags=re.I | re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"\s+", " ", s).strip()


def extract_mobi_text(path: Path, work_dir: Path) -> tuple[str, str]:
    """Return (text, method)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("ebook-convert"):
        out_txt = work_dir / (path.stem + ".txt")
        cp = run(["ebook-convert", str(path), str(out_txt)])
        if cp.returncode == 0 and out_txt.exists() and out_txt.stat().st_size > 0:
            return out_txt.read_text(encoding="utf-8", errors="ignore"), "ebook-convert"
    # Fallback: crude, but still lets the game stage pages for ugly OCR-like judgment.
    cp = run(["strings", "-n", "5", str(path)])
    if cp.returncode == 0 and cp.stdout.strip():
        return cp.stdout, "strings-fallback"
    data = path.read_bytes()
    return data.decode("utf-8", errors="ignore"), "utf8-fallback"


def extract_book_page(path: Path, page: int) -> dict[str, Any]:
    ext = path.suffix.lower()
    book_id = slug(path.stem)
    page_dir = PAGES / book_id
    page_dir.mkdir(parents=True, exist_ok=True)
    if ext == ".pdf":
        text = extract_pdf_page(path, page)
        method = "pdftotext-page"
    elif ext in {".txt", ".md"}:
        pages = split_text_pages(path.read_text(encoding="utf-8", errors="ignore"))
        if page < 1 or page > len(pages):
            raise SystemExit(f"page {page} out of range 1..{len(pages)}")
        text = pages[page - 1]
        method = "text-chunk"
    elif ext in {".mobi", ".azw", ".azw3", ".epub"}:
        txt, method = extract_mobi_text(path, EXTRACTED_BOOKS / book_id)
        pages = split_text_pages(strip_html(txt) if "<html" in txt.lower() else txt)
        if page < 1 or page > len(pages):
            raise SystemExit(f"page {page} out of range 1..{len(pages)} using {method}")
        text = pages[page - 1]
    else:
        raise SystemExit(f"unsupported extension: {ext}")

    page_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()
    case = {
        "benchmark_id": f"book_{book_id}_p{page:04d}",
        "title": f"{path.name} — page {page}",
        "input_text": text,
        "book": {
            "path": str(path),
            "book_id": book_id,
            "source_sha256": sha256(path),
            "page_number": page,
            "page_hash": page_hash,
            "extract_method": method,
            "do_not_infer_beyond_page": True,
        },
        "expected": {
            "required_primitives": ["DOCUMENT_EXAMINATION"],
            "forbidden_primitives": ["THE_SPIRAL_IS_COMPLETE", "ARCHONIC_CONTROL_GRID"],
            "allowed_bps": [6, 10, 50],
            "must_have_falsifier": True,
            "must_have_local_gate": True,
            "claim_lifecycle": ["CLAIM_UNVERIFIED", "CORROBORATION_REQUIRED", "DESIGN_PATTERN_EXTRACTED"],
        },
        "judge_focus": [
            "Did the parser avoid reading beyond this page?",
            "Did it separate text says / route suggests / speculation?",
            "Did it avoid high-label noise on first contact?",
            "Did it create useful carry-forward threads without spoilers?",
        ],
    }
    out = CASES / f"{case['benchmark_id']}.json"
    out.write_text(json.dumps(case, indent=2, sort_keys=True), encoding="utf-8")
    return {"case_path": str(out), "benchmark_id": case["benchmark_id"], "chars": len(text), "extract_method": method, "page_hash": page_hash}


def list_cases() -> dict[str, Any]:
    return {"cases": sorted(p.stem for p in CASES.glob("*.json"))}


def judge(case_id: str, submission_id: str, decision: str, score: int, notes: str, repair: str = "") -> dict[str, Any]:
    JUDGMENTS.mkdir(parents=True, exist_ok=True)
    obj = {
        "benchmark_id": case_id,
        "submission_id": submission_id,
        "decision": decision,
        "score": score,
        "notes": notes,
        "repair_instruction": repair,
        "judge": "Northern.Strike",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out = JUDGMENTS / f"{case_id}__{submission_id}__judgment.json"
    out.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return {"judgment_path": str(out), **obj}


def show_case(case_id: str, width: int = 100) -> str:
    path = CASES / f"{case_id}.json"
    case = json.loads(path.read_text(encoding="utf-8"))
    text = case.get("input_text", "")
    return f"CASE: {case_id}\nTITLE: {case.get('title','')}\n\n" + textwrap.fill(text, width=width) + "\n"



def library() -> dict[str, Any]:
    BOOKS.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(BOOKS.glob("*")):
        if p.is_file() and p.suffix.lower() in {".pdf", ".mobi", ".azw", ".azw3", ".epub", ".txt", ".md"}:
            rows.append({"id": slug(p.stem), "path": str(p), "name": p.name, "size_bytes": p.stat().st_size, "ext": p.suffix.lower()})
    return {"book_dir": str(BOOKS), "count": len(rows), "books": rows}


def resolve_book(book_ref: str) -> Path:
    p = Path(book_ref).expanduser()
    if p.exists():
        return p.resolve()
    lib = library()["books"]
    matches = [b for b in lib if b["id"] == book_ref or book_ref.lower() in b["name"].lower() or book_ref.lower() in b["id"]]
    if not matches:
        raise SystemExit(f"book not found in {BOOKS}: {book_ref}")
    if len(matches) > 1:
        msg = "multiple books match:\n" + "\n".join(f"- {m['id']} :: {m['name']}" for m in matches)
        raise SystemExit(msg)
    return Path(matches[0]["path"])

def main() -> int:
    ap = argparse.ArgumentParser(description="ROOT-414 Gauntlet Game")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("library")
    sub.add_parser("list")
    p_extract = sub.add_parser("extract-page")
    p_extract.add_argument("book")
    p_extract.add_argument("page", type=int)
    p_show = sub.add_parser("show")
    p_show.add_argument("case_id")
    p_judge = sub.add_parser("judge")
    p_judge.add_argument("case_id")
    p_judge.add_argument("submission_id")
    p_judge.add_argument("decision", choices=["approved", "needs_repair", "rejected", "comment"])
    p_judge.add_argument("score", type=int)
    p_judge.add_argument("notes")
    p_judge.add_argument("--repair", default="")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.cmd == "library":
        out = library()
    elif args.cmd == "list":
        out = list_cases()
    elif args.cmd == "extract-page":
        out = extract_book_page(resolve_book(args.book), args.page)
    elif args.cmd == "show":
        txt = show_case(args.case_id)
        if not args.json:
            print(txt)
            return 0
        out = {"case_id": args.case_id, "text": txt}
    elif args.cmd == "judge":
        out = judge(args.case_id, args.submission_id, args.decision, args.score, args.notes, args.repair)
    else:
        raise SystemExit("unknown command")

    print(json.dumps(out, indent=2, sort_keys=True) if args.json else out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
