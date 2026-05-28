#!/usr/bin/env python3
"""INDY_READs — GO-25 page-by-page reading game.

INDY_READs is a she: reading companion, margin-noter, judgment collector.

Dynamic library: /home/mfspx/LUCIDOTA/BOOKS
State/data:      /home/mfspx/LUCIDOTA/BOOKS/.indy_reads

No page rewind. One page at a time. Fast heuristic v0.50 parser notes.
"""
from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ROOT / "BOOKS"
DATA = BOOKS / ".indy_reads"
PAGES = DATA / "pages"
CACHE = DATA / "parser_cache"
STATE_PATH = DATA / "state.json"
CSV_PATH = DATA / "indy_reads_judgments.csv"
SCHEMA_PATH = BOOKS / "GO_GAME_GRADING_SCHEMA.json"
ONTOLOGY_PATH = BOOKS / "GO_ACTIVE_TERMS.json"
PARSER_VERSION = "go_fast_indy_reads_v0.1"
PERSONA_ID = "INDY_READs"
PERSONA_DISPLAY = "INDY_READs"
PERSONA_PRONOUNS = "she/her"
MAIN_AI_PERSONA = True
PERSONA_CONFIG_PATH = ROOT / "04_RUNTIME" / "indy_reads_persona_config.json"
ADAPTER_REGISTRY_PATH = ROOT / "04_RUNTIME" / "indy_reads_adapter_registry.json"
CHARS_PER_PAGE = 2200
SUPPORTED = {".pdf", ".epub", ".mobi", ".azw", ".azw3", ".txt", ".md"}
CANONICAL_BPS = [0, 2, 4, 6, 10, 50, 69, 150]


DEFAULT_PERSONA_CONFIG: dict[str, Any] = {
    "schema": "lucidota.indy_reads.persona_config.v1",
    "persona_id": PERSONA_ID,
    "display_name": PERSONA_DISPLAY,
    "pronouns": PERSONA_PRONOUNS,
    "main_ai_persona": MAIN_AI_PERSONA,
    "active_ontology": {
        "name": "GO",
        "expanded_name": "Global Ontology",
        "terms_path": "BOOKS/GO_ACTIVE_TERMS.json",
    },
    "mission": "Page-locked reading companion, margin-noter, and judgment collector for the GO reading game.",
    "permissions": {
        "read_paths": ["BOOKS", "BOOKS/.indy_reads", "04_RUNTIME/indy_reads_adapter_registry.json"],
        "write_paths": ["BOOKS/.indy_reads", "04_RUNTIME/indy_reads_persona_config.json", "04_RUNTIME/indy_reads_adapter_registry.json"],
        "may_update_adapter_registry": True,
        "may_edit_active_go_terms": False,
        "may_touch_graph_core_sql": False,
        "may_create_doctrine_markdown": False,
    },
    "memory_boundaries": {
        "page_locked_reading": True,
        "no_forward_book_claims": True,
        "persistent_memory_paths": ["BOOKS/.indy_reads/state.json", "BOOKS/.indy_reads/indy_reads_judgments.csv"],
        "cache_paths": ["BOOKS/.indy_reads/pages", "BOOKS/.indy_reads/parser_cache"],
        "external_truth_default": "unverified_until_evidence",
    },
}

DEFAULT_ADAPTER_REGISTRY: dict[str, Any] = {
    "schema": "lucidota.indy_reads.adapter_registry.v1",
    "registry_id": "indy_reads_lora_adapter_candidates",
    "owner_persona": PERSONA_ID,
    "active_ontology": "GO / Global Ontology",
    "write_policy": "append_or_update_candidates_only; no graph-core SQL writes",
    "default_base_model": "deepseek-1.5b-indy_reads-reads",
    "candidates": [
        {
            "adapter_id": "indy_reads_go_margin_v0",
            "kind": "lora",
            "target_model_id": "deepseek-1.5b-indy_reads-reads",
            "status": "planned",
            "training_sources": ["BOOKS/.indy_reads/indy_reads_judgments.csv"],
            "permission_scope": "private_local_only",
            "memory_boundary": "page_locked_go_margin_notes",
            "notes": "Candidate adapter for INDY_READs GO margin-note style; not trained yet.",
        },
        {
            "adapter_id": "indy_reads_go_router_v0",
            "kind": "prompt_or_lora",
            "target_model_id": "deepseek-1.5b-indy_reads-reads",
            "status": "watch",
            "training_sources": ["BOOKS/GO_ACTIVE_TERMS.json"],
            "permission_scope": "terms_read_only",
            "memory_boundary": "term-routing only; no doctrine expansion",
            "notes": "Lightweight candidate for GO term routing and adapter browsing.",
        },
    ],
}


def write_json_if_missing(path: Path, data: dict[str, Any]) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json_or_default(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_persona_config() -> dict[str, Any]:
    return load_json_or_default(PERSONA_CONFIG_PATH, DEFAULT_PERSONA_CONFIG)


def load_adapter_registry() -> dict[str, Any]:
    return load_json_or_default(ADAPTER_REGISTRY_PATH, DEFAULT_ADAPTER_REGISTRY)


def load_go_terms() -> list[dict[str, str]]:
    try:
        data = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
        return data.get("terms", [])
    except (OSError, json.JSONDecodeError, TypeError):
        return []

GO_TERMS = load_go_terms()
GO_BY_TERM = {t["term"]: t for t in GO_TERMS}
GO_BY_ID = {t["id"]: t for t in GO_TERMS}

CORE_TERMS = [t["term"] for t in GO_TERMS]
MYTHIC_TERMS = {"NAUGHTY", "NICE"}


def ensure_dirs() -> None:
    for p in [BOOKS, DATA, PAGES, CACHE, ROOT / "04_RUNTIME"]:
        p.mkdir(parents=True, exist_ok=True)
    write_json_if_missing(PERSONA_CONFIG_PATH, DEFAULT_PERSONA_CONFIG)
    write_json_if_missing(ADAPTER_REGISTRY_PATH, DEFAULT_ADAPTER_REGISTRY)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()


def clear() -> None:
    os.system("clear" if os.name == "posix" else "cls")


def pause(msg: str = "ENTER continues...") -> None:
    input(f"\n{msg}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_").lower()[:96] or "book"


@dataclass
class Book:
    id: str
    name: str
    path: str
    ext: str
    size_bytes: int


def library() -> list[Book]:
    ensure_dirs()
    rows = []
    for p in sorted(BOOKS.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            rows.append(Book(slug(p.stem), p.name, str(p), p.suffix.lower(), p.stat().st_size))
    return rows


def load_state() -> dict[str, Any]:
    ensure_dirs()
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"books": {}, "active_book_id": ""}


def save_state(st: dict[str, Any]) -> None:
    ensure_dirs()
    STATE_PATH.write_text(json.dumps(st, indent=2, sort_keys=True), encoding="utf-8")


def get_book_state(st: dict[str, Any], b: Book) -> dict[str, Any]:
    bs = st.setdefault("books", {}).setdefault(b.id, {})
    bs.setdefault("current_page", 1)
    bs.setdefault("completed_pages", [])
    bs.setdefault("source_sha256", sha_file(Path(b.path)))
    bs.setdefault("last_opened", now())
    bs.setdefault("name", b.name)
    bs.setdefault("path", b.path)
    return bs


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def strip_html(raw: str) -> str:
    raw = re.sub(r"<script.*?</script>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<style.*?</style>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"</p>|<br\s*/?>|</h\d>|</div>", "\n\n", raw, flags=re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    raw = re.sub(r"\n\s+", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def epub_text(path: Path) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
        # Sort path-wise; good enough for game reading. Future: parse OPF spine.
        for name in sorted(names):
            try:
                raw = z.read(name).decode("utf-8", errors="ignore")
            except (KeyError, RuntimeError, UnicodeError, zipfile.BadZipFile):
                continue
            txt = strip_html(raw)
            if txt:
                parts.append(txt)
    return "\n\n".join(parts)


def whole_text_for_book(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext == ".epub":
        return epub_text(path), "epub-zip-html"
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore"), "text"
    if ext in {".mobi", ".azw", ".azw3"}:
        if shutil.which("ebook-convert"):
            out = CACHE / (slug(path.stem) + ".txt")
            cp = run(["ebook-convert", str(path), str(out)])
            if cp.returncode == 0 and out.exists() and out.stat().st_size > 0:
                return out.read_text(encoding="utf-8", errors="ignore"), "ebook-convert"
        cp = run(["strings", "-n", "5", str(path)])
        return cp.stdout, "strings-fallback"
    raise ValueError(f"whole_text_for_book unsupported for {ext}")


def split_pages(text: str, chars: int = CHARS_PER_PAGE) -> list[str]:
    paras = re.split(r"\n\s*\n", text)
    pages: list[str] = []
    cur = ""
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if cur and len(cur) + len(para) + 2 > chars:
            pages.append(cur.strip())
            cur = para
        else:
            cur = (cur + "\n\n" + para).strip() if cur else para
    if cur:
        pages.append(cur.strip())
    return pages or [text[:chars]]


def extract_page(book: Book, page: int) -> dict[str, Any]:
    path = Path(book.path)
    book_dir = PAGES / book.id
    book_dir.mkdir(parents=True, exist_ok=True)
    page_file = book_dir / f"p{page:04d}.json"
    if page_file.exists():
        return json.loads(page_file.read_text(encoding="utf-8"))

    if book.ext == ".pdf":
        if not shutil.which("pdftotext"):
            raise RuntimeError("pdftotext missing")
        cp = run(["pdftotext", "-f", str(page), "-l", str(page), book.path, "-"])
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr.strip() or "pdftotext failed")
        text = cp.stdout.strip()
        method = "pdftotext-page"
    else:
        txt, method = whole_text_for_book(path)
        pages = split_pages(txt)
        if page < 1 or page > len(pages):
            raise RuntimeError(f"page {page} out of range 1..{len(pages)} by {method}")
        text = pages[page - 1]

    obj = {
        "book_id": book.id,
        "book_name": book.name,
        "book_path": book.path,
        "page": page,
        "text": text,
        "page_hash": sha_text(text),
        "source_sha256": sha_file(path),
        "extract_method": method,
        "chars": len(text),
        "created_at": now(),
        "do_not_infer_beyond_page": True,
    }
    page_file.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return obj


def sentenceish(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    bits = re.split(r"(?<=[.!?])\s+", cleaned)
    return [b.strip() for b in bits if len(b.strip()) > 20]


def fast_parse(page: dict[str, Any]) -> dict[str, Any]:
    cache = CACHE / f"{page['book_id']}_p{int(page['page']):04d}_{page['page_hash'][:12]}.json"
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        cached.setdefault("persona_id", PERSONA_ID)
        cached.setdefault("main_ai_persona", MAIN_AI_PERSONA)
        return cached
    text = page["text"]
    low = text.lower()
    sents = sentenceish(text)
    local_gates = ["EVIDENCE", "CLAIM"]
    terms = ["EVIDENCE", "CLAIM", "TERM"]
    if any(k in low for k in ["source", "according to", "reported", "archive", "document", "book"]):
        terms.append("SOURCE")
    if any(k in low for k in ["witness", "saw", "observed", "testified"]):
        terms.append("WITNESS")
    if any(k in low for k in ["rumour", "rumor", "alleged", "apparently", "they say"]):
        terms.append("RUMOUR")
    if any(k in low for k in ["threat", "risk", "danger", "coerc", "harm", "exploit"]):
        terms.append("THREAT")
    if any(k in low for k in ["license", "licence", "permit", "certif"]):
        terms.append("LICENSE")
    if any(k in low for k in ["regulator", "government", "ministry", "agency", "bureau"]):
        terms.extend(["REGULATOR", "GOVERNMENT"])
    if any(k in low for k in ["law", "rule", "statute", "regulation", "policy"]):
        terms.extend(["LAW", "RULE"])
    if any(k in low for k in ["where", "street", "avenue", "road", "city", "glasgow", "malta"]):
        terms.append("LOCATION")
    if any(k in low for k in ["said", "asked", "replied", "told"]):
        terms.append("SIGNAL")
    if any(k in low for k in ["because", "therefore", "so that", "result", "caused"]):
        terms.append("RELATIONSHIP")
        local_gates.append("RELATIONSHIP")
    if re.search(r"\b\d{1,2}:\d{2}\b|monday|tuesday|wednesday|thursday|friday|saturday|sunday|september|january|february|march", low):
        terms.append("TIME")
        local_gates.append("TIME")
    if any(k in low for k in ["dream", "like", "as if", "metaphor", "song", "game", "story"]):
        terms.extend(["PATTERN", "GLOW"])
    if page.get("extract_method") == "strings-fallback" or len(re.findall(r"\b[a-zA-Z]{1,2}\b", text)) > 80:
        terms.extend(["SIGNAL", "COMMENT"])
        local_gates.append("SIGNAL")
    # preserve order unique
    seen = set(); terms = [p for p in terms if not (p in seen or seen.add(p))]
    seen = set(); local_gates = [p for p in local_gates if not (p in seen or seen.add(p))]

    notes = []
    notes.append("PAGE_LOCK: interpreting this page/chunk only; no forward-book claims.")
    if page.get("extract_method") == "strings-fallback":
        notes.append("MOBI_STRINGS_HARD_MODE: extraction is noisy; treat as noise-resistance round, not clean prose custody.")
    if sents:
        notes.append("TEXT_SAYS: " + sents[0][:260])
    if len(sents) > 1:
        notes.append("CARRY_FORWARD_THREAD: " + sents[1][:220])
    notes.append("GO_ROUTE: " + " ∩ ".join(terms[:5]) + " → PAGE_LEVEL_READING_PACKET")

    parser = {
        "parser_version": PARSER_VERSION,
        "persona_id": PERSONA_ID,
        "main_ai_persona": MAIN_AI_PERSONA,
        "packet_id": f"indy::{page['book_id']}::p{int(page['page']):04d}",
        "raw_text_anchor": sents[0][:300] if sents else text[:300],
        "local_gates": local_gates,
        "terms": terms,
        "route": {
            "anchor": terms[0],
            "operator": "∩",
            "vector": terms[1:5],
            "resolution": "PAGE_LEVEL_READING_PACKET",
        },
        "ternary_state": {"text_presence": 1, "internal_scope": 1, "external_truth": 0},
        "claim_lifecycle": "CLAIM_UNVERIFIED",
        "confidence_bps": 10 if page.get("extract_method") == "strings-fallback" else 50,
        "falsifier": "Later page or cleaner source extraction contradicts this page-level interpretation.",
        "notes": notes,
        "mythic_terms_available_but_not_forced": sorted(MYTHIC_TERMS),
        "created_at": now(),
    }
    cache.write_text(json.dumps(parser, indent=2, sort_keys=True), encoding="utf-8")
    return parser


CSV_FIELDS = [
    "timestamp", "book_id", "book_name", "page", "page_hash", "extract_method", "parser_version",
    "packet_id", "parser_terms", "parser_bps", "decision", "score", "score_label",
    "term_correction", "notes", "repair_instruction", "favorite_line", "confusion", "raw_csv_json",
]


def score_label(score: int) -> str:
    if score >= 95: return "CAKE"
    if score >= 80: return "COOKED"
    if score >= 60: return "NEEDS_REPAIR"
    if score >= 30: return "SLOP_DETECTED"
    return "ARCHON_BAIT"


def append_csv(row: dict[str, Any]) -> None:
    ensure_dirs()
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def wrap_print(text: str, width: int = 92, max_lines: int | None = None) -> None:
    lines: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=width, replace_whitespace=True))
    if max_lines is None:
        max_lines = len(lines)
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print(f"\n…[{len(lines)-max_lines} more lines hidden]")


def banner(subtitle: str) -> None:
    clear()
    print("╔" + "═" * 92 + "╗")
    print("║" + "INDY_READs — main AI persona (she/her)".center(92) + "║")
    print("║" + subtitle.center(92) + "║")
    print("╚" + "═" * 92 + "╝")


def pick_book(st: dict[str, Any]) -> Book | None:
    books = library()
    while True:
        banner("BOOKS — dynamic /LUCIDOTA/BOOKS library")
        if not books:
            print(f"No books in {BOOKS}")
            pause(); return None
        for i, b in enumerate(books, 1):
            bs = st.get("books", {}).get(b.id, {})
            cur = bs.get("current_page", 1)
            mark = " ← active" if st.get("active_book_id") == b.id else ""
            print(f"{i:>2}. {b.name} [{b.ext}] — page {cur}{mark}")
        print("\np. GO term browser   a. adapter candidates   q. quit")
        ans = input("\nPick book: ").strip().lower()
        if ans == "q": return None
        if ans == "p":
            term_browser(); continue
        if ans == "a":
            adapter_browser(); continue
        try:
            n = int(ans)
            if 1 <= n <= len(books):
                b = books[n-1]
                st["active_book_id"] = b.id
                get_book_state(st, b)["last_opened"] = now()
                save_state(st)
                return b
        except ValueError:
            print("Invalid numeric choice.")


def term_browser() -> None:
    while True:
        banner("GO TERM BROWSER — use @number or #TERM")
        print("Examples: @01, @13, @37, #EVIDENCE, #ANOMALY, search words like law")
        q = input("lookup> ").strip()
        if q.lower() in {"q", "quit", "back", ""}:
            return
        results = []
        if q.startswith("@") and q[1:].isdigit():
            key = "@" + q[1:].zfill(2)
            if key in GO_BY_ID:
                t = GO_BY_ID[key]
                results = [t]
        elif q.startswith("#"):
            target = q[1:].upper()
            results = [t for t in GO_TERMS if target in t["term"]]
        else:
            target = q.upper()
            results = [t for t in GO_TERMS if target in t["term"] or target in t.get("definition", "").upper()]
        if not results:
            print("No hit.")
        else:
            for t in results[:40]:
                print(f"{t['id']} #{t['term']} — {t.get('definition','')}")
        pause()



def adapter_browser() -> None:
    cfg = load_persona_config()
    reg = load_adapter_registry()
    banner("ADAPTER CANDIDATES — INDY_READs browse/update seed")
    print(f"Persona: {cfg.get('display_name', PERSONA_DISPLAY)} ({cfg.get('pronouns', PERSONA_PRONOUNS)})")
    print(f"Main AI persona: {cfg.get('main_ai_persona', MAIN_AI_PERSONA)}")
    ontology = cfg.get("active_ontology", {})
    print(f"Ontology: {ontology.get('name', 'GO')} — {ontology.get('expanded_name', 'Global Ontology')}")
    print(f"Registry: {ADAPTER_REGISTRY_PATH}")
    print(f"Policy: {reg.get('write_policy', '')}\n")
    for c in reg.get("candidates", []):
        print(f"- {c.get('adapter_id')} [{c.get('kind')}/{c.get('status')}]")
        print(f"  target: {c.get('target_model_id', reg.get('default_base_model', ''))}")
        print(f"  scope:  {c.get('permission_scope', '')}")
        print(f"  memory: {c.get('memory_boundary', '')}")
        if c.get("notes"):
            print(f"  notes:  {c.get('notes')}")
    pause()

def standard_flow(book: Book, st: dict[str, Any]) -> None:
    bs = get_book_state(st, book)
    while True:
        page_n = int(bs.get("current_page", 1))
        try:
            page = extract_page(book, page_n)
            parser = fast_parse(page)
            # preload next page in background-ish foreground quick cache, ignore failures
            try:
                next_page = extract_page(book, page_n + 1)
                fast_parse(next_page)
            except Exception as exc:
                _ = exc  # next-page cache miss is non-fatal
        except Exception as e:
            banner("EXTRACTION FAILURE")
            print(e); pause(); return

        banner(f"{book.name[:70]} — PAGE {page_n}")
        print(f"extract={page['extract_method']} | chars={page['chars']} | hash={page['page_hash'][:12]} | parser={PARSER_VERSION}")
        print("\nPAGE TEXT")
        print("─" * 96)
        wrap_print(page["text"], max_lines=30)
        print("─" * 96)
        print("\nINDY MARGIN NOTES")
        for note in parser["notes"]:
            print(f"▸ {note}")
        print("\nGO TERMS:", " ".join(f"#{p}" for p in parser.get("terms", parser.get("primitives", []))))
        print(f"BPS: {parser['confidence_bps']} | lifecycle: {parser['claim_lifecycle']}")
        print("\nOptions: [j]udge  [p]terms  [a]dapters  [s]kip/comment  [q]uit to library")
        ans = input("move> ").strip().lower()
        if ans == "p":
            term_browser(); continue
        if ans == "a":
            adapter_browser(); continue
        if ans == "q":
            return
        if ans == "s":
            decision, score = "comment", 50
        else:
            decision, score = judgment_prompt()
        notes = input("Your notes / correction / piss judgment: ").strip()
        repair = input("Repair instruction (optional): ").strip() if decision in {"needs_repair", "rejected"} else ""
        term_correction = input("Term correction (#TERMS or blank): ").strip()
        favorite_line = input("Favorite/important line (optional): ").strip()
        confusion = input("Confusion / carry-forward question (optional): ").strip()
        row = {
            "timestamp": now(),
            "book_id": book.id,
            "book_name": book.name,
            "page": page_n,
            "page_hash": page["page_hash"],
            "extract_method": page["extract_method"],
            "parser_version": PARSER_VERSION,
            "packet_id": parser["packet_id"],
            "parser_terms": "|".join(parser.get("terms", parser.get("primitives", []))),
            "parser_bps": parser["confidence_bps"],
            "decision": decision,
            "score": score,
            "score_label": score_label(score),
            "term_correction": term_correction,
            "notes": notes,
            "repair_instruction": repair,
            "favorite_line": favorite_line,
            "confusion": confusion,
            "raw_csv_json": json.dumps({"page": page, "parser": parser}, sort_keys=True),
        }
        append_csv(row)
        bs.setdefault("completed_pages", []).append(page_n)
        bs["current_page"] = page_n + 1
        bs["last_judgment"] = row
        save_state(st)
        print(f"\nSaved to CSV: {CSV_PATH}")
        print(f"Round result: {score_label(score)} ({score}) — page {page_n} locked. Advancing to page {page_n+1}.")
        pause()


def judgment_prompt() -> tuple[str, int]:
    print("\nDecision:")
    opts = ["approved", "needs_repair", "rejected", "comment"]
    for i, o in enumerate(opts, 1): print(f" {i}. {o}")
    while True:
        ans = input("decision> ").strip().lower()
        if ans.isdigit() and 1 <= int(ans) <= len(opts):
            decision = opts[int(ans)-1]; break
        if ans in opts:
            decision = ans; break
    while True:
        try:
            score = max(0, min(100, int(input("score 0-100> ").strip())))
            return decision, score
        except ValueError:
            print("number please")


def main() -> int:
    ensure_dirs()
    st = load_state()
    while True:
        b = pick_book(st)
        if not b:
            banner("EXIT")
            print(f"CSV data: {CSV_PATH}")
            print(f"Persona config: {PERSONA_CONFIG_PATH}")
            print(f"Adapter registry: {ADAPTER_REGISTRY_PATH}")
            print("INDY_READs paused.")
            return 0
        standard_flow(b, st)


if __name__ == "__main__":
    raise SystemExit(main())
