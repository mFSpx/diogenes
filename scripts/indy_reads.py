#!/usr/bin/env python3
"""INDY_READs — GO-25 page-by-page reading game.

INDY_READs is a she: reading companion, margin-noter, judgment collector.

Dynamic library: /home/mfspx/LUCIDOTA/BOOKS
State/data:      /home/mfspx/LUCIDOTA/BOOKS/.indy_reads

No page rewind. One page at a time. Fast heuristic v0.50 parser notes.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import html
import json
import os
import re
import pickle
import resource
import select
import socket
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psycopg
except Exception:  # pragma: no cover - psycopg should exist in the active venv
    psycopg = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ROOT / "BOOKS"
DATA = BOOKS / ".indy_reads"
PAGES = DATA / "pages"
WIKI_DIR = DATA / "wiki"
WIKI_PAGES_DIR = WIKI_DIR / "pages"
JOURNAL_DIR = DATA / "private_journal"
CACHE = DATA / "parser_cache"
STATE_PATH = DATA / "state.json"
CSV_PATH = DATA / "indy_reads_judgments.csv"
SCHEMA_PATH = BOOKS / "GO_GAME_GRADING_SCHEMA.json"
ONTOLOGY_PATH = BOOKS / "GO_ACTIVE_TERMS.json"
RIVER_MODEL_PATH = DATA / "indy_reads_attention_model.pkl"
TRANSPORT_SOCKET = Path("/tmp/lucidota_ego.sock")
INDY_CONDUIT_RECEIPT_DIR = ROOT / "05_OUTPUTS" / "indy_conduit"
INDY_OPERATOR_RESPONSE_OUTBOX = ROOT / "05_OUTPUTS" / "indy_conduit" / "indy_operator_responses.jsonl"
GOALS_HANDOFF_MD = ROOT / "GOALS" / "CURRENT_HANDOFF.md"
GOALS_NEXT_GOAL_QUEUE = ROOT / "GOALS" / "NEXT_GOAL_QUEUE.json"
DB_URL = os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"
ATTENTION_TIMEOUT_SECONDS = 45.0
AUTONOMOUS_TICK_SECONDS = 1.0
PARSER_VERSION = "go_fast_indy_reads_v0.1"
PERSONA_ID = "INDY_READs"
DAEMON_NAME = "indy_reads"
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
    for p in [BOOKS, DATA, PAGES, WIKI_DIR, WIKI_PAGES_DIR, JOURNAL_DIR, CACHE, ROOT / "04_RUNTIME"]:
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


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path.resolve())


def write_journal_entry(
    *,
    title: str,
    body: str,
    kind: str = "note",
    journal_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    target_dir = Path(journal_dir) if journal_dir is not None else JOURNAL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_title = slug(title)
    path = target_dir / f"{now().replace(':', '').replace('-', '').replace('T', '_').replace('+', '_')}_{safe_title}.md"
    payload = {
        "schema": "lucidota.indy_reads.journal_entry.v1",
        "kind": kind,
        "title": title,
        "body": body,
        "created_at": now(),
        "source": "indy_reads",
        "path": rel_or_abs(path),
        "abs_path": str(path.resolve()),
    }
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"- kind: {kind}",
                f"- created_at: {payload['created_at']}",
                "",
                body.strip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    payload["sha256"] = sha_file(path)
    return payload


def write_wiki_page(
    *,
    title: str,
    body: str,
    wiki_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    target_dir = Path(wiki_dir) if wiki_dir is not None else WIKI_PAGES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{slug(title)}.md"
    page = {
        "schema": "lucidota.indy_reads.wiki_page.v1",
        "title": title,
        "body": body,
        "created_at": now(),
        "source": "indy_reads",
        "path": rel_or_abs(path),
        "abs_path": str(path.resolve()),
    }
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                body.strip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    page["sha256"] = sha_file(path)
    return page


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
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    else:
        state = {"books": {}, "active_book_id": ""}
    state.setdefault("books", {})
    state.setdefault("active_book_id", "")
    state.setdefault("slow_lane", {})
    state["slow_lane"].setdefault("ingestion_batch_size", 4)
    state["slow_lane"].setdefault("transport_socket", str(TRANSPORT_SOCKET))
    state["slow_lane"].setdefault("last_autonomous_tick_at", "")
    return state


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
        ans_raw = timed_input("\nPick book: ", ATTENTION_TIMEOUT_SECONDS)
        if ans_raw is TIMEOUT_SENTINEL:
            socket_active = transport_socket_active()
            if socket_active:
                tune_and_record_heartbeat(st, None, None, None, score=None, terminal_active=False)
                print("\nCollaborative companion mode: local transport socket is active.")
            else:
                run_autonomous_slow_lane_tick(st, None, None, None)
                print("\nAutonomous slot claimed while waiting for a book selection.")
            continue
        ans = str(ans_raw).strip().lower()
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


def db_available() -> bool:
    return psycopg is not None


def hardware_telemetry() -> dict[str, Any]:
    telemetry: dict[str, Any] = {
        "cpu_count": os.cpu_count() or 1,
        "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        telemetry.update({
            "memory_total_bytes": int(vm.total),
            "memory_available_bytes": int(vm.available),
            "memory_percent": float(vm.percent),
        })
    except Exception:
        pass
    return telemetry


def transport_socket_active(path: Path = TRANSPORT_SOCKET) -> bool:
    if not path.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.1)
            return sock.connect_ex(str(path)) == 0
    except OSError:
        return False


def load_attention_model() -> Any | None:
    if not RIVER_MODEL_PATH.exists():
        try:
            from river import compose, linear_model, preprocessing  # type: ignore
        except Exception:
            return None
        return compose.Pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression())
    try:
        return pickle.loads(RIVER_MODEL_PATH.read_bytes())
    except Exception:
        return None


def save_attention_model(model: Any) -> None:
    try:
        RIVER_MODEL_PATH.write_bytes(pickle.dumps(model))
    except Exception:
        pass


def attention_features(book: Book | None, page: dict[str, Any] | None, parser: dict[str, Any] | None, socket_active: bool, score: int | None = None) -> dict[str, Any]:
    page_chars = int(page.get("chars", 0)) if page else 0
    parser_bps = int((parser or {}).get("confidence_bps", 0))
    completed_pages = 0
    if page:
        try:
            completed_pages = len(page.get("completed_pages", []))
        except Exception:
            completed_pages = 0
    return {
        "page_chars": page_chars,
        "parser_bps": parser_bps,
        "completed_pages": completed_pages,
        "socket_active": int(socket_active),
        "score": int(score or 0),
        "book_size_bytes": int(getattr(book, "size_bytes", 0) or 0),
    }


def tune_ingestion_batch_size(st: dict[str, Any], book: Book | None, page: dict[str, Any] | None, parser: dict[str, Any] | None, socket_active: bool, score: int | None = None) -> dict[str, Any]:
    model = load_attention_model()
    features = attention_features(book, page, parser, socket_active, score=score)
    proba = 0.5
    if model is not None:
        try:
            probs = model.predict_proba_one(features)
            proba = float(probs.get(True, probs.get(1, 0.5)))
        except Exception:
            proba = 0.5
        try:
            label = bool((score or 0) >= 80)
            model.learn_one(features, label)
            save_attention_model(model)
        except Exception:
            pass
    batch_size = max(1, min(32, int(round(2 + proba * 14 + min(features["page_chars"] / 1200.0, 6.0)))))
    slow_lane = st.setdefault("slow_lane", {})
    slow_lane["ingestion_batch_size"] = batch_size
    slow_lane["river_probability"] = proba
    slow_lane["last_feature_vector"] = features
    slow_lane["transport_socket_active"] = socket_active
    slow_lane["updated_at"] = now()
    return {"batch_size": batch_size, "river_probability": proba, "features": features}


def record_daemon_heartbeat(*, daemon_name: str, socket_active: bool, terminal_active: bool, batch_size: int | None, book: Book | None = None, page: dict[str, Any] | None = None, parser: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> None:
    if not db_available():
        return
    telemetry = hardware_telemetry()
    if page:
        telemetry["page"] = {"book_id": page.get("book_id"), "page": page.get("page"), "page_hash": page.get("page_hash")}
    if parser:
        telemetry["parser"] = {"parser_version": parser.get("parser_version"), "confidence_bps": parser.get("confidence_bps"), "terms": parser.get("terms", [])}
    if book:
        telemetry["book"] = {"book_id": book.id, "book_name": book.name}
    if extra:
        telemetry["extra"] = extra
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ironclaw.daemon_heartbeats
                      (daemon_name, host_name, process_id, transport_socket, socket_active, terminal_active, batch_size, river_state, telemetry, detail)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
                    """,
                    (
                        daemon_name,
                        socket.gethostname(),
                        os.getpid(),
                        str(TRANSPORT_SOCKET),
                        socket_active,
                        terminal_active,
                        batch_size,
                        json.dumps({"book_id": getattr(book, "id", ""), "page": page.get("page") if page else None, "score": extra.get("score") if extra else None}),
                        json.dumps(telemetry, default=str),
                        json.dumps({"source": "scripts/indy_reads.py", "attention": "collaborative" if socket_active else "autonomous", **(extra or {})}, default=str),
                    ),
                )
            conn.commit()
    except Exception:
        return


def record_indy_judgment(*, book: Book, page: dict[str, Any], parser: dict[str, Any], decision: str, score: int, score_label_value: str, notes: str, repair_instruction: str, term_correction: str, favorite_line: str, confusion: str, socket_active: bool, terminal_active: bool, batch_size: int | None, extra: dict[str, Any] | None = None) -> None:
    if not db_available():
        return
    telemetry = hardware_telemetry()
    telemetry.update({
        "attention_state": "collaborative" if socket_active else "autonomous",
        "batch_size": batch_size,
        "cpu_count": telemetry.get("cpu_count"),
    })
    source_payload = {
        "page": page,
        "parser": parser,
        "extra": extra or {},
    }
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ironclaw.indy_read_judgments
                      (daemon_name, book_id, book_name, page_number, page_hash, parser_version, decision, score, score_label, term_correction, notes, repair_instruction, favorite_line, confusion, transport_socket, socket_active, terminal_active, batch_size, telemetry, source_payload)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                    """,
                    (
                        DAEMON_NAME,
                        book.id,
                        book.name,
                        int(page["page"]),
                        page["page_hash"],
                        parser["parser_version"],
                        decision,
                        int(score),
                        score_label_value,
                        term_correction,
                        notes,
                        repair_instruction,
                        favorite_line,
                        confusion,
                        str(TRANSPORT_SOCKET),
                        socket_active,
                        terminal_active,
                        batch_size,
                        json.dumps(telemetry, default=str),
                        json.dumps(source_payload, default=str),
                    ),
                )
            conn.commit()
    except Exception:
        return


def tune_and_record_heartbeat(st: dict[str, Any], book: Book | None, page: dict[str, Any] | None, parser: dict[str, Any] | None, *, score: int | None = None, terminal_active: bool = False) -> dict[str, Any]:
    socket_active = transport_socket_active()
    tune = tune_ingestion_batch_size(st, book, page, parser, socket_active, score=score)
    record_daemon_heartbeat(
        daemon_name=DAEMON_NAME,
        socket_active=socket_active,
        terminal_active=terminal_active,
        batch_size=tune["batch_size"],
        book=book,
        page=page,
        parser=parser,
        extra={"score": score, "river_probability": tune["river_probability"]},
    )
    save_state(st)
    return {"socket_active": socket_active, **tune}


def timed_input(prompt: str, timeout_seconds: float | None = None) -> str | object:
    if timeout_seconds is None or not sys.stdin.isatty():
        return input(prompt)
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if ready:
        return sys.stdin.readline().rstrip("\n")
    return TIMEOUT_SENTINEL


TIMEOUT_SENTINEL = object()


def run_autonomous_slow_lane_tick(st: dict[str, Any], book: Book | None, page: dict[str, Any] | None, parser: dict[str, Any] | None) -> dict[str, Any]:
    tick = tune_and_record_heartbeat(st, book, page, parser, score=(parser or {}).get("confidence_bps", 0), terminal_active=False)
    slow_lane = st.setdefault("slow_lane", {})
    slow_lane["last_autonomous_tick_at"] = now()
    slow_lane["last_autonomous_reason"] = "terminal_timeout"
    save_state(st)
    return tick


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
        attention = "collaborative" if transport_socket_active() else "autonomous"
        print(f"Attention: {attention} | transport socket: {TRANSPORT_SOCKET}")
        print("\nOptions: [j]udge  [p]terms  [a]dapters  [s]kip/comment  [q]uit to library")
        ans_raw = timed_input("move> ", ATTENTION_TIMEOUT_SECONDS)
        if ans_raw is TIMEOUT_SENTINEL:
            socket_active = transport_socket_active()
            if socket_active:
                tune_and_record_heartbeat(st, book, page, parser, score=parser.get("confidence_bps", 0), terminal_active=False)
                print("\nCollaborative companion mode: local transport socket is active. Waiting for operator input.")
            else:
                run_autonomous_slow_lane_tick(st, book, page, parser)
                print("\nAutonomous slot claimed: tuned batch size and wrote daemon heartbeat.")
            continue
        ans = str(ans_raw).strip().lower()
        if ans == "p":
            term_browser(); continue
        if ans == "a":
            adapter_browser(); continue
        if ans == "q":
            return
        if ans == "s":
            decision, score = "comment", 50
        else:
            judgment = judgment_prompt(st=st, book=book, page=page, parser=parser)
            if judgment is None:
                continue
            decision, score = judgment
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
        record_indy_judgment(
            book=book,
            page=page,
            parser=parser,
            decision=decision,
            score=score,
            score_label_value=score_label(score),
            notes=notes,
            repair_instruction=repair,
            term_correction=term_correction,
            favorite_line=favorite_line,
            confusion=confusion,
            socket_active=transport_socket_active(),
            terminal_active=True,
            batch_size=int(st.get("slow_lane", {}).get("ingestion_batch_size", 0) or 0) or None,
            extra={"csv_row": row},
        )
        tune_and_record_heartbeat(st, book, page, parser, score=score, terminal_active=True)
        bs.setdefault("completed_pages", []).append(page_n)
        bs["current_page"] = page_n + 1
        bs["last_judgment"] = row
        save_state(st)
        print(f"\nSaved to CSV: {CSV_PATH}")
        print(f"Round result: {score_label(score)} ({score}) — page {page_n} locked. Advancing to page {page_n+1}.")
        pause()


def judgment_prompt(*, st: dict[str, Any], book: Book, page: dict[str, Any], parser: dict[str, Any]) -> tuple[str, int] | None:
    print("\nDecision:")
    opts = ["approved", "needs_repair", "rejected", "comment"]
    for i, o in enumerate(opts, 1): print(f" {i}. {o}")
    while True:
        ans_raw = timed_input("decision> ", ATTENTION_TIMEOUT_SECONDS)
        if ans_raw is TIMEOUT_SENTINEL:
            socket_active = transport_socket_active()
            if socket_active:
                tune_and_record_heartbeat(st, book, page, parser, score=parser.get("confidence_bps", 0), terminal_active=False)
                print("\nCollaborative companion mode detected at decision prompt; waiting for operator.")
            else:
                run_autonomous_slow_lane_tick(st, book, page, parser)
                print("\nDecision prompt timed out; autonomous slot claimed.")
            return None
        ans = str(ans_raw).strip().lower()
        if ans.isdigit() and 1 <= int(ans) <= len(opts):
            decision = opts[int(ans)-1]; break
        if ans in opts:
            decision = ans; break
    while True:
        score_raw = timed_input("score 0-100> ", ATTENTION_TIMEOUT_SECONDS)
        if score_raw is TIMEOUT_SENTINEL:
            socket_active = transport_socket_active()
            if socket_active:
                tune_and_record_heartbeat(st, book, page, parser, score=parser.get("confidence_bps", 0), terminal_active=False)
                print("\nCollaborative companion mode detected at score prompt; waiting for operator.")
            else:
                run_autonomous_slow_lane_tick(st, book, page, parser)
                print("\nScore prompt timed out; autonomous slot claimed.")
            return None
        try:
            score = max(0, min(100, int(str(score_raw).strip())))
            return decision, score
        except ValueError:
            print("number please")


def load_goals_handoff_text() -> str:
    if GOALS_HANDOFF_MD.exists():
        try:
            return GOALS_HANDOFF_MD.read_text(encoding="utf-8")
        except OSError:
            return ""
    return ""


def load_queued_conduit_dialogue(limit: int = 5) -> list[dict[str, Any]]:
    """Read queued Matrix/Conduit rows for the Indy_READs operator chat surface."""
    if not db_available():
        return []
    try:
        from indy_conduit_driver import read_queued_dialogue_rows
        with psycopg.connect(DB_URL) as conn:  # type: ignore[union-attr]
            return read_queued_dialogue_rows(conn, limit=limit)
    except Exception:
        return []


def format_conduit_dialogue_row(row: dict[str, Any], idx: int) -> str:
    clean = str(row.get("clean_text") or row.get("raw_text") or "").replace("\n", " ").strip()
    if len(clean) > 220:
        clean = clean[:217] + "..."
    sender = row.get("sender_id") or "matrix"
    event_id = row.get("event_id") or row.get("id") or ""
    return f"{idx}. {sender} {event_id}: {clean}"


def queued_dialogue_context(row: dict[str, Any]) -> tuple[Book, dict[str, Any], dict[str, Any]]:
    text = str(row.get("clean_text") or row.get("raw_text") or "")
    row_id = str(row.get("id") or row.get("event_id") or "")
    book = Book(
        id=f"waking_dialogue::{row_id}",
        name="ironclaw.waking_dialogue_stream",
        path="postgresql:///lucidota_state/ironclaw.waking_dialogue_stream",
        ext=".db",
        size_bytes=len(text.encode("utf-8")),
    )
    page = {
        "page": 1,
        "page_hash": sha_text(json.dumps(row, sort_keys=True, default=str)),
        "text": text,
        "extract_method": "waking_dialogue_stream",
        "chars": len(text),
    }
    entities = row.get("extracted_entities") if isinstance(row.get("extracted_entities"), dict) else {}
    parser = {
        "parser_version": "waking_dialogue_chat_v1",
        "packet_id": f"waking_dialogue::{row_id}",
        "confidence_bps": 10000,
        "terms": list(entities.get("hashtags") or []) + list(entities.get("slash_commands") or []),
        "claim_lifecycle": "WAKING_DIALOGUE_CHAT",
    }
    return book, page, parser


def parse_conduit_response_command(ans: str, dialogue_rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, str]:
    parts = ans.split(maxsplit=2)
    if len(parts) < 2 or parts[0].lower() != "respond":
        return None, "", "not_respond_command"
    idx = int(parts[1]) if parts[1].isdigit() else 1
    if not (1 <= idx <= len(dialogue_rows)):
        return None, "", "dialogue_index_out_of_range"
    reply_text = parts[2].strip() if len(parts) > 2 else ""
    return dialogue_rows[idx - 1], reply_text, ""


def dialogue_row_ref(row: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(row.get("id") or ""),
        "event_id": str(row.get("event_id") or ""),
        "room_id": str(row.get("room_id") or ""),
        "sender_id": str(row.get("sender_id") or ""),
        "receipt_id": str(row.get("receipt_id") or ""),
        "comms_channel": str(row.get("comms_channel") or "matrix"),
    }


def operator_response_id(row: dict[str, Any], reply_text: str) -> str:
    return "indy_response:" + sha_text(
        json.dumps(
            {
                "schema": "lucidota.indy_reads.operator_chat_response.v1",
                "dialogue_row": dialogue_row_ref(row),
                "reply_text": reply_text,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    )[:16]


def pid_ram_guard() -> dict[str, Any]:
    telemetry = hardware_telemetry()
    return {
        "pid_check_performed": True,
        "process_id": os.getpid(),
        "rss_bytes": int(telemetry.get("rss_bytes") or 0),
        "memory_available_bytes": int(telemetry.get("memory_available_bytes") or 0),
        "memory_percent": float(telemetry.get("memory_percent") or 0.0),
        "cpu_count": int(telemetry.get("cpu_count") or 1),
        "heavy_model_launch_performed": False,
    }


def queue_operator_chat_response(
    row: dict[str, Any],
    reply_text: str,
    outbox: Path | None = None,
    *,
    db_identity: dict[str, Any] | None = None,
    db_api_status: str = "db_api_unavailable_fallback",
) -> dict[str, Any]:
    """Queue Indy_READs' chat response for the active operator surface.

    This is deliberately local and quiet: no Matrix/email/Signal send occurs
    here, and PID/RAM guard facts are captured before any sender can pick the
    packet up.
    """
    outbox = outbox or INDY_OPERATOR_RESPONSE_OUTBOX
    response_id = operator_response_id(row, reply_text)
    guard = pid_ram_guard()
    packet = {
        "schema": "lucidota.indy_reads.operator_chat_response.v1",
        "queued_at": now(),
        "response_id": response_id,
        "persona": PERSONA_DISPLAY,
        "target_path": "active_operator_chat_surface",
        "route": "luci_operator_direct_chat",
        "source_table": "ironclaw.waking_dialogue_stream",
        "dialogue_row": dialogue_row_ref(row),
        "body": reply_text,
        "body_sha256": sha_text(reply_text),
        "operator_delivery_status": "QUEUED_FOR_CHAT_SURFACE",
        "outbound_matrix_send_performed": False,
        "direct_network_send_performed": False,
        "db_api_status": db_api_status,
        "db_identity": db_identity or {},
        "pid_ram_guard": guard,
    }
    outbox.parent.mkdir(parents=True, exist_ok=True)
    with outbox.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(packet, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    return {
        "ok": True,
        "response_id": response_id,
        "operator_delivery_status": packet["operator_delivery_status"],
        "operator_response_outbox": str(outbox),
        "db_api_status": db_api_status,
        "db_identity": db_identity or {},
        "outbound_matrix_send_performed": False,
        "direct_network_send_performed": False,
        "pid_ram_guard": guard,
    }


def mark_conduit_dialogue_done(row: dict[str, Any], response_id: str, reply_text: str) -> dict[str, Any]:
    if not db_available():
        return {"ok": False, "error": "database_unavailable", "processed_status": ""}
    row_id = str(row.get("id") or "")
    event_id = str(row.get("event_id") or "")
    body_sha = sha_text(reply_text)
    try:
        with psycopg.connect(DB_URL) as conn:  # type: ignore[union-attr]
            with conn.cursor() as cur:
                if row_id:
                    cur.execute(
                        """
                        UPDATE ironclaw.waking_dialogue_stream
                        SET processed_status = 'done',
                            receipt_id = CASE WHEN receipt_id = '' THEN %s ELSE receipt_id END,
                            last_response_id = %s,
                            last_response_body = %s,
                            last_response_body_sha256 = %s,
                            response_queued_at = now(),
                            response_delivery_status = 'QUEUED_FOR_CHAT_SURFACE',
                            updated_at = now()
                        WHERE id = %s::uuid
                        RETURNING id::text, processed_status, receipt_id,
                            last_response_id, response_delivery_status,
                            response_queued_at, last_response_body_sha256
                        """,
                        (response_id, response_id, reply_text, body_sha, row_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE ironclaw.waking_dialogue_stream
                        SET processed_status = 'done',
                            receipt_id = CASE WHEN receipt_id = '' THEN %s ELSE receipt_id END,
                            last_response_id = %s,
                            last_response_body = %s,
                            last_response_body_sha256 = %s,
                            response_queued_at = now(),
                            response_delivery_status = 'QUEUED_FOR_CHAT_SURFACE',
                            updated_at = now()
                        WHERE comms_channel = 'matrix'
                          AND event_id = %s
                        RETURNING id::text, processed_status, receipt_id,
                            last_response_id, response_delivery_status,
                            response_queued_at, last_response_body_sha256
                        """,
                        (response_id, response_id, reply_text, body_sha, event_id),
                    )
                rows = cur.fetchall()
            conn.commit()
        if not rows:
            return {
                "ok": False,
                "updated_rows": 0,
                "error": "dialogue_row_not_found",
                "processed_status": "",
                "response_id": response_id,
                "response_body_sha256": body_sha,
            }
        return {
            "ok": True,
            "updated_rows": len(rows),
            "dialogue_id": rows[0][0] if rows else "",
            "processed_status": rows[0][1] if rows else "",
            "receipt_id": rows[0][2] if rows else "",
            "response_id": rows[0][3] if rows else "",
            "response_delivery_status": rows[0][4] if rows else "",
            "response_queued_at": rows[0][5] if rows else "",
            "response_body_sha256": rows[0][6] if rows else body_sha,
        }
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)[:240], "processed_status": ""}


def default_conduit_reply(row: dict[str, Any]) -> str:
    clean = str(row.get("clean_text") or row.get("raw_text") or "").replace("\n", " ").strip()
    if len(clean) > 180:
        clean = clean[:177] + "..."
    sender = str(row.get("sender_id") or "operator")
    return f"Indy_READs saw queued chat from {sender}: {clean}"


def write_online_once_receipt(payload: dict[str, Any], receipt_dir: Path) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    digest = sha_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))[:16]
    stamp_value = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = receipt_dir / f"indy_reads_online_once_{stamp_value}_{digest}.json"
    payload["receipt_path"] = str(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def process_queued_conduit_once(
    st: dict[str, Any],
    *,
    limit: int = 5,
    response_text: str | None = None,
    receipt_dir: Path | str = INDY_CONDUIT_RECEIPT_DIR,
) -> dict[str, Any]:
    guard = pid_ram_guard()
    rows = load_queued_conduit_dialogue(limit=limit)
    if not rows:
        payload = {
            "schema": "lucidota.indy_reads.online_once_receipt.v1",
            "generated_at": now(),
            "ok": True,
            "status": "IDLE_NO_QUEUED_DIALOGUE",
            "row": None,
            "response": None,
            "pid_ram_guard": guard,
            "model_calls_performed": False,
            "heavy_model_launch_performed": False,
        }
        receipt_path = write_online_once_receipt(payload, Path(receipt_dir))
        return {**payload, "receipt_path": str(receipt_path)}
    row = rows[0]
    reply = response_text if response_text is not None else default_conduit_reply(row)
    response = record_conduit_dialogue_response(row, reply, st)
    payload = {
        "schema": "lucidota.indy_reads.online_once_receipt.v1",
        "generated_at": now(),
        "ok": bool(response.get("ok")),
        "status": "RESPONDED" if response.get("ok") else "RESPONSE_FAILED",
        "row": dialogue_row_ref(row),
        "response": response,
        "pid_ram_guard": guard,
        "model_calls_performed": False,
        "heavy_model_launch_performed": False,
    }
    receipt_path = write_online_once_receipt(payload, Path(receipt_dir))
    return {**payload, "receipt_path": str(receipt_path)}


def record_conduit_dialogue_response(row: dict[str, Any], reply_text: str, st: dict[str, Any]) -> dict[str, Any]:
    book, page, parser = queued_dialogue_context(row)
    response_id = operator_response_id(row, reply_text)
    processed_status_update = mark_conduit_dialogue_done(row, response_id, reply_text)
    db_api_status = "ok" if processed_status_update.get("ok") else "db_api_unavailable_fallback"
    operator_response = queue_operator_chat_response(
        row,
        reply_text,
        db_identity=processed_status_update,
        db_api_status=db_api_status,
    )
    record_indy_judgment(
        book=book,
        page=page,
        parser=parser,
        decision="comment",
        score=100 if reply_text else 50,
        score_label_value=score_label(100 if reply_text else 50),
        notes=reply_text,
        repair_instruction="",
        term_correction="",
        favorite_line="",
        confusion="",
        socket_active=transport_socket_active(),
        terminal_active=True,
        batch_size=int(st.get("slow_lane", {}).get("ingestion_batch_size", 0) or 0) or None,
        extra={
            "dialogue_row": row,
            "reply": reply_text,
            "response_kind": "terminal_conduit_response",
            "operator_response": operator_response,
            "processed_status_update": processed_status_update,
            "db_api_status": db_api_status,
            "outbound_matrix_send_performed": False,
            "direct_network_send_performed": False,
        },
    )
    tune_and_record_heartbeat(st, book, page, parser, score=100 if reply_text else 50, terminal_active=True)
    return {
        "ok": bool(operator_response.get("ok")) and bool(processed_status_update.get("ok")),
        "decision": "comment",
        "score": 100 if reply_text else 50,
        "response_id": operator_response["response_id"],
        "operator_response_queued": bool(operator_response.get("ok")),
        "operator_delivery_status": operator_response["operator_delivery_status"],
        "operator_response_outbox": operator_response["operator_response_outbox"],
        "db_api_status": db_api_status,
        "processed_status_update": processed_status_update,
        "outbound_matrix_send_performed": False,
        "direct_network_send_performed": False,
    }


def load_next_goal_queue() -> list[dict[str, Any]]:
    if not GOALS_NEXT_GOAL_QUEUE.exists():
        return []
    try:
        data = json.loads(GOALS_NEXT_GOAL_QUEUE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    orders = data.get("queue", [])
    if not isinstance(orders, list):
        return []
    return [o for o in orders if isinstance(o, dict)]


def goals_handoff_context(text: str, orders: list[dict[str, Any]]) -> tuple[Book, dict[str, Any], dict[str, Any]]:
    book = Book(
        id="goals_handoff",
        name="GOALS/CURRENT_HANDOFF.md",
        path=str(GOALS_HANDOFF_MD),
        ext=".md",
        size_bytes=len(text.encode("utf-8")),
    )
    page = {
        "page": 1,
        "page_hash": sha_text(text + json.dumps(orders, sort_keys=True, default=str)),
        "text": text,
        "extract_method": "goals_handoff",
        "chars": len(text),
    }
    parser = {
        "parser_version": "goals_handoff_chat_v1",
        "packet_id": "goals::handoff::v1",
        "confidence_bps": 10000 if orders else 7500,
        "terms": ["SESSION", "HANDOFF", "WORK_ORDER", "QUEUE"],
        "claim_lifecycle": "GOALS_CHAT",
    }
    return book, page, parser


def enqueue_goal_work_order(order: dict[str, Any]) -> dict[str, Any]:
    if not db_available():
        return {"ok": False, "error": "database_unavailable", "order_id": order.get("order_id", "")}
    queue = str(order.get("queue") or "control")
    workflow = str(order.get("workflow") or "goal_work_order")
    job_kind = str(order.get("job_kind") or "external_command")
    payload = dict(order.get("payload") or {})
    if not payload:
        payload = {
            "command": [
                ".venv/bin/python",
                "scripts/goal_swarm_dispatch.py",
                "--target",
                "generic",
                "--task",
                str(order.get("objective") or order.get("title") or "goal continuation"),
                "--jobs",
                "1",
                "--json",
            ]
        }
    idempotency_key = str(order.get("order_id") or sha256_obj(order))
    result: dict[str, Any] = {
        "ok": False,
        "queue": queue,
        "workflow": workflow,
        "job_kind": job_kind,
        "idempotency_key": idempotency_key,
    }
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.absurd_queue_job
                      (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
                    VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                    ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at=now()
                    RETURNING job_uuid::text, (xmax = 0) AS inserted_new
                    """,
                    (
                        queue,
                        workflow,
                        job_kind,
                        idempotency_key,
                        json.dumps(payload, default=str),
                        int(order.get("priority") or 100),
                        int(order.get("max_attempts") or 3),
                        json.dumps({"source": "indy_reads_chat", "order_id": order.get("order_id", ""), "title": order.get("title", "")}, default=str),
                    ),
                )
                job_uuid, inserted_new = cur.fetchone()
                if inserted_new:
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, detail)
                        VALUES (%s,%s,'enqueued',%s::jsonb)
                        """,
                        (
                            job_uuid,
                            queue,
                            json.dumps({"workflow": workflow, "job_kind": job_kind, "order_id": order.get("order_id", "")}, default=str),
                        ),
                    )
                conn.commit()
                result.update({"ok": True, "job_uuid": job_uuid, "inserted_new": bool(inserted_new)})
    except Exception as exc:
        result.update({"error": type(exc).__name__, "message": str(exc)})
    return result


def goals_chat_loop(st: dict[str, Any]) -> int:
    while True:
        handoff_text = load_goals_handoff_text()
        orders = load_next_goal_queue()
        book, page, parser = goals_handoff_context(handoff_text, orders)
        banner("GOALS CHAT — handoff / next orders / operator reply")
        print("CURRENT HANDOFF")
        print("─" * 96)
        wrap_print(handoff_text or "(no GOALS/CURRENT_HANDOFF.md found)", max_lines=40)
        print("─" * 96)
        conduit_rows = load_queued_conduit_dialogue(limit=5)
        print("\nMATRIX / CONDUIT QUEUE FOR Indy_READs")
        if not conduit_rows:
            print("(no queued ironclaw.waking_dialogue_stream rows visible to Indy_READs)")
        else:
            for i, row in enumerate(conduit_rows, 1):
                print(format_conduit_dialogue_row(row, i))
        print("\nNEXT GOAL QUEUE")
        if not orders:
            print("(no GOALS/NEXT_GOAL_QUEUE.json found)")
        else:
            for i, order in enumerate(orders, 1):
                print(f"{i}. {order.get('title', order.get('order_id', 'goal'))}")
                print(f"   queue={order.get('queue', 'control')} workflow={order.get('workflow', '')} job_kind={order.get('job_kind', '')}")
                if order.get("objective"):
                    print(f"   objective={order.get('objective')}")
                elif order.get("summary"):
                    print(f"   summary={order.get('summary')}")
        print("\nReplies: `respond 1 text...`, `approve 1`, `reject 2`, `note ...`, `enqueue 3`, `q` to quit")
        ans_raw = timed_input("reply> ", ATTENTION_TIMEOUT_SECONDS)
        if ans_raw is TIMEOUT_SENTINEL:
            run_autonomous_slow_lane_tick(st, book, page, parser)
            print("\nSession chat timed out; autonomous heartbeat written.")
            continue
        ans = str(ans_raw).strip()
        if ans.lower() in {"q", "quit", "exit"}:
            return 0
        lowered = ans.lower()
        decision = "comment"
        score = 50
        selected_order: dict[str, Any] | None = None
        enqueue_result: dict[str, Any] = {}
        notes = ans
        if lowered.startswith("respond "):
            selected_dialogue, reply_text, error = parse_conduit_response_command(ans, conduit_rows)
            if selected_dialogue is None:
                print(f"No queued dialogue response saved: {error}")
                continue
            response_result = record_conduit_dialogue_response(selected_dialogue, reply_text, st)
            print(f"Indy_READs terminal response saved: {response_result['decision']} {response_result['score']} | outbound_matrix_send=False")
            continue
        if lowered.startswith(("approve ", "enqueue ", "reject ")):
            parts = lowered.split()
            verb = parts[0]
            idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            if 1 <= idx <= len(orders):
                selected_order = orders[idx - 1]
            if verb == "reject":
                decision, score = "rejected", 10
            else:
                decision, score = "approved", 100
                if selected_order is not None:
                    enqueue_result = enqueue_goal_work_order(selected_order)
                    notes = f"{ans} | enqueue={enqueue_result.get('ok', False)}"
            if verb == "reject" and selected_order is not None:
                notes = f"{ans} | rejected"
        elif lowered.startswith("note "):
            notes = ans[5:].strip() or ans
        record_indy_judgment(
            book=book,
            page=page,
            parser=parser,
            decision=decision,
            score=score,
            score_label_value=score_label(score),
            notes=notes,
            repair_instruction="",
            term_correction="",
            favorite_line="",
            confusion="",
            socket_active=transport_socket_active(),
            terminal_active=True,
            batch_size=int(st.get("slow_lane", {}).get("ingestion_batch_size", 0) or 0) or None,
            extra={"selected_order": selected_order, "enqueue_result": enqueue_result, "reply": ans},
        )
        tune_and_record_heartbeat(st, book, page, parser, score=score, terminal_active=True)
        print(f"Judgment saved: {decision} {score} | enqueue={enqueue_result.get('ok', False)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["reader", "chat", "journal"])
    ap.add_argument("--mode", dest="mode_flag", choices=["reader", "chat", "journal"])
    ap.add_argument("--respond-once", action="store_true", help="Process one queued Indy/Conduit chat row and exit.")
    ap.add_argument("--response-text", default=None, help="Optional explicit response body for --respond-once.")
    ap.add_argument("--journal-title", default=None)
    ap.add_argument("--journal-body", default=None)
    ap.add_argument("--journal-kind", default="note")
    ap.add_argument("--receipt-dir", default=str(INDY_CONDUIT_RECEIPT_DIR))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    mode = args.mode_flag or args.mode or "reader"
    ensure_dirs()
    st = load_state()
    tune_and_record_heartbeat(st, None, None, None, score=None, terminal_active=sys.stdin.isatty())
    if mode == "journal":
        title = (args.journal_title or "Indy Journal Entry").strip()
        body = (args.journal_body or sys.stdin.read() or "").strip()
        if not body:
            body = title
        journal = write_journal_entry(title=title, body=body, kind=args.journal_kind)
        wiki = write_wiki_page(title=title, body=body)
        result = {"ok": True, "journal": journal, "wiki": wiki}
        if args.json:
            print(json.dumps(result, sort_keys=True, default=str))
        else:
            print("INDY_JOURNAL=PASS")
            print(f"JOURNAL_PATH={journal['path']}")
            print(f"WIKI_PATH={wiki['path']}")
        return 0
    if mode == "chat":
        if args.respond_once:
            result = process_queued_conduit_once(st, response_text=args.response_text, receipt_dir=Path(args.receipt_dir))
            if args.json:
                print(json.dumps(result, sort_keys=True, default=str))
            else:
                print(f"INDY_READS_CHAT={result['status']}")
                print(f"RECEIPT_PATH={result['receipt_path']}")
                if result.get("response"):
                    print(f"RESPONSE_ID={result['response'].get('response_id', '')}")
            return 0 if result.get("ok") else 1
        return goals_chat_loop(st)
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
