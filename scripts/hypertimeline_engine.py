#!/usr/bin/env python3
"""LUCIDOTA ABSURD hypertimeline sweep, ingest, and circadian engine test."""
from __future__ import annotations

import argparse, csv, hashlib, io, json, mimetypes, os, re, stat, sys, time, zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

import dateparser
import networkx as nx  # dependency sentinel + future graph expansion
import psycopg2
from psycopg2.extras import Json, execute_values
from psycopg2.pool import SimpleConnectionPool
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(os.getenv("LUCIDOTA_ROOT", "/home/mfspx/LUCIDOTA"))
sys.path.insert(0, str(ROOT))
from ALGOS.krampus_chrono import circadian_activity_curve, circadian_match_metrics  # noqa: E402

HOME = Path.home()
DB_DSN = os.getenv("LUCIDOTA_GO_STORAGE_DSN", "dbname=lucidota_storage")
LOG = ROOT / "04_RUNTIME" / "hypertimeline_ops.log"
WORKER_ERROR_LOG = ROOT / "05_OUTPUTS" / "AUTONOMOUS_WORKER_ERRORS.log"
DEFAULT_CASE_KEY = os.getenv("LUCIDOTA_DEFAULT_CASE_KEY", "KE26-00001")
JSON_STREAM_THRESHOLD_BYTES = int(os.getenv("LUCIDOTA_JSON_STREAM_THRESHOLD_MB", "64")) * 1024 * 1024
JSON_STREAM_CHARS = int(os.getenv("LUCIDOTA_JSON_STREAM_CHARS", os.getenv("LUCIDOTA_STREAM_CHUNK_BYTES", str(1024 * 1024))))
HYPERTIMELINE_BATCH_SIZE = int(os.getenv("LUCIDOTA_HYPERTIMELINE_BATCH_SIZE", "512"))
HYPERTIMELINE_POOL_MAX = max(1, int(os.getenv("LUCIDOTA_MAX_WORKERS", "1")))
CANDIDATE_DIRS = [
    ROOT / "03_VAULT" / "cas",
    ROOT / "03_VAULT" / "korpus_krampii" / "DIGESTED",
    ROOT / "01_REPOS" / "claudecode" / "rust" / "target" / "debug",
]
TS_KEYS = {"created_at","created","create_time","timestamp","time","date","datetime","updated_at","sent_at","received_at","start_time","end_time","event_time","ts","time_ms","timestamp_ms"}
CONTENT_KEYS = ("text","content","body","message","messages","description","summary","snippet","title","subject","display")
ENTITY_KEYS = ("sender","author","from","username","user","name","participant","role","email","phone","display_name")
ACTION_KEYS = ("action","type","event","kind","status","display","verb","operation")
SOURCE_KIND = {"facebook":"facebook","whatsapp":"whatsapp","signal":"signal","email":"email","gmail":"email","sms":"sms","imessage":"imessage","cell_phone":"sms"}
console = Console()

@dataclass
class SweepResult:
    gitignore_added: list[str] = field(default_factory=list)
    scanned: int = 0
    duplicate_groups: int = 0
    relinked: int = 0
    skipped: int = 0
    freed_bytes: int = 0
    errors: list[str] = field(default_factory=list)

@dataclass
class IngestStats:
    files_seen: int = 0
    files_loaded: int = 0
    files_failed: int = 0
    events_found: int = 0
    events_inserted: int = 0
    sources: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class Event:
    occurred_at: datetime
    source_label: str
    source_kind: str
    entity: str
    action: str
    content: str
    raw: dict[str, Any]
    source_member: str
    provider_id: str
    sequence: int


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")


def worker_error(msg: str) -> None:
    WORKER_ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()} hypertimeline_engine {msg}\n"
    with WORKER_ERROR_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line)
    log(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fmt_bytes(n: int) -> str:
    for unit in ("B","KB","MB","GB","TB"):
        if abs(n) < 1024 or unit == "TB": return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"


def ensure_gitignore() -> list[str]:
    gi = ROOT / ".gitignore"
    existing = gi.read_text(errors="ignore").splitlines() if gi.exists() else []
    needed = [".codex/log/", ".codex/history.jsonl"]
    added = [p for p in needed if p not in existing]
    if added:
        with gi.open("a", encoding="utf-8") as f:
            if existing and existing[-1].strip(): f.write("\n")
            f.write("\n# Local Codex history/logs: retain on disk, never track.\n")
            for p in added: f.write(p + "\n")
    return added


def is_rust_binary(path: Path) -> bool:
    try:
        if not os.access(path, os.X_OK) or path.stat().st_size < 1_000_000: return False
        with path.open("rb") as f: return f.read(4) == b"\x7fELF"
    except OSError:
        return False


def gc_candidate(path: Path) -> bool:
    if not path.is_file() or ".log" in path.suffixes: return False
    s = path.stat()
    if time.time() - s.st_mtime < 120: return False
    sp = path.as_posix()
    return ("/03_VAULT/cas/" in sp and s.st_size > 1_000_000) or path.suffix.lower() == ".zip" or ("target/debug" in sp and is_rust_binary(path))


def keeper_key(p: Path) -> tuple[int, int, str]:
    s = p.as_posix()
    return (0 if "/03_VAULT/cas/" in s else 1 if "/target/debug/" in s and "/deps/" not in s else 2, len(s), s)


def hardlink_dedupe(dry_run: bool = False) -> SweepResult:
    res = SweepResult(gitignore_added=ensure_gitignore())
    by_size: dict[int, list[Path]] = {}
    for base in CANDIDATE_DIRS:
        if not base.exists(): continue
        for p in base.rglob("*"):
            try:
                if gc_candidate(p):
                    by_size.setdefault(p.stat().st_size, []).append(p); res.scanned += 1
            except OSError as e:
                res.errors.append(f"scan {p}: {e}")
    by_hash: dict[tuple[int, str], list[Path]] = {}
    for size, paths in by_size.items():
        if len(paths) < 2: continue
        for p in paths:
            try: by_hash.setdefault((size, sha256_file(p)), []).append(p)
            except OSError as e: res.errors.append(f"hash {p}: {e}")
    for (size, _), paths in by_hash.items():
        paths = sorted(paths, key=keeper_key)
        if len(paths) < 2: continue
        res.duplicate_groups += 1
        keep = paths[0]
        for dup in paths[1:]:
            try:
                ks, ds = keep.stat(), dup.stat()
                if ks.st_ino == ds.st_ino:
                    res.skipped += 1; continue
                blocks = max(ds.st_blocks * 512, ds.st_size)
                if not dry_run:
                    tmp = dup.with_name(f".{dup.name}.dedupe-{os.getpid()}")
                    if ks.st_dev == ds.st_dev and stat.S_IMODE(ks.st_mode) == stat.S_IMODE(ds.st_mode):
                        os.link(keep, tmp)
                    else:
                        os.symlink(os.path.relpath(keep, dup.parent), tmp)
                    os.replace(tmp, dup)
                res.relinked += 1; res.freed_bytes += blocks
            except OSError as e:
                res.errors.append(f"dedupe {dup}: {e}")
    return res


def source_label(path: str, obj: Any = None) -> str:
    low = path.lower()
    if "chatgpt" in low or "openai" in low or "conversations.json" in low: return "chatgpt"
    if "claude" in low: return "claude"
    if "codex" in low: return "codex"
    if "facebook" in low or "/meta-" in low or "metaai" in low: return "facebook"
    if "whatsapp" in low: return "whatsapp"
    if "gmail" in low or "email" in low or "mail" in low: return "email"
    if any(x in low for x in ("sms","imessage","phone","call_log","cell")): return "cell_phone"
    return "generic"


def kind_for(label: str) -> str:
    return SOURCE_KIND.get(label, "generic")


def parse_ts(v: Any) -> datetime | None:
    try:
        if isinstance(v, (int, float)) or (isinstance(v, str) and re.fullmatch(r"\d+(?:\.\d+)?", v.strip())):
            x = float(v)
            if x > 10**17: x /= 1_000_000_000
            elif x > 10**14: x /= 1_000_000
            elif x > 10**11: x /= 1000
            if x < 631152000 or x > time.time() + 5 * 365 * 86400: return None
            return datetime.fromtimestamp(x, timezone.utc)
        if isinstance(v, str) and v.strip():
            dt = dateparser.parse(v, settings={"TIMEZONE":"UTC","TO_TIMEZONE":"UTC","RETURN_AS_TIMEZONE_AWARE":True})
            if dt: return dt.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError):
        return None
    return None


def iter_json_array_stream(read_chunk, label: str) -> Iterator[tuple[int, Any]]:
    """Incrementally decode a top-level JSON array without loading the whole file/member."""
    decoder = json.JSONDecoder()
    buf = ""
    in_array = False
    eof = False
    idx = 0
    while True:
        if not eof and len(buf) < JSON_STREAM_CHARS * 4:
            chunk = read_chunk()
            if chunk:
                buf += chunk.decode("utf-8", errors="replace") if isinstance(chunk, bytes) else str(chunk)
            else:
                eof = True
        buf = buf.lstrip()
        if not in_array:
            if not buf and not eof:
                continue
            if not buf.startswith("["):
                raise ValueError(f"{label}: large JSON streaming supports top-level arrays; got non-array")
            buf = buf[1:]
            in_array = True
        buf = buf.lstrip()
        if buf.startswith(","):
            buf = buf[1:].lstrip()
        if buf.startswith("]"):
            return
        if not buf:
            if eof:
                return
            continue
        try:
            obj, end = decoder.raw_decode(buf)
        except json.JSONDecodeError:
            if eof:
                raise
            if len(buf) > JSON_STREAM_CHARS * 256:
                raise RuntimeError(f"{label}: JSON element exceeded streaming buffer")
            continue
        yield idx, obj
        idx += 1
        buf = buf[end:]


def zip_member_sha(zf: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    h = hashlib.sha256()
    with zf.open(info, "r") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compact(value: Any, depth: int = 0) -> Any:
    if depth > 4: return "…"
    if isinstance(value, dict): return {str(k)[:80]: compact(v, depth + 1) for k, v in list(value.items())[:80]}
    if isinstance(value, list): return [compact(v, depth + 1) for v in value[:20]]
    if isinstance(value, str): return value[:4000]
    if isinstance(value, (int, float, bool)) or value is None: return value
    return str(value)[:500]


def first_text(d: dict[str, Any], keys: Iterable[str], default: str = "") -> str:
    for k in keys:
        if k in d and d[k] not in (None, "", [], {}):
            v = d[k]
            if isinstance(v, dict): return json.dumps(compact(v), ensure_ascii=False)[:1000]
            if isinstance(v, list): return json.dumps(compact(v), ensure_ascii=False)[:1000]
            return str(v)[:1000]
    return default


def event_from_dict(d: dict[str, Any], path: str, seq: int) -> Event | None:
    hit_key, hit_dt, raw_v = "", None, None
    for k, v in d.items():
        lk = str(k).lower().replace("-", "_")
        if lk in TS_KEYS or lk.endswith("_at") or "timestamp" in lk:
            hit_dt = parse_ts(v)
            if hit_dt: hit_key, raw_v = str(k), v; break
    if not hit_dt: return None
    label = source_label(path, d)
    entity = first_text(d, ENTITY_KEYS, label)
    action = first_text(d, ACTION_KEYS, "activity")
    content = first_text(d, CONTENT_KEYS, "")
    raw = {"source_label": label, "timestamp_key": hit_key, "timestamp_raw": str(raw_v)[:200], "entity": entity, "action": action, "path": path, "record": compact(d)}
    pid = hashlib.sha256(f"{path}|{seq}|{hit_dt.isoformat()}|{entity}|{action}|{content[:500]}".encode()).hexdigest()
    return Event(hit_dt, label, kind_for(label), entity, action, content, raw, path[:900], pid, seq)


def walk_json(obj: Any, path: str, seq0: int = 0) -> Iterator[Event]:
    stack = [(obj, seq0)]
    seq = seq0
    while stack:
        cur, _ = stack.pop()
        if isinstance(cur, dict):
            ev = event_from_dict(cur, path, seq); seq += 1
            if ev: yield ev
            for v in cur.values():
                if isinstance(v, (dict, list)): stack.append((v, seq))
        elif isinstance(cur, list):
            for v in reversed(cur):
                if isinstance(v, (dict, list)): stack.append((v, seq))



def apply_case_link_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
        CREATE EXTENSION IF NOT EXISTS pgcrypto;
        ALTER TABLE lucidota_commdump.export_object ADD COLUMN IF NOT EXISTS case_uuid uuid REFERENCES lucidota_investigation.case_file(case_uuid);
        ALTER TABLE lucidota_commdump.export_object ADD COLUMN IF NOT EXISTS case_key text NOT NULL DEFAULT '';
        ALTER TABLE lucidota_commdump.export_object ADD COLUMN IF NOT EXISTS file_uuid uuid REFERENCES lucidota_investigation.artifact(artifact_uuid);
        ALTER TABLE lucidota_commdump.export_object ADD COLUMN IF NOT EXISTS case_artifact_uuid uuid REFERENCES lucidota_investigation.case_artifact(case_artifact_uuid);
        ALTER TABLE lucidota_commdump.export_object ADD COLUMN IF NOT EXISTS evidence_id text NOT NULL DEFAULT '';
        ALTER TABLE lucidota_commdump.thread ADD COLUMN IF NOT EXISTS case_uuid uuid REFERENCES lucidota_investigation.case_file(case_uuid);
        ALTER TABLE lucidota_commdump.thread ADD COLUMN IF NOT EXISTS case_key text NOT NULL DEFAULT '';
        ALTER TABLE lucidota_commdump.message ADD COLUMN IF NOT EXISTS case_uuid uuid REFERENCES lucidota_investigation.case_file(case_uuid);
        ALTER TABLE lucidota_commdump.message ADD COLUMN IF NOT EXISTS case_key text NOT NULL DEFAULT '';
        ALTER TABLE lucidota_commdump.message ADD COLUMN IF NOT EXISTS file_uuid uuid REFERENCES lucidota_investigation.artifact(artifact_uuid);
        ALTER TABLE lucidota_commdump.message ADD COLUMN IF NOT EXISTS case_artifact_uuid uuid REFERENCES lucidota_investigation.case_artifact(case_artifact_uuid);
        ALTER TABLE lucidota_commdump.message ADD COLUMN IF NOT EXISTS evidence_id text NOT NULL DEFAULT '';
        CREATE INDEX IF NOT EXISTS commdump_message_case_time_idx ON lucidota_commdump.message(case_key, occurred_at);
        CREATE INDEX IF NOT EXISTS commdump_export_case_idx ON lucidota_commdump.export_object(case_key, source_kind);
        """)
    conn.commit()


def file_kind_for(path: str) -> str:
    ext = Path(path.split('!', 1)[-1]).suffix.lower()
    if ext in {'.zip', '.tar', '.gz', '.7z', '.rar'}: return 'archive'
    if ext in {'.json', '.jsonl', '.csv', '.txt', '.md', '.html', '.xml', '.log'}: return 'text'
    if ext in {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.heic'}: return 'image'
    if ext in {'.pdf', '.doc', '.docx', '.odt', '.xls', '.xlsx'}: return 'document'
    if ext in {'.mp3', '.m4a', '.wav', '.ogg'}: return 'audio'
    if ext in {'.mp4', '.mov', '.mkv', '.webm'}: return 'video'
    return 'binary'


def ensure_case(cur, case_key: str, title: str) -> str:
    if not re.fullmatch(r"(?:CASE-[0-9]{8}-[A-Z0-9][A-Z0-9_-]{1,80}|KE26-[0-9]{5})", case_key):
        raise ValueError(f"bad case key: {case_key}; expected KE26-##### (or legacy one-off CASE-YYYYMMDD-SLUG)")
    folder = f"03_VAULT/cases/{case_key}"
    cur.execute("""
      INSERT INTO lucidota_investigation.case_file(case_key,title,status,folder_relative_path,detail)
      VALUES(%s,%s,'active',%s,%s)
      ON CONFLICT(case_key) DO UPDATE SET title=EXCLUDED.title, status='active', updated_at=now(), detail=lucidota_investigation.case_file.detail || EXCLUDED.detail
      RETURNING case_uuid
    """, (case_key, title, folder, Json({"source":"hypertimeline_engine", "rule":"every evidence file has case_uuid/file_uuid/evidence_id"})))
    return str(cur.fetchone()[0])


def ensure_file_evidence(cur, case_uuid: str, case_key: str, source_path: str, sha: str, size: int, label: str) -> tuple[str, str, str]:
    name = Path(source_path.split('!', 1)[-1]).name[:500]
    mime = mimetypes.guess_type(name)[0] or ''
    evidence_id = f"{case_key}-EVID-{sha[:12].upper()}"
    cur.execute("""
      INSERT INTO lucidota_investigation.artifact(sha256,cas_uri,locked_relative_path,original_path,original_name,mime,file_kind,size_bytes,title,metadata_json)
      VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
      ON CONFLICT(sha256) DO UPDATE SET original_path=COALESCE(NULLIF(lucidota_investigation.artifact.original_path,''), EXCLUDED.original_path), updated_at=now(), metadata_json=lucidota_investigation.artifact.metadata_json || EXCLUDED.metadata_json
      RETURNING artifact_uuid
    """, (sha, f"cas://sha256/{sha}", source_path[:900], source_path[:1500], name, mime, file_kind_for(source_path), size, name, Json({"source_label": label, "hypertimeline": True})))
    file_uuid = str(cur.fetchone()[0])
    cur.execute("""
      INSERT INTO lucidota_investigation.case_artifact(case_uuid,artifact_uuid,evidence_id,artifact_title,custody,status)
      VALUES(%s,%s,%s,%s,%s,'indexed')
      ON CONFLICT(case_uuid,artifact_uuid) DO UPDATE SET status='indexed', custody=lucidota_investigation.case_artifact.custody || EXCLUDED.custody, updated_at=now()
      RETURNING case_artifact_uuid, evidence_id
    """, (case_uuid, file_uuid, evidence_id, name, Json({"original_path": source_path, "sha256": sha, "ingested_by": "hypertimeline_engine"})))
    case_artifact_uuid, stable_evidence_id = cur.fetchone()
    return file_uuid, str(case_artifact_uuid), stable_evidence_id

class HypertimelineDBIngester:
    def __init__(self, dsn: str = DB_DSN, batch_size: int = HYPERTIMELINE_BATCH_SIZE, case_key: str = DEFAULT_CASE_KEY, case_title: str = "ABSURD Hypertimeline Intake"):
        self.pool = SimpleConnectionPool(1, HYPERTIMELINE_POOL_MAX, dsn)
        self.batch_size = batch_size
        self.case_key = case_key.upper()
        self.case_title = case_title
        self.stats = IngestStats()
        conn = self.conn()
        try: apply_case_link_schema(conn)
        finally: self.put(conn)

    def close(self) -> None: self.pool.closeall()

    def conn(self): return self.pool.getconn()
    def put(self, conn) -> None: self.pool.putconn(conn)

    def upsert_export_thread(self, cur, source_path: str, sha: str, size: int, kind: str, label: str) -> tuple[str, str, str, str, str, str]:
        case_uuid = ensure_case(cur, self.case_key, self.case_title)
        file_uuid, case_artifact_uuid, evidence_id = ensure_file_evidence(cur, case_uuid, self.case_key, source_path, sha, size, label)
        cur.execute("""
          INSERT INTO lucidota_commdump.export_object(source_kind,source_path,source_sha256,size_bytes,mime,status,detail,case_uuid,case_key,file_uuid,case_artifact_uuid,evidence_id)
          VALUES(%s,%s,%s,%s,%s,'running',%s,%s,%s,%s,%s,%s)
          ON CONFLICT(source_sha256,source_kind) DO UPDATE SET status='running', updated_at=now(), case_uuid=EXCLUDED.case_uuid, case_key=EXCLUDED.case_key, file_uuid=EXCLUDED.file_uuid, case_artifact_uuid=EXCLUDED.case_artifact_uuid, evidence_id=EXCLUDED.evidence_id, detail=lucidota_commdump.export_object.detail || EXCLUDED.detail
          RETURNING export_uuid
        """, (kind, source_path, sha, size, mimetypes.guess_type(source_path)[0] or "", Json({"hypertimeline": True, "source_label": label, "evidence_id": evidence_id}), case_uuid, self.case_key, file_uuid, case_artifact_uuid, evidence_id))
        export_uuid = cur.fetchone()[0]
        cur.execute("""
          INSERT INTO lucidota_commdump.thread(export_uuid,source_kind,provider_thread_id,title,source_member,detail,case_uuid,case_key)
          VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
          ON CONFLICT(export_uuid,source_member,provider_thread_id) DO UPDATE SET case_uuid=EXCLUDED.case_uuid, case_key=EXCLUDED.case_key, detail=lucidota_commdump.thread.detail || EXCLUDED.detail
          RETURNING thread_uuid
        """, (export_uuid, kind, hashlib.sha256(source_path.encode()).hexdigest()[:32], Path(source_path).name[:500], source_path[:900], Json({"source_label": label, "evidence_id": evidence_id}), case_uuid, self.case_key))
        return str(export_uuid), str(cur.fetchone()[0]), case_uuid, file_uuid, case_artifact_uuid, evidence_id

    def insert_events(self, cur, export_uuid: str, thread_uuid: str, case_uuid: str, file_uuid: str, case_artifact_uuid: str, evidence_id: str, events: list[Event]) -> int:
        if not events: return 0
        rows = [(thread_uuid, export_uuid, e.source_kind, e.provider_id, e.entity, e.occurred_at, e.raw["timestamp_raw"], e.raw["timestamp_key"], 9200, e.sequence, e.action, e.content, hashlib.sha256(e.content.encode()).hexdigest() if e.content else "", e.source_label, Json(e.raw | {"case_key": self.case_key, "evidence_id": evidence_id, "file_uuid": file_uuid}), case_uuid, self.case_key, file_uuid, case_artifact_uuid, evidence_id) for e in events]
        execute_values(cur, """
          INSERT INTO lucidota_commdump.message(thread_uuid,export_uuid,source_kind,provider_message_id,sender,occurred_at,occurred_at_raw,time_source,time_confidence_bps,sequence_index,subject,content_text,content_sha256,content_kind,raw,case_uuid,case_key,file_uuid,case_artifact_uuid,evidence_id)
          VALUES %s ON CONFLICT(thread_uuid,provider_message_id,sequence_index) DO NOTHING
        """, rows, page_size=self.batch_size)
        return cur.rowcount if cur.rowcount >= 0 else len(rows)

    def ingest_events(self, source_path: str, sha: str, size: int, events: Iterator[Event]) -> int:
        conn = self.conn(); total = 0; label = source_label(source_path); kind = kind_for(label)
        export_uuid = thread_uuid = case_uuid = file_uuid = case_artifact_uuid = evidence_id = ""

        def flush_batch(batch: list[Event]) -> int:
            if not batch:
                return 0
            with conn.cursor() as cur:
                inserted = self.insert_events(cur, export_uuid, thread_uuid, case_uuid, file_uuid, case_artifact_uuid, evidence_id, batch)
            conn.commit()
            return inserted

        try:
            with conn.cursor() as cur:
                export_uuid, thread_uuid, case_uuid, file_uuid, case_artifact_uuid, evidence_id = self.upsert_export_thread(cur, source_path, sha, size, kind, label)
            conn.commit()
            batch: list[Event] = []
            for ev in events:
                batch.append(ev); self.stats.events_found += 1; self.stats.sources[ev.source_label] = self.stats.sources.get(ev.source_label, 0) + 1
                if len(batch) >= self.batch_size:
                    total += flush_batch(batch); batch.clear()
            total += flush_batch(batch)
            with conn.cursor() as cur:
                cur.execute("UPDATE lucidota_commdump.export_object SET status='succeeded', message_count=%s, latest_message_at=(SELECT max(occurred_at) FROM lucidota_commdump.message WHERE export_uuid=%s), earliest_message_at=(SELECT min(occurred_at) FROM lucidota_commdump.message WHERE export_uuid=%s) WHERE export_uuid=%s", (total, export_uuid, export_uuid, export_uuid))
            conn.commit()
            self.stats.events_inserted += total; return total
        except Exception as e:
            conn.rollback(); self.stats.files_failed += 1; self.stats.errors.append(f"{source_path}: {e}"); worker_error(f"ROLLBACK {source_path}: {e}")
            if export_uuid:
                try:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE lucidota_commdump.export_object SET status='failed', detail=detail || %s WHERE export_uuid=%s", (Json({"error": str(e), "partial_inserted": total}), export_uuid))
                    conn.commit()
                except Exception as mark_exc:
                    conn.rollback(); worker_error(f"FAILED_MARK {source_path}: {mark_exc}")
            return total
        finally:
            self.put(conn)

    def parse_file(self, path: Path) -> Iterator[Event]:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        try: obj = json.loads(line)
                        except json.JSONDecodeError as e: log(f"bad jsonl {path}:{i+1}: {e}"); continue
                        yield from walk_json(obj, str(path), i)
        elif suffix == ".json":
            if path.stat().st_size >= JSON_STREAM_THRESHOLD_BYTES:
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as f:
                        for i, obj in iter_json_array_stream(lambda: f.read(JSON_STREAM_CHARS), str(path)):
                            yield from walk_json(obj, str(path), i)
                    return
                except ValueError as e:
                    worker_error(f"large-json-nonarray {path}: {e}")
                    if os.getenv("LUCIDOTA_ALLOW_LARGE_JSON_FULL_LOAD", "").lower() not in {"1", "true", "yes", "on"}:
                        return
            yield from walk_json(json.loads(path.read_text(encoding="utf-8", errors="replace")), str(path), 0)
        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                for i, row in enumerate(csv.DictReader(f)):
                    ev = event_from_dict(dict(row), str(path), i)
                    if ev: yield ev

    def ingest_file(self, path: Path) -> int:
        self.stats.files_seen += 1
        try:
            size = path.stat().st_size
            if path.suffix.lower() == ".zip": return self.ingest_zip(path)
            if path.suffix.lower() not in {".json", ".jsonl", ".csv"}: return 0
            inserted = self.ingest_events(str(path), sha256_file(path), size, self.parse_file(path))
            self.stats.files_loaded += 1 if inserted else 0
            return inserted
        except Exception as e:
            self.stats.files_failed += 1; self.stats.errors.append(f"{path}: {e}"); worker_error(f"FAILED {path}: {e}"); return 0

    def ingest_zip(self, path: Path, max_member_mb: int = int(os.getenv("LUCIDOTA_ZIP_MAX_MEMBER_MB", "0"))) -> int:
        inserted = 0
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir() or Path(info.filename).suffix.lower() not in {".json", ".jsonl", ".csv"}: continue
                if max_member_mb and info.file_size > max_member_mb * 1024 * 1024: worker_error(f"skip large zip member {path}!{info.filename}"); continue
                member_path = f"{path}!{info.filename}"; suffix = Path(info.filename).suffix.lower(); member_sha = zip_member_sha(z, info)
                def events() -> Iterator[Event]:
                    if suffix == ".jsonl":
                        with z.open(info, "r") as raw:
                            for i, line in enumerate(io.TextIOWrapper(raw, encoding="utf-8", errors="replace")):
                                if line.strip():
                                    try: yield from walk_json(json.loads(line), member_path, i)
                                    except json.JSONDecodeError as e: worker_error(f"bad zip jsonl {member_path}:{i+1}: {e}")
                    elif suffix == ".json":
                        with z.open(info, "r") as raw:
                            txt = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
                            if info.file_size >= JSON_STREAM_THRESHOLD_BYTES:
                                try:
                                    for i, obj in iter_json_array_stream(lambda: txt.read(JSON_STREAM_CHARS), member_path):
                                        yield from walk_json(obj, member_path, i)
                                    return
                                except ValueError as e:
                                    worker_error(f"large-zip-json-nonarray {member_path}: {e}")
                                    if os.getenv("LUCIDOTA_ALLOW_LARGE_JSON_FULL_LOAD", "").lower() not in {"1", "true", "yes", "on"}:
                                        return
                            txt.seek(0)
                            yield from walk_json(json.loads(txt.read()), member_path, 0)
                    else:
                        with z.open(info, "r") as raw:
                            for i, row in enumerate(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))):
                                ev = event_from_dict(dict(row), member_path, i)
                                if ev: yield ev
                inserted += self.ingest_events(member_path, member_sha, info.file_size, events())
        return inserted

    def discover(self, roots: list[Path]) -> Iterator[Path]:
        skip_dirs = {".git", ".venv", "node_modules", "__pycache__", "target", "site-packages"}
        for root in roots:
            if root.is_file(): yield root; continue
            if not root.exists(): continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in skip_dirs and d != "log"]
                for name in filenames:
                    p = Path(dirpath) / name
                    if p.suffix.lower() in {".json", ".jsonl", ".csv", ".zip"}: yield p

    def ingest_roots(self, roots: list[Path]) -> IngestStats:
        for path in self.discover(roots): self.ingest_file(path)
        return self.stats


def curve_for(times: list[datetime], bins: int = 48) -> np.ndarray:
    return circadian_activity_curve(times, bins=bins)


def pick_two_sources(dsn: str) -> tuple[str, list[datetime], str, list[datetime]]:
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("""
              SELECT coalesce(raw->>'source_label', source_kind) src, count(*)
              FROM lucidota_commdump.message WHERE occurred_at IS NOT NULL
              GROUP BY 1 HAVING count(*) >= 30 ORDER BY count(*) DESC LIMIT 8
            """)
            sources = [r[0] for r in cur.fetchall()]
            preferred = [s for s in ("chatgpt","codex","claude","cell_phone","whatsapp","facebook","email") if s in sources]
            chosen = (preferred + [s for s in sources if s not in preferred])[:2]
            if len(chosen) < 2: raise RuntimeError("need at least two populated timestamp sources")
            out = []
            for s in chosen:
                cur.execute("SELECT occurred_at FROM lucidota_commdump.message WHERE coalesce(raw->>'source_label', source_kind)=%s AND occurred_at IS NOT NULL ORDER BY occurred_at", (s,))
                out.append((s, [r[0].astimezone(timezone.utc) for r in cur.fetchall()]))
            return out[0][0], out[0][1], out[1][0], out[1][1]
    finally:
        conn.close()


def render_engine(a_name: str, a_times: list[datetime], b_name: str, b_times: list[datetime], sweep: SweepResult, ingest: IngestStats) -> None:
    metrics = circadian_match_metrics(a_times, b_times, bins=48)
    a, b = metrics["curve_a"], metrics["curve_b"]
    dead_a, dead_b = metrics["dead_mask_a"], metrics["dead_mask_b"]
    corr, dead, score = float(metrics["pearson_r"]), float(metrics["dead_zone_overlap"]), float(metrics["score"])
    chars = " ▁▂▃▄▅▆▇█"
    table = Table(title="ABSURD Hypertimeline Circadian Overlay (UTC, 48 bins)", box=box.SIMPLE_HEAVY)
    table.add_column("UTC", justify="right"); table.add_column(a_name, style="cyan"); table.add_column(b_name, style="yellow"); table.add_column("Ω", justify="center")
    for i in range(48):
        ia, ib = int(round(a[i] * 8)), int(round(b[i] * 8))
        omega = Text("×", "bold red") if dead_a[i] and dead_b[i] else Text(chars[ia] + chars[ib], "magenta")
        table.add_row(f"{i//2:02d}:{'30' if i%2 else '00'}", "█"*ia + "░"*(8-ia), "█"*ib + "░"*(8-ib), omega)
    console.print(table)
    verdict = "HIGH SAME-ENTITY CIRCADIAN CONSISTENCY" if score >= 85 and dead >= .65 else "CIRCADIAN SIGNAL RECORDED; MORE SOURCES IMPROVE POWER"
    panel = f"System Sweep Complete: {fmt_bytes(sweep.freed_bytes)} Freed & DB Populated\nCase-linked evidence UUIDs active. Files={ingest.files_seen} loaded={ingest.files_loaded} failed={ingest.files_failed} inserted={ingest.events_inserted}\nSources={ingest.sources}\nMatch={score:.1f}%  Dead-Zone={dead*100:.1f}%  Pearson r={corr:+.3f}\nVERDICT: {verdict}"
    console.print(Panel(Text(panel, style="bold white"), border_style="bright_green", box=box.DOUBLE))


def default_inputs() -> list[Path]:
    paths = [ROOT / "KRAMPUSCHEWING"]
    for p in [HOME / ".codex" / "history.jsonl", HOME / "Downloads" / "Lucidota" / "Lucidota" / "Evidence" / "EXTRACTED_WA_779-5230"]:
        if p.exists(): paths.append(p)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="LUCIDOTA ABSURD hypertimeline: dedupe, ingest, circadian test.")
    ap.add_argument("--input", action="append", type=Path, help="file/dir/zip to ingest; repeatable")
    ap.add_argument("--dsn", default=DB_DSN)
    ap.add_argument("--case-key", default=DEFAULT_CASE_KEY, help="CASE-YYYYMMDD-SLUG assigned to every imported evidence file")
    ap.add_argument("--case-title", default="ABSURD Hypertimeline Intake")
    ap.add_argument("--batch-size", type=int, default=HYPERTIMELINE_BATCH_SIZE, help="Postgres execute_values commit batch size")
    ap.add_argument("--dry-gc", action="store_true", help="scan duplicates without hardlinking")
    ap.add_argument("--skip-gc", action="store_true")
    ap.add_argument("--skip-ingest", action="store_true")
    args = ap.parse_args()
    console.print(Panel.fit("[bold]LUCIDOTA ABSURD HYPERTIMELINE OPS[/bold]", border_style="bright_blue"))
    sweep = SweepResult(gitignore_added=ensure_gitignore()) if args.skip_gc else hardlink_dedupe(args.dry_gc)
    ingest = IngestStats()
    if not args.skip_ingest:
        ing = HypertimelineDBIngester(args.dsn, batch_size=args.batch_size, case_key=args.case_key, case_title=args.case_title)
        try: ingest = ing.ingest_roots(args.input or default_inputs())
        finally: ing.close()
    try:
        a_name, a_times, b_name, b_times = pick_two_sources(args.dsn)
        render_engine(a_name, a_times, b_name, b_times, sweep, ingest)
    except Exception as e:
        worker_error(f"ENGINE_TEST_UNAVAILABLE: {e}")
        console.print(Panel(f"Sweep={fmt_bytes(sweep.freed_bytes)} freed; Ingested={ingest.events_inserted}; engine test unavailable: {e}", border_style="yellow"))
    if sweep.errors or ingest.errors:
        err = Table(title="Non-fatal errors", box=box.ROUNDED); err.add_column("error")
        for e in (sweep.errors + ingest.errors)[:20]: err.add_row(str(e)[:180])
        console.print(err)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
