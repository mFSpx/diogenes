#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import mimetypes
import os
import re
import shutil
import socket
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import psycopg2
from psycopg2.extras import Json, execute_values

from db_core import DEFAULT_DSN, DurableConnection

SPINE_SCHEMA_SQL = r'''
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_spine;
CREATE TABLE IF NOT EXISTS lucidota_spine.ingest_run (
    run_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    root_path text NOT NULL,
    status text NOT NULL CHECK (status IN ('running','succeeded','failed')),
    host_name text NOT NULL,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    files_seen integer NOT NULL DEFAULT 0,
    files_written integer NOT NULL DEFAULT 0,
    files_failed integer NOT NULL DEFAULT 0,
    components_written integer NOT NULL DEFAULT 0,
    entities_written integer NOT NULL DEFAULT 0,
    concepts_written integer NOT NULL DEFAULT 0,
    bytes_seen bigint NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS lucidota_spine.cas_object (
    cas_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    sha256_hash text UNIQUE NOT NULL CHECK (sha256_hash ~ '^[0-9a-f]{64}$'),
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    first_seen_path text NOT NULL,
    mime_type text NOT NULL DEFAULT '',
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS lucidota_spine.file_object (
    file_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    cas_uuid uuid NOT NULL REFERENCES lucidota_spine.cas_object(cas_uuid),
    absolute_path text UNIQUE NOT NULL,
    relative_path text NOT NULL,
    file_size_bytes bigint NOT NULL CHECK (file_size_bytes >= 0),
    mime_type text NOT NULL DEFAULT '',
    sha256_hash text NOT NULL CHECK (sha256_hash ~ '^[0-9a-f]{64}$'),
    os_created_at timestamptz,
    os_modified_at timestamptz,
    content_regex_date date,
    best_estimated_date timestamptz,
    status text NOT NULL CHECK (status IN ('indexed','deferred','error')),
    detail jsonb NOT NULL DEFAULT '{}',
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS lucidota_spine.file_occurrence (
    occurrence_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_uuid uuid NOT NULL REFERENCES lucidota_spine.ingest_run(run_uuid),
    file_uuid uuid NOT NULL REFERENCES lucidota_spine.file_object(file_uuid),
    absolute_path text NOT NULL,
    observed_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(run_uuid, absolute_path)
);
CREATE TABLE IF NOT EXISTS lucidota_spine.component (
    component_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    file_uuid uuid NOT NULL REFERENCES lucidota_spine.file_object(file_uuid),
    component_index integer NOT NULL,
    component_kind text NOT NULL,
    title text NOT NULL DEFAULT '',
    content text NOT NULL DEFAULT '',
    sha256_hash text NOT NULL CHECK (sha256_hash ~ '^[0-9a-f]{64}$'),
    start_line integer NOT NULL DEFAULT 0,
    end_line integer NOT NULL DEFAULT 0,
    token_count integer NOT NULL DEFAULT 0,
    detail jsonb NOT NULL DEFAULT '{}',
    UNIQUE(file_uuid, component_index)
);
CREATE TABLE IF NOT EXISTS lucidota_spine.entity (
    entity_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    component_uuid uuid NOT NULL REFERENCES lucidota_spine.component(component_uuid),
    entity_kind text NOT NULL,
    value text NOT NULL,
    normalized_value text NOT NULL,
    confidence_bps integer NOT NULL CHECK (confidence_bps BETWEEN 0 AND 10000),
    source text NOT NULL,
    context text NOT NULL DEFAULT '',
    UNIQUE(component_uuid, entity_kind, normalized_value, source)
);
CREATE TABLE IF NOT EXISTS lucidota_spine.concept (
    concept_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    component_uuid uuid NOT NULL REFERENCES lucidota_spine.component(component_uuid),
    concept_kind text NOT NULL,
    value text NOT NULL,
    normalized_value text NOT NULL,
    confidence_bps integer NOT NULL CHECK (confidence_bps BETWEEN 0 AND 10000),
    source text NOT NULL,
    UNIQUE(component_uuid, concept_kind, normalized_value, source)
);
CREATE TABLE IF NOT EXISTS lucidota_spine.sticker_signal (
    signal_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    component_uuid uuid NOT NULL REFERENCES lucidota_spine.component(component_uuid),
    uppercase_ratio numeric NOT NULL,
    ellipsis_density numeric NOT NULL,
    punctuation_velocity numeric NOT NULL,
    first_person_gravity numeric NOT NULL,
    directive_ratio numeric NOT NULL,
    ledger_density numeric NOT NULL,
    raw_features jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(component_uuid)
);
CREATE TABLE IF NOT EXISTS lucidota_spine.graph_node (
    node_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    node_kind text NOT NULL,
    stable_uri text UNIQUE NOT NULL,
    title text NOT NULL,
    detail jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS lucidota_spine.graph_edge (
    edge_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_node_uuid uuid NOT NULL REFERENCES lucidota_spine.graph_node(node_uuid),
    target_node_uuid uuid NOT NULL REFERENCES lucidota_spine.graph_node(node_uuid),
    edge_kind text NOT NULL,
    detail jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(source_node_uuid, target_node_uuid, edge_kind)
);
CREATE TABLE IF NOT EXISTS lucidota_spine.ingest_error (
    error_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_uuid uuid,
    absolute_path text NOT NULL,
    error_kind text NOT NULL,
    error_text text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
'''

ISO_8601_RE = re.compile(r"\b(20\d{2}-[01]\d-[0-3]\d(?:[T ][0-2]\d:[0-5]\d(?::[0-5]\d(?:\.\d{1,6})?)?(?:Z|[+-][0-2]\d:? [0-5]\d|[+-][0-2]\d:?[0-5]\d)?)?)\b".replace(" ", ""))
SLASH_DATE_RE = re.compile(r"\b((?:20|19)\d{2}/[01]?\d/[0-3]?\d)\b")
UNIX_TS_RE = re.compile(r"\b(1[5-9]\d{8}|2[0-1]\d{8}|1[5-9]\d{11}|2[0-1]\d{11})\b")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL_RE = re.compile(r"\bhttps?://[^\s<>'\"]+", re.I)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)")
SHA_RE = re.compile(r"\b[0-9a-f]{64}\b", re.I)
MONEY_RE = re.compile(r"(?<!\w)\$\s?\d[\d,]*(?:\.\d{2})?\b")
ACTION_RE = re.compile(r"\b(?:GOAL|OBJECTIVE|MISSION|NEXT|FIXME|ACTION ITEM)\b\s*[:\-]?\s*(.{3,220})", re.I)
IDEA_RE = re.compile(r"\b(?:IDEA|HUNCH|MAYBE|PROPOSAL)\b\s*[:\-]?\s*(.{3,220})", re.I)
TEXT_SUFFIXES = {".txt", ".md", ".csv", ".json", ".jsonl", ".yaml", ".yml", ".html", ".htm", ".py", ".sh", ".sql", ".log"}
MAX_TEXT_BYTES = int(os.environ.get("LUCIDOTA_SPINE_MAX_TEXT_BYTES", str(8 * 1024 * 1024)))
CHUNK_LINES = int(os.environ.get("LUCIDOTA_SPINE_CHUNK_LINES", "220"))


@dataclass
class DateHit:
    value: str
    source: str
    timestamp: dt.datetime


@dataclass
class ComponentDraft:
    index: int
    kind: str
    title: str
    content: str
    start_line: int
    end_line: int


@dataclass
class FileWork:
    absolute_path: str
    relative_path: str
    size_bytes: int
    mime_type: str
    sha256_hash: str
    os_created_at: dt.datetime
    os_modified_at: dt.datetime
    content_regex_date: dt.date | None
    best_estimated_date: dt.datetime
    text: str
    status: str
    detail: dict[str, Any]


class Spine:
    def __init__(self, root: Path, dsn: str = DEFAULT_DSN, dashboard_json: Path | None = None) -> None:
        self.root = root.resolve()
        self.dsn = dsn
        self.dashboard_json = dashboard_json or Path("05_OUTPUTS/spine_dashboard.json")
        self.run_uuid = ""
        self.counts = {"files_seen": 0, "files_written": 0, "files_failed": 0, "components_written": 0, "entities_written": 0, "concepts_written": 0, "bytes_seen": 0}

    @staticmethod
    def clean_text(value: str) -> str:
        return value.replace("\x00", "�")

    @staticmethod
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def utc_from_ts(value: float) -> dt.datetime:
        return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc)

    def walk_files(self) -> Iterator[Path]:
        stack = [self.root]
        while stack:
            current = stack.pop()
            try:
                if current.is_file():
                    yield current
                elif current.is_dir():
                    with os.scandir(current) as scan:
                        for entry in scan:
                            stack.append(Path(entry.path))
            except OSError as exc:
                self.record_error(None, str(current), exc)

    def read_text_sample(self, path: Path) -> tuple[str, dict[str, Any]]:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            return "", {"text_status": "binary_or_unsupported"}
        try:
            data = path.read_bytes()[:MAX_TEXT_BYTES]
        except OSError as exc:
            return "", {"text_status": "read_error", "error": str(exc)}
        for enc in ("utf-8", "utf-16", "latin-1"):
            try:
                return self.clean_text(data.decode(enc, errors="replace")), {"text_status": "decoded", "encoding": enc, "nul_sanitized": b"\x00" in data}
            except UnicodeError:
                continue
        return self.clean_text(data.decode("utf-8", errors="replace")), {"text_status": "decoded", "encoding": "utf-8", "nul_sanitized": b"\x00" in data}

    def extract_dates(self, text: str) -> list[DateHit]:
        hits: list[DateHit] = []
        for match in ISO_8601_RE.finditer(text[:500_000]):
            raw = match.group(1)
            try:
                value = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if value.tzinfo is None:
                    value = value.replace(tzinfo=dt.timezone.utc)
                hits.append(DateHit(raw, "regex_iso_8601", value.astimezone(dt.timezone.utc)))
            except ValueError:
                continue
        for match in SLASH_DATE_RE.finditer(text[:500_000]):
            raw = match.group(1)
            try:
                value = dt.datetime.strptime(raw, "%Y/%m/%d").replace(tzinfo=dt.timezone.utc)
                hits.append(DateHit(raw, "regex_yyyy_slash_mm_slash_dd", value))
            except ValueError:
                continue
        for match in UNIX_TS_RE.finditer(text[:500_000]):
            raw = match.group(1)
            try:
                numeric = int(raw)
                if numeric > 9_999_999_999:
                    numeric = numeric // 1000
                value = dt.datetime.fromtimestamp(numeric, tz=dt.timezone.utc)
                if 2017 <= value.year <= 2036:
                    hits.append(DateHit(raw, "regex_unix_timestamp", value))
            except (OverflowError, ValueError):
                continue
        return hits

    def build_file_work(self, path: Path) -> FileWork:
        st = path.stat()
        text, detail = self.read_text_sample(path)
        hits = self.extract_dates(text)
        os_created = self.utc_from_ts(getattr(st, "st_birthtime", st.st_ctime))
        os_modified = self.utc_from_ts(st.st_mtime)
        content_date = hits[0].timestamp.date() if hits else None
        best = hits[0].timestamp if hits else min(os_created, os_modified)
        detail["date_hits"] = [
            {"value": hit.value, "source": hit.source, "timestamp": hit.timestamp.isoformat()}
            for hit in hits[:20]
        ]
        return FileWork(
            absolute_path=str(path.resolve()),
            relative_path=str(path.resolve().relative_to(self.root)) if path.resolve().is_relative_to(self.root) else path.name,
            size_bytes=int(st.st_size),
            mime_type=mimetypes.guess_type(str(path))[0] or "application/octet-stream",
            sha256_hash=self.sha256_file(path),
            os_created_at=os_created,
            os_modified_at=os_modified,
            content_regex_date=content_date,
            best_estimated_date=best,
            text=text,
            status="indexed" if text else "deferred",
            detail=detail,
        )

    def chunk_text(self, text: str) -> list[ComponentDraft]:
        if not text:
            return []
        lines = text.splitlines()
        out: list[ComponentDraft] = []
        for start in range(0, len(lines), CHUNK_LINES):
            chunk = "\n".join(lines[start:start + CHUNK_LINES]).strip()
            if not chunk:
                continue
            index = len(out)
            title = lines[start].strip()[:160] if lines[start:start + 1] else f"chunk {index}"
            out.append(ComponentDraft(index=index, kind="text_chunk", title=title, content=chunk, start_line=start + 1, end_line=min(start + CHUNK_LINES, len(lines))))
        return out

    def extract_entities(self, component: ComponentDraft) -> list[dict[str, Any]]:
        rules = [
            ("email", EMAIL_RE, 7600),
            ("url", URL_RE, 7400),
            ("ip", IP_RE, 6200),
            ("phone", PHONE_RE, 6500),
            ("sha256", SHA_RE, 8000),
            ("money", MONEY_RE, 5600),
            ("date", ISO_8601_RE, 6900),
            ("date", SLASH_DATE_RE, 6100),
        ]
        entities: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for kind, regex, confidence in rules:
            for match in regex.finditer(component.content):
                value = match.group(0).strip()
                normalized = re.sub(r"\s+", " ", value.lower())
                key = (kind, normalized)
                if key in seen:
                    continue
                seen.add(key)
                entities.append({
                    "entity_kind": kind,
                    "value": value[:500],
                    "normalized_value": normalized[:500],
                    "confidence_bps": confidence,
                    "source": f"regex:{kind}",
                    "context": component.content[max(0, match.start() - 90):match.end() + 90][:500],
                })
        return entities

    def extract_concepts(self, component: ComponentDraft) -> list[dict[str, Any]]:
        concepts: list[dict[str, Any]] = []
        for kind, regex, confidence in (("action_item", ACTION_RE, 6900), ("idea", IDEA_RE, 5200)):
            for match in regex.finditer(component.content):
                value = match.group(1).strip()
                concepts.append({"concept_kind": kind, "value": value[:700], "normalized_value": value.lower()[:700], "confidence_bps": confidence, "source": f"regex:{kind}"})
        lower = component.content.lower()
        keywords = {
            "absurd": "ABSURD workflow",
            "krampus": "KRAMPUS ingestion",
            "korpus": "KORPUS componentization",
            "chrono": "Chrono-Ledger",
            "river": "River ML",
            "graph": "graph promotion",
            "cas": "content addressed storage",
            "dedupe": "dedupe",
        }
        for key, label in keywords.items():
            if key in lower:
                concepts.append({"concept_kind": "keyword", "value": label, "normalized_value": key, "confidence_bps": 5000, "source": "keyword_map"})
        return concepts

    def sticker_features(self, component: ComponentDraft) -> dict[str, float]:
        text = component.content
        chars = max(1, len(text))
        words = re.findall(r"\b\w+\b", text.lower())
        word_count = max(1, len(words))
        imperatives = {"do", "run", "make", "build", "fix", "write", "execute", "ingest", "scan", "hash", "dedupe"}
        return {
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / chars,
            "ellipsis_density": text.count("…") / chars,
            "punctuation_velocity": (text.count("!") + text.count("?") + text.count(";")) / chars,
            "first_person_gravity": sum(1 for w in words if w in {"i", "me", "my", "mine"}) / word_count,
            "directive_ratio": sum(1 for w in words if w in imperatives) / word_count,
            "ledger_density": (text.count("\n-") + text.count("\n*") + text.count("|")) / max(1, text.count("\n") + 1),
        }

    def init_db(self) -> None:
        with DurableConnection(self.dsn) as db:
            def work(conn):
                with conn.cursor() as cur:
                    cur.execute(SPINE_SCHEMA_SQL)
            db.run_serialized(work)

    def create_run(self) -> str:
        with DurableConnection(self.dsn) as db:
            def work(conn):
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO lucidota_spine.ingest_run(root_path,status,host_name) VALUES (%s,'running',%s) RETURNING run_uuid::text", (str(self.root), socket.gethostname()))
                    return str(cur.fetchone()[0])
            return db.run_serialized(work)

    def cas_upsert(self, cur, work: FileWork) -> str:
        cur.execute("SELECT cas_uuid::text FROM lucidota_spine.cas_object WHERE sha256_hash = %s", (work.sha256_hash,))
        row = cur.fetchone()
        if row:
            return str(row[0])
        cur.execute("INSERT INTO lucidota_spine.cas_object(sha256_hash,size_bytes,first_seen_path,mime_type) VALUES (%s,%s,%s,%s) RETURNING cas_uuid::text", (work.sha256_hash, work.size_bytes, work.absolute_path, work.mime_type))
        return str(cur.fetchone()[0])

    def graph_node(self, cur, kind: str, uri: str, title: str, detail: dict[str, Any]) -> str:
        cur.execute("INSERT INTO lucidota_spine.graph_node(node_kind,stable_uri,title,detail) VALUES (%s,%s,%s,%s) ON CONFLICT(stable_uri) DO UPDATE SET title=EXCLUDED.title, detail=lucidota_spine.graph_node.detail || EXCLUDED.detail RETURNING node_uuid::text", (kind, uri, title[:500], Json(detail)))
        return str(cur.fetchone()[0])

    def graph_edge(self, cur, left: str, right: str, kind: str, detail: dict[str, Any]) -> None:
        cur.execute("INSERT INTO lucidota_spine.graph_edge(source_node_uuid,target_node_uuid,edge_kind,detail) VALUES (%s::uuid,%s::uuid,%s,%s) ON CONFLICT(source_node_uuid,target_node_uuid,edge_kind) DO UPDATE SET detail=lucidota_spine.graph_edge.detail || EXCLUDED.detail", (left, right, kind, Json(detail)))

    def write_file(self, conn, work: FileWork) -> tuple[int, int, int]:
        components = self.chunk_text(work.text)
        with conn.cursor() as cur:
            with DurableConnection(self.dsn).file_savepoint(cur):
                cas_uuid = self.cas_upsert(cur, work)
                cur.execute(
                    """
                    INSERT INTO lucidota_spine.file_object(cas_uuid,absolute_path,relative_path,file_size_bytes,mime_type,sha256_hash,os_created_at,os_modified_at,content_regex_date,best_estimated_date,status,detail)
                    VALUES (%s::uuid,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT(absolute_path) DO UPDATE SET cas_uuid=EXCLUDED.cas_uuid,file_size_bytes=EXCLUDED.file_size_bytes,mime_type=EXCLUDED.mime_type,sha256_hash=EXCLUDED.sha256_hash,os_created_at=EXCLUDED.os_created_at,os_modified_at=EXCLUDED.os_modified_at,content_regex_date=EXCLUDED.content_regex_date,best_estimated_date=EXCLUDED.best_estimated_date,status=EXCLUDED.status,detail=EXCLUDED.detail,updated_at=now()
                    RETURNING file_uuid::text
                    """,
                    (cas_uuid, work.absolute_path, work.relative_path, work.size_bytes, work.mime_type, work.sha256_hash, work.os_created_at, work.os_modified_at, work.content_regex_date, work.best_estimated_date, work.status, Json(work.detail)),
                )
                file_uuid = str(cur.fetchone()[0])
                cur.execute("INSERT INTO lucidota_spine.file_occurrence(run_uuid,file_uuid,absolute_path) VALUES (%s::uuid,%s::uuid,%s) ON CONFLICT(run_uuid, absolute_path) DO NOTHING", (self.run_uuid, file_uuid, work.absolute_path))
                file_node = self.graph_node(cur, "file", f"spine-file://{file_uuid}", Path(work.absolute_path).name, {"sha256_hash": work.sha256_hash, "path": work.absolute_path})
                component_rows = []
                component_payloads = []
                for comp in components:
                    clean_content = self.clean_text(comp.content)
                    component_rows.append((file_uuid, comp.index, comp.kind, comp.title, clean_content, hashlib.sha256(clean_content.encode("utf-8", errors="replace")).hexdigest(), comp.start_line, comp.end_line, len(re.findall(r"\S+", clean_content)), Json({})))
                    component_payloads.append(comp)
                component_ids: list[str] = []
                if component_rows:
                    execute_values(cur, """
                        INSERT INTO lucidota_spine.component(file_uuid,component_index,component_kind,title,content,sha256_hash,start_line,end_line,token_count,detail)
                        VALUES %s
                        ON CONFLICT(file_uuid, component_index) DO UPDATE SET component_kind=EXCLUDED.component_kind,title=EXCLUDED.title,content=EXCLUDED.content,sha256_hash=EXCLUDED.sha256_hash,start_line=EXCLUDED.start_line,end_line=EXCLUDED.end_line,token_count=EXCLUDED.token_count,detail=EXCLUDED.detail
                        RETURNING component_uuid::text, component_index
                    """, component_rows)
                    returned = cur.fetchall()
                    component_ids = [str(row[0]) for row in sorted(returned, key=lambda item: item[1])]
                entity_rows = []
                concept_rows = []
                signal_rows = []
                for comp_uuid, comp in zip(component_ids, component_payloads):
                    comp_node = self.graph_node(cur, "component", f"spine-component://{comp_uuid}", comp.title or f"component {comp.index}", {"file_uuid": file_uuid, "component_kind": comp.kind})
                    self.graph_edge(cur, file_node, comp_node, "FILE_HAS_COMPONENT", {"component_index": comp.index})
                    for ent in self.extract_entities(comp):
                        entity_rows.append((comp_uuid, ent["entity_kind"], ent["value"], ent["normalized_value"], ent["confidence_bps"], ent["source"], ent["context"]))
                    for con in self.extract_concepts(comp):
                        concept_rows.append((comp_uuid, con["concept_kind"], con["value"], con["normalized_value"], con["confidence_bps"], con["source"]))
                    feat = self.sticker_features(comp)
                    signal_rows.append((comp_uuid, feat["uppercase_ratio"], feat["ellipsis_density"], feat["punctuation_velocity"], feat["first_person_gravity"], feat["directive_ratio"], feat["ledger_density"], Json(feat)))
                if entity_rows:
                    execute_values(cur, "INSERT INTO lucidota_spine.entity(component_uuid,entity_kind,value,normalized_value,confidence_bps,source,context) VALUES %s ON CONFLICT(component_uuid,entity_kind,normalized_value,source) DO UPDATE SET value=EXCLUDED.value,confidence_bps=EXCLUDED.confidence_bps,context=EXCLUDED.context", entity_rows)
                if concept_rows:
                    execute_values(cur, "INSERT INTO lucidota_spine.concept(component_uuid,concept_kind,value,normalized_value,confidence_bps,source) VALUES %s ON CONFLICT(component_uuid,concept_kind,normalized_value,source) DO UPDATE SET value=EXCLUDED.value,confidence_bps=EXCLUDED.confidence_bps", concept_rows)
                if signal_rows:
                    execute_values(cur, "INSERT INTO lucidota_spine.sticker_signal(component_uuid,uppercase_ratio,ellipsis_density,punctuation_velocity,first_person_gravity,directive_ratio,ledger_density,raw_features) VALUES %s ON CONFLICT(component_uuid) DO UPDATE SET uppercase_ratio=EXCLUDED.uppercase_ratio,ellipsis_density=EXCLUDED.ellipsis_density,punctuation_velocity=EXCLUDED.punctuation_velocity,first_person_gravity=EXCLUDED.first_person_gravity,directive_ratio=EXCLUDED.directive_ratio,ledger_density=EXCLUDED.ledger_density,raw_features=EXCLUDED.raw_features", signal_rows)
                return len(component_rows), len(entity_rows), len(concept_rows)

    def record_error(self, conn, absolute_path: str, exc: BaseException) -> None:
        if not self.run_uuid:
            return
        if conn is None:
            try:
                with psycopg2.connect(self.dsn) as direct:
                    with direct.cursor() as cur:
                        cur.execute("INSERT INTO lucidota_spine.ingest_error(run_uuid,absolute_path,error_kind,error_text) VALUES (%s::uuid,%s,%s,%s)", (self.run_uuid, absolute_path, exc.__class__.__name__, str(exc)[:1000]))
                return
            except psycopg2.Error:
                return
        with conn.cursor() as cur:
            cur.execute("INSERT INTO lucidota_spine.ingest_error(run_uuid,absolute_path,error_kind,error_text) VALUES (%s::uuid,%s,%s,%s)", (self.run_uuid, absolute_path, exc.__class__.__name__, str(exc)[:1000]))

    def write_dashboard_json(self) -> None:
        payload = {"run_uuid": self.run_uuid, "root": str(self.root), "counts": self.counts, "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        target = self.dashboard_json
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(temp_name, target)

    def run(self) -> dict[str, Any]:
        self.init_db()
        self.run_uuid = self.create_run()
        with DurableConnection(self.dsn) as db:
            conn = db.require_conn()
            for path in self.walk_files():
                self.counts["files_seen"] += 1
                try:
                    work = self.build_file_work(path)
                    self.counts["bytes_seen"] += work.size_bytes
                    def txn(active_conn):
                        c_count, e_count, k_count = self.write_file(active_conn, work)
                        self.counts["components_written"] += c_count
                        self.counts["entities_written"] += e_count
                        self.counts["concepts_written"] += k_count
                        return True
                    db.run_serialized(txn)
                    self.counts["files_written"] += 1
                except Exception as exc:
                    conn.rollback()
                    self.counts["files_failed"] += 1
                    self.record_error(conn, str(path), exc)
                    conn.commit()
                self.write_dashboard_json()
            with conn.cursor() as cur:
                cur.execute("UPDATE lucidota_spine.ingest_run SET status='succeeded', finished_at=now(), files_seen=%s, files_written=%s, files_failed=%s, components_written=%s, entities_written=%s, concepts_written=%s, bytes_seen=%s WHERE run_uuid=%s::uuid", (self.counts["files_seen"], self.counts["files_written"], self.counts["files_failed"], self.counts["components_written"], self.counts["entities_written"], self.counts["concepts_written"], self.counts["bytes_seen"], self.run_uuid))
            conn.commit()
        self.write_dashboard_json()
        return {"ok": True, "run_uuid": self.run_uuid, "counts": self.counts, "dashboard_json": str(self.dashboard_json)}


def main() -> int:
    parser = argparse.ArgumentParser(prog="spine")
    parser.add_argument("root", nargs="?")
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--init-db", action="store_true")
    parser.add_argument("--dashboard-json", type=Path, default=Path("05_OUTPUTS/spine_dashboard.json"))
    args = parser.parse_args()
    if args.init_db:
        Spine(Path.cwd(), args.dsn, args.dashboard_json).init_db()
        print("schema_ready")
        return 0
    if not args.root:
        parser.error("root is required unless --init-db is set")
    result = Spine(Path(args.root).expanduser().resolve(), args.dsn, args.dashboard_json).run()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
