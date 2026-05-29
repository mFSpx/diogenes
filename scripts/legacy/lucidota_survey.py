#!/usr/bin/env python3
"""LUCIDOTA Survey Protocol: first-contact URL/file triage + local CAS + pivot hints."""
from __future__ import annotations

import argparse
import hashlib
import html.parser
import ipaddress
import json
import mimetypes
import os
import re
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import psycopg
import requests
from lucidota_cas_journal import append_record as append_cas_journal

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[2] if THIS_FILE.parent.name == "legacy" else THIS_FILE.parents[1]
SCHEMA_PATH = ROOT / "06_SCHEMA" / "003_survey_protocol.sql"
DEFAULT_DB = "postgresql://mfspx@/lucidota_graph"
DEFAULT_STATE_DB = "postgresql://mfspx@/lucidota_state"
DEFAULT_VAULT = ROOT / "03_VAULT" / "cas"
DEFAULT_MAX_BYTES = int(os.environ.get("LUCIDOTA_MAX_FILE_BYTES", "1500000"))
MAX_REDIRECTS = 5


def optional_tree_sitter_summary(text: str, language_hint: str = "") -> dict:
    """Best-effort structural parse slot.

    Uses tree_sitter when a runtime grammar is present; otherwise returns an
    explicit unavailable marker so callers can distinguish fallback parsing from
    a fake tree-sitter result. No network, no parser downloads.
    """
    try:
        from tree_sitter import Parser  # type: ignore
        try:
            from tree_sitter_languages import get_language  # type: ignore
        except Exception as exc:
            return {"engine": "tree_sitter", "status": "grammar_unavailable", "error": str(exc)[:120]}
        lang_name = "python" if language_hint.endswith(".py") or "python" in language_hint else "html"
        parser = Parser()
        language = get_language(lang_name)
        if hasattr(parser, "set_language"):
            parser.set_language(language)
        else:
            parser.language = language
        tree = parser.parse(text[:1_000_000].encode("utf-8", errors="ignore"))
        root = tree.root_node
        return {
            "engine": "tree_sitter",
            "status": "ok",
            "language": lang_name,
            "root_type": root.type,
            "named_child_count": root.named_child_count,
            "has_error": root.has_error,
        }
    except Exception as exc:
        return {"engine": "tree_sitter", "status": "unavailable", "error": str(exc)[:120]}


def gate_action(
    state_db_url: str,
    workflow_id: str,
    run_id: str,
    source_id: str,
    action_name: str,
    action_kind: str,
    target: str,
    requested_by: str = "lucidota_survey",
    operator_approved: bool = False,
    explicit_cli_target: bool = False,
) -> str:
    """Enforce source_policy through governance_gate before privileged Survey actions."""
    with psycopg.connect(state_db_url) as conn:
        policy = conn.execute("""
          SELECT allowed_actions, requires_user_approval
          FROM lucidota_control.source_policy
          WHERE source_id=%s
        """, (source_id,)).fetchone()
        if not policy:
            status = "denied"
            rationale = f"missing source_policy for {source_id}"
        else:
            allowed_actions, requires_user_approval = policy
            allowed = action_name in (allowed_actions or [])
            if not allowed:
                status = "denied"
                rationale = f"action {action_name} not allowed by source_policy {source_id}"
            elif requires_user_approval and not (operator_approved or explicit_cli_target):
                status = "pending"
                rationale = f"operator approval required by source_policy {source_id}"
            else:
                status = "approved" if operator_approved else "not_required"
                rationale = f"allowed by source_policy {source_id}"
        conn.execute("""
          INSERT INTO lucidota_control.governance_gate
            (workflow_id, run_id, action_kind, requested_by, target, risk_level, policy_mode, approval_status, rationale, evidence, decided_at)
          VALUES (%s,%s,%s,%s,%s,%s,'user_controlled',%s,%s,%s::jsonb, CASE WHEN %s IN ('approved','denied','not_required') THEN now() ELSE NULL END)
        """, (
            workflow_id,
            run_id,
            action_kind,
            requested_by,
            target,
            "medium" if status == "pending" else "low",
            status,
            rationale,
            json.dumps({"source_id": source_id, "action": action_name, "explicit_cli_target": explicit_cli_target}),
            status,
        ))
        conn.commit()
    if status in ("denied", "pending"):
        raise PermissionError(f"governance gate {status}: {rationale}")
    return status


class LinkAndStructureParser(html.parser.HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[str] = []
        self.hidden_links = 0
        self.tags: dict[str, int] = {}
        self.depth = 0
        self.max_depth = 0

    def handle_starttag(self, tag: str, attrs):
        self.tags[tag] = self.tags.get(tag, 0) + 1
        self.depth += 1
        self.max_depth = max(self.max_depth, self.depth)
        attr = {k.lower(): (v or "") for k, v in attrs}
        if tag == "a" and attr.get("href"):
            href = urljoin(self.base_url, attr["href"])
            self.links.append(href)
            style = attr.get("style", "").lower()
            if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", ""):
                self.hidden_links += 1

    def handle_endtag(self, tag: str):
        self.depth = max(0, self.depth - 1)

    def summary(self) -> dict:
        return {
            "parser": "html.parser+optional-tree-sitter-slot",
            "link_count": len(self.links),
            "hidden_link_count": self.hidden_links,
            "tag_count": sum(self.tags.values()),
            "max_depth": self.max_depth,
            "top_tags": sorted(self.tags.items(), key=lambda kv: (-kv[1], kv[0]))[:12],
        }


class MultiPatternIndex:
    def __init__(self, patterns: Iterable[str]):
        self.goto: list[dict[str, int]] = [dict()]
        self.fail: list[int] = [0]
        self.out: list[list[str]] = [[]]
        for raw in patterns:
            pat = raw.lower().strip()
            if not pat:
                continue
            state = 0
            for ch in pat:
                nxt = self.goto[state].get(ch)
                if nxt is None:
                    nxt = len(self.goto)
                    self.goto[state][ch] = nxt
                    self.goto.append({})
                    self.fail.append(0)
                    self.out.append([])
                state = nxt
            self.out[state].append(pat)
        queue = list(self.goto[0].values())
        for state in queue:
            self.fail[state] = 0
        head = 0
        while head < len(queue):
            r = queue[head]
            head += 1
            for ch, s in self.goto[r].items():
                queue.append(s)
                f = self.fail[r]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[s] = self.goto[f].get(ch, 0)
                self.out[s].extend(self.out[self.fail[s]])

    def find(self, text: str) -> list[dict]:
        if len(self.goto) == 1:
            return []
        state = 0
        counts: dict[str, int] = {}
        first: dict[str, int] = {}
        for idx, ch in enumerate(text.lower()):
            while state and ch not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(ch, 0)
            for pat in self.out[state]:
                counts[pat] = counts.get(pat, 0) + 1
                first.setdefault(pat, idx - len(pat) + 1)
        return [{"keyword": k, "count": counts[k], "first_offset": first[k]} for k in sorted(counts)]


@dataclass
class SurveyResult:
    target: str
    method: str
    status_code: int | None
    final_url: str | None
    content_length: int | None
    mime: str
    etag: str
    last_modified: str
    sha256: str | None
    cas_uri: str | None
    decision: str
    keyword_hits: list[dict]
    structure: dict
    wayback: dict
    pivot_candidates: list[dict]
    notes: str


def apply_schema(db_url: str) -> None:
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text())
        conn.commit()


def store_cas(vault: Path, data: bytes) -> tuple[str, str, Path]:
    digest = hashlib.sha256(data).hexdigest()
    subdir = vault / digest[:2] / digest[2:4]
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / digest
    cas_uri = f"cas://sha256/{digest}"
    if not path.exists():
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(str(subdir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError as exc:
            append_cas_journal({
                "stage": "directory_fsync_warning",
                "sha256": digest,
                "cas_uri": cas_uri,
                "relative_path": path.relative_to(vault).as_posix(),
                "error": str(exc),
                "source": "survey",
            })
    append_cas_journal({
        "stage": "written",
        "sha256": digest,
        "cas_uri": cas_uri,
        "relative_path": path.relative_to(vault).as_posix(),
        "size_bytes": len(data),
        "source": "survey",
    })
    return digest, cas_uri, path


def wayback_lookup(url: str, timeout: float) -> dict:
    api = "https://web.archive.org/cdx"
    params = {
        "url": url,
        "output": "json",
        "fl": "timestamp,original,statuscode,mimetype,digest",
        "filter": "statuscode:200",
        "limit": "3",
        "collapse": "digest",
    }
    try:
        r = requests.get(api, params=params, timeout=timeout)
        if r.status_code != 200:
            return {"status": "error", "http_status": r.status_code}
        rows = r.json()
        if not rows or len(rows) == 1:
            return {"status": "none", "captures": []}
        keys = rows[0]
        captures = [dict(zip(keys, row)) for row in rows[1:]]
        return {"status": "ok", "captures": captures}
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def validate_source_url(url: str, allow_local_addresses: bool) -> None:
    """Keep URL source intake on public web addresses by default."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"unsupported URL scheme: {parsed.scheme or '<none>'}")
    if not parsed.hostname:
        raise ValueError("URL host is required")
    if allow_local_addresses:
        return
    host = parsed.hostname
    try:
        literal = ipaddress.ip_address(host)
        addresses = [literal]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise ValueError(f"DNS resolution failed for {host}: {exc}") from exc
        addresses = sorted({ipaddress.ip_address(info[4][0]) for info in infos}, key=str)
    blocked = [
        str(ip)
        for ip in addresses
        if (
            not ip.is_global
            or ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    ]
    if blocked:
        raise ValueError(f"source address requires explicit local-address option for {host}: {', '.join(blocked[:4])}")


def source_request(method: str, url: str, timeout: float, allow_local_addresses: bool, *, stream: bool = False) -> requests.Response:
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        validate_source_url(current, allow_local_addresses)
        response = requests.request(method, current, allow_redirects=False, timeout=timeout, stream=stream)
        if response.is_redirect or response.is_permanent_redirect:
            location = response.headers.get("location")
            if not location:
                return response
            current = urljoin(response.url, location)
            continue
        return response
    raise ValueError(f"too many redirects; max {MAX_REDIRECTS}")


def fetch_target(target: str, max_bytes: int, timeout: float, do_fetch: bool, allow_local_addresses: bool = False) -> tuple[SurveyResult, bytes | None]:
    parsed = urlparse(target)
    if parsed.scheme in ("http", "https"):
        head = source_request("HEAD", target, timeout, allow_local_addresses)
        mime = head.headers.get("content-type", "").split(";", 1)[0]
        clen = int(head.headers.get("content-length") or 0) or None
        base = SurveyResult(
            target=target, method="HEAD", status_code=head.status_code, final_url=head.url,
            content_length=clen, mime=mime, etag=head.headers.get("etag", ""),
            last_modified=head.headers.get("last-modified", ""), sha256=None, cas_uri=None,
            decision="metadata_only", keyword_hits=[], structure={}, wayback={}, pivot_candidates=[], notes="",
        )
        if not do_fetch:
            return base, None
        if clen is not None and clen > max_bytes:
            base.decision = "too_large"
            base.notes = f"content-length {clen} exceeds max {max_bytes}"
            return base, None
        get = source_request("GET", target, timeout, allow_local_addresses, stream=True)
        chunks = []
        total = 0
        for chunk in get.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                base.method = "GET"
                base.status_code = get.status_code
                base.final_url = get.url
                base.decision = "too_large"
                base.notes = f"download exceeded max {max_bytes}"
                return base, None
            chunks.append(chunk)
        data = b"".join(chunks)
        base.method = "GET"
        base.status_code = get.status_code
        base.final_url = get.url
        base.content_length = len(data)
        base.mime = get.headers.get("content-type", mime).split(";", 1)[0]
        return base, data
    path = Path(target).expanduser().resolve()
    data = path.read_bytes()
    mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    result = SurveyResult(
        target=str(path), method="FILE", status_code=None, final_url=str(path),
        content_length=len(data), mime=mime, etag="", last_modified=str(int(path.stat().st_mtime)),
        sha256=None, cas_uri=None, decision="stored_artifact", keyword_hits=[], structure={}, wayback={}, pivot_candidates=[], notes="",
    )
    if len(data) > max_bytes:
        result.decision = "too_large"
        result.notes = f"file size {len(data)} exceeds max {max_bytes}"
        return result, None
    return result, data


def analyze(result: SurveyResult, data: bytes | None, keywords: list[str], do_wayback: bool, timeout: float) -> None:
    text = ""
    if data:
        text = data[:1_000_000].decode("utf-8", errors="ignore")
        result.keyword_hits = MultiPatternIndex(keywords).find(text)
        if "html" in result.mime or "<html" in text[:2000].lower():
            parser = LinkAndStructureParser(result.final_url or result.target)
            parser.feed(text)
            result.structure = parser.summary()
            result.structure["tree_sitter"] = optional_tree_sitter_summary(text, result.mime)
            for link in parser.links[:50]:
                result.pivot_candidates.append({"candidate": link, "kind": "link", "score": 20, "reason": "html link"})
        elif result.method == "FILE" and result.target.endswith((".py", ".rs", ".js", ".ts", ".c", ".h")):
            result.structure = {"parser": "optional-tree-sitter-slot", "tree_sitter": optional_tree_sitter_summary(text, result.target)}
        for hit in result.keyword_hits:
            result.pivot_candidates.append({"candidate": hit["keyword"], "kind": "keyword", "score": 50 + hit["count"], "reason": "multi-pattern hit"})
    if do_wayback and urlparse(result.target).scheme in ("http", "https"):
        result.wayback = wayback_lookup(result.target, timeout)
        if result.wayback.get("status") == "ok":
            for cap in result.wayback.get("captures", [])[:3]:
                ts = cap.get("timestamp")
                original = cap.get("original", result.target)
                if ts:
                    result.pivot_candidates.append({"candidate": f"https://web.archive.org/web/{ts}/{original}", "kind": "archive", "score": 35, "reason": "wayback capture"})


def persist(db_url: str, state_db_url: str, result: SurveyResult, size: int | None) -> None:
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            artifact_id = None
            if result.sha256 and result.cas_uri:
                cur.execute(
                    """
                    INSERT INTO lucidota_survey.artifact (cas_uri, sha256, size_bytes, mime, source_url)
                    VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT (sha256) DO UPDATE SET source_url = COALESCE(lucidota_survey.artifact.source_url, EXCLUDED.source_url)
                    RETURNING artifact_id
                    """,
                    (result.cas_uri, result.sha256, size or 0, result.mime, result.target),
                )
                artifact_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO lucidota_survey.survey_observation
                (target, method, status_code, final_url, content_length, mime, etag, last_modified, sha256, artifact_id,
                 survey_decision, keyword_hits, structure, wayback, notes)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s)
                RETURNING observation_id
                """,
                (result.target, result.method, result.status_code, result.final_url, result.content_length, result.mime,
                 result.etag, result.last_modified, result.sha256, artifact_id, result.decision,
                 json.dumps(result.keyword_hits), json.dumps(result.structure), json.dumps(result.wayback), result.notes),
            )
            obs_id = cur.fetchone()[0]
            for cand in result.pivot_candidates[:100]:
                cur.execute(
                    """
                    INSERT INTO lucidota_survey.pivot_candidate
                    (observation_id, source_target, candidate, candidate_kind, score, reason)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    """,
                    (obs_id, result.target, cand["candidate"], cand["kind"], cand["score"], cand["reason"]),
                )
        conn.commit()
    if result.sha256 and result.cas_uri:
        append_cas_journal({
            "stage": "db_committed",
            "sha256": result.sha256,
            "cas_uri": result.cas_uri,
            "relative_path": result.sha256[:2] + "/" + result.sha256[2:4] + "/" + result.sha256,
            "size_bytes": size or 0,
            "source": result.target,
        })
    try:
        with psycopg.connect(state_db_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.workflow_event (workflow_id, run_id, phase, status, source, detail)
                    VALUES ('survey-protocol', %s, 'survey', 'succeeded', 'lucidota_survey', %s::jsonb)
                    """,
                    (str(obs_id), json.dumps({"target": result.target, "decision": result.decision, "sha256": result.sha256})),
                )
            conn.commit()
    except Exception as exc:
        append_cas_journal({
            "stage": "workflow_event_warning",
            "target": result.target,
            "sha256": result.sha256,
            "error": str(exc),
            "source": "survey",
        })



def run_one(args, target: str) -> SurveyResult:
    parsed = urlparse(target)
    run_id = hashlib.sha256(f"{target}:{time.time_ns()}".encode()).hexdigest()[:24]
    if not getattr(args, "no_db", False):
        if parsed.scheme in ("http", "https"):
            gate_action(
                args.state_db_url, "survey-protocol", run_id, "public_web",
                "get_small_body" if args.fetch else "get_metadata",
                "external_read", target,
                operator_approved=getattr(args, "operator_approved", False),
            )
        elif parsed.scheme == "":
            gate_action(
                args.state_db_url, "survey-protocol", run_id, "local_files",
                "read_operator_selected_file", "drive_read", target,
                operator_approved=getattr(args, "operator_approved", False),
                explicit_cli_target=True,
            )
        if args.wayback:
            gate_action(
                args.state_db_url, "survey-protocol", run_id, "wayback",
                "lookup_metadata", "external_read", target,
                operator_approved=getattr(args, "operator_approved", False),
            )
    result, data = fetch_target(target, args.max_bytes, args.timeout, args.fetch, getattr(args, "allow_local_addresses", False))
    if data is not None:
        digest, cas_uri, _path = store_cas(args.vault, data)
        result.sha256 = digest
        result.cas_uri = cas_uri
        result.decision = "stored_artifact"
    keywords = args.keyword + [k.strip() for k in args.keywords.split(",") if k.strip()]
    analyze(result, data, keywords, args.wayback, args.timeout)
    if not args.no_db:
        persist(args.db_url, args.state_db_url, result, len(data) if data is not None else None)
    return result



def promotion_candidates(root: SurveyResult, hops: list[dict], threshold: int) -> list[dict]:
    candidates = []
    for cand in root.pivot_candidates:
        if int(cand.get("score", 0)) >= threshold:
            candidates.append({"source": root.target, **cand})
    for hop in hops:
        for cand in hop.get("pivot_candidates", []) or []:
            if int(cand.get("score", 0)) >= threshold:
                candidates.append({"source": hop.get("target", ""), "depth": hop.get("depth", 0), **cand})
    candidates.sort(key=lambda c: (-int(c.get("score", 0)), str(c.get("candidate", ""))))
    return candidates[:50]


def bounded_hop(args, root: SurveyResult) -> list[dict]:
    if args.hop_depth <= 0:
        return []
    seen = {root.target}
    frontier = [c["candidate"] for c in root.pivot_candidates if c.get("kind") == "link"][: args.max_pivots]
    hop_results = []
    for depth in range(1, args.hop_depth + 1):
        next_frontier = []
        for target in frontier[: args.max_pivots]:
            if target in seen or urlparse(target).scheme not in ("http", "https"):
                continue
            seen.add(target)
            try:
                prior_fetch = args.fetch
                if depth > 1:
                    args.fetch = False
                result = run_one(args, target)
                args.fetch = prior_fetch
                hop_results.append({"depth": depth, **asdict(result)})
                next_frontier.extend(c["candidate"] for c in result.pivot_candidates if c.get("kind") == "link")
            except Exception as exc:
                hop_results.append({"depth": depth, "target": target, "decision": "failed", "error": str(exc)})
        frontier = [t for t in next_frontier if t not in seen][: args.max_pivots]
    return hop_results

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LUCIDOTA Survey Protocol: HEAD/GET/file triage into local CAS + Postgres.")
    ap.add_argument("target", help="URL or local file path")
    ap.add_argument("--keyword", action="append", default=[], help="Keyword/pattern for multi-pattern scan. Repeatable.")
    ap.add_argument("--keywords", default="", help="Comma-separated keywords.")
    ap.add_argument("--fetch", action="store_true", help="Fetch/store small URL bodies after HEAD. Files are read by default.")
    ap.add_argument("--wayback", action="store_true", help="Query Wayback CDX for archival pivots.")
    ap.add_argument("--allow-local-addresses", action="store_true", help="Allow operator-confirmed local-address URL intake. Default is public-web only.")
    ap.add_argument("--operator-approved", action="store_true", help="Mark this explicit operator action as approved for governance-gated enrichment.")
    ap.add_argument("--hop-depth", type=int, default=0, help="Bounded link-hop depth for pivot surveying.")
    ap.add_argument("--max-pivots", type=int, default=8, help="Maximum pivots to follow per hop.")
    ap.add_argument("--promote-threshold", type=int, default=35, help="Score threshold for promotion candidates.")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--db-url", default=os.environ.get("LUCIDOTA_GRAPH_DATABASE_URL", DEFAULT_DB))
    ap.add_argument("--state-db-url", default=os.environ.get("DBOS_SYSTEM_DATABASE_URL", DEFAULT_STATE_DB))
    ap.add_argument("--vault", type=Path, default=Path(os.environ.get("LUCIDOTA_CAS_VAULT", DEFAULT_VAULT)))
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args(argv)

    if not args.no_db:
        apply_schema(args.db_url)
    result = run_one(args, args.target)
    payload = asdict(result)
    hops = bounded_hop(args, result)
    if hops:
        payload["hop_results"] = hops
    promotions = promotion_candidates(result, hops, args.promote_threshold)
    if promotions:
        payload["promotion_candidates"] = promotions
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)
