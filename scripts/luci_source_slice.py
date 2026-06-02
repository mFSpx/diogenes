#!/usr/bin/env python3
"""LUCI current-world source slice: live source -> extract -> score -> receipt.

Reusable class-handler for current-world reading tasks. It maps the board state,
fetches a bounded live source item, normalizes it into source_item/claim/extract
records, scores novelty/relevance, and writes DB-backed work records plus a
receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "luci_source"
USER_AGENT = "LUCIDOTA-LUCI/1.0 (+https://github.com/mfspx/LUCIDOTA)"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def db_url(value: str | None = None) -> str:
    return (
        value
        or os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL")
        or os.environ.get("LUCIDOTA_GO_STORAGE_DSN")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def http_get(url: str, *, timeout: float = 15.0, headers: dict[str, str] | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec: live source fetch by operator request
        return resp.read()


def http_get_json(url: str, *, timeout: float = 15.0, headers: dict[str, str] | None = None) -> Any:
    return json.loads(http_get(url, timeout=timeout, headers=headers).decode("utf-8", errors="replace"))


@dataclass
class BoardState:
    actors: list[str]
    resources: list[str]
    constraints: list[str]
    timing: list[str]
    leverage: list[str]
    friction: list[str]
    inertia: list[str]
    visibility: list[str]
    incentives: list[str]
    terrain: list[str]
    available_moves: list[str]
    expected_counter_moves: list[str]
    cheapest_probes: list[str]
    highest_gain_pivots: list[str]


@dataclass
class SourceItem:
    source_kind: str
    source_label: str
    source_item_id: str
    title: str
    url: str
    author: str
    published_at: str
    snippet: str
    claim_text: str
    novelty_score: float
    relevance_score: float
    extract: str
    metadata: dict[str, Any]


def _pick(text: str, options: dict[str, list[str]]) -> list[str]:
    low = text.lower()
    picked: list[str] = []
    for label, keys in options.items():
        if any(k in low for k in keys):
            picked.append(label)
    return picked


def map_board_state(text: str, source_kind: str) -> BoardState:
    low = text.lower()
    return BoardState(
        actors=["operator", "Indy_READs", "LUCI", source_kind],
        resources=_pick(low, {
            "live source adapters": ["live world", "current world", "source adapter"],
            "hn/advisory feeds": ["hacker news", "hn"],
            "arxiv paper search": ["arxiv", "paper", "preprint"],
            "github releases/issues": ["github", "release", "issues"],
            "reddit community signal": ["reddit", "subreddit", "localllama"],
        }),
        constraints=_pick(low, {
            "receipt law": ["receipt", "proof"],
            "bounded live fetch": ["bounded", "cheap", "small"],
            "no browser by default": ["browser", "visual"],
            "rust/db-first": ["rust", "db", "postgres"],
        }),
        timing=_pick(low, {
            "now": ["now", "today", "immediately"],
            "iterative": ["iterate", "retry", "mutate", "learn"],
            "asynchronous": ["async", "queue", "receipt"],
        }),
        leverage=_pick(low, {
            "source class-handler": ["adapter", "source", "class-handler"],
            "novelty scoring": ["novelty", "relevance", "score"],
            "database receipts": ["db", "postgres", "receipt"],
        }),
        friction=_pick(low, {
            "rate limits": ["rate", "limit", "quota"],
            "stale source drift": ["stale", "drift"],
            "one-off script risk": ["one-off", "script"],
        }),
        inertia=_pick(low, {
            "legacy names": ["claw", "dbos"],
            "wrapper leakage": ["wrapper", "shell"],
        }),
        visibility=_pick(low, {
            "receipt-backed": ["receipt"],
            "operator-visible": ["operator", "Indy_READs", "luci"],
            "live world": ["live world", "current world", "source"],
        }),
        incentives=_pick(low, {
            "faster routing": ["fast", "speed", "quick"],
            "learn by doing": ["learn", "study", "improve"],
            "current-world memory": ["current", "world", "live"],
        }),
        terrain=[source_kind, "live_api"],
        available_moves=[
            "fetch live source",
            "normalize source_item",
            "score novelty/relevance",
            "write receipt",
        ],
        expected_counter_moves=[
            "rate limited",
            "source unavailable",
            "parse failure",
            "empty result set",
        ],
        cheapest_probes=[
            "fetch one bounded live item",
            "validate normalized fields",
        ],
        highest_gain_pivots=[
            "promote a live-source adapter into a reusable class-handler",
            "convert recurring source patterns into Treelite/router features",
        ],
    )


def classify_candidate(text: str, source_kind: str, source_label: str) -> dict[str, Any]:
    low = text.lower()
    if any(token in low for token in ("adapter", "source", "live world", "current world", "hacker news", "arxiv", "reddit", "github")):
        kind = "current_world_source_adapter"
    else:
        kind = "operator_learning_class"
    return {
        "candidate_kind": kind,
        "candidate_name": f"luci_{kind}",
        "source_kind": source_kind,
        "source_label": source_label,
        "feature_hypothesis": [
            "live source fetch",
            "source_item normalization",
            "novelty/relevance scoring",
            "receipt-backed promotion",
        ],
    }


class LiveSourceAdapter:
    kind = "source"
    label = "live-source"

    def fetch(self, query: str, limit: int = 5) -> list[SourceItem]:  # pragma: no cover - interface
        raise NotImplementedError


class HackerNewsAdapter(LiveSourceAdapter):
    kind = "hn"
    label = "Hacker News"

    def fetch(self, query: str, limit: int = 5) -> list[SourceItem]:
        ids = http_get_json("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10.0)
        items: list[SourceItem] = []
        for story_id in ids[:limit]:
            item = http_get_json(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json", timeout=10.0)
            if not item:
                continue
            title = str(item.get("title") or "")
            url = str(item.get("url") or f"https://news.ycombinator.com/item?id={story_id}")
            text = str(item.get("text") or "")
            snippet = (text or title)[:280]
            published = datetime.fromtimestamp(int(item.get("time") or 0), tz=timezone.utc).isoformat().replace("+00:00", "Z") if item.get("time") else now()
            items.append(
                SourceItem(
                    source_kind=self.kind,
                    source_label=self.label,
                    source_item_id=str(item.get("id") or story_id),
                    title=title,
                    url=url,
                    author=str(item.get("by") or ""),
                    published_at=published,
                    snippet=snippet,
                    claim_text=title or snippet or "HN item",
                    novelty_score=0.0,
                    relevance_score=0.0,
                    extract=snippet,
                    metadata={"score": item.get("score"), "descendants": item.get("descendants")},
                )
            )
        return items


class ArxivAdapter(LiveSourceAdapter):
    kind = "arxiv"
    label = "arXiv"

    def fetch(self, query: str, limit: int = 5) -> list[SourceItem]:
        q = urllib.parse.quote(query or "machine learning systems")
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query=all:{q}&start=0&max_results={limit}&sortBy=submittedDate&sortOrder=descending"
        )
        root = ET.fromstring(http_get(url, timeout=15.0))
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        items: list[SourceItem] = []
        for entry in root.findall("atom:entry", ns):
            title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
            summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())
            link = ""
            for link_el in entry.findall("atom:link", ns):
                if link_el.attrib.get("rel") == "alternate":
                    link = link_el.attrib.get("href", "")
                    break
            authors = [a.findtext("atom:name", default="", namespaces=ns) or "" for a in entry.findall("atom:author", ns)]
            published = entry.findtext("atom:published", default=now(), namespaces=ns) or now()
            item_id = (entry.findtext("atom:id", default="", namespaces=ns) or link or title).rsplit("/", 1)[-1]
            items.append(
                SourceItem(
                    source_kind=self.kind,
                    source_label=self.label,
                    source_item_id=item_id,
                    title=title,
                    url=link or item_id,
                    author=", ".join(a for a in authors if a),
                    published_at=published,
                    snippet=summary[:280],
                    claim_text=title or summary or "arXiv paper",
                    novelty_score=0.0,
                    relevance_score=0.0,
                    extract=summary[:500],
                    metadata={"query": query},
                )
            )
        return items


class RedditAdapter(LiveSourceAdapter):
    kind = "reddit"
    label = "Reddit"

    def __init__(self, subreddit: str = "LocalLLaMA") -> None:
        self.subreddit = subreddit

    def fetch(self, query: str, limit: int = 5) -> list[SourceItem]:
        url = f"https://www.reddit.com/r/{self.subreddit}/hot.json?limit={limit}"
        payload = http_get_json(url, timeout=15.0, headers={"Accept": "application/json"})
        items: list[SourceItem] = []
        for child in payload.get("data", {}).get("children", [])[:limit]:
            data = child.get("data", {})
            title = str(data.get("title") or "")
            permalink = "https://www.reddit.com" + str(data.get("permalink") or "")
            created = datetime.fromtimestamp(float(data.get("created_utc") or 0), tz=timezone.utc).isoformat().replace("+00:00", "Z") if data.get("created_utc") else now()
            excerpt = (str(data.get("selftext") or "")[:280] or title)
            items.append(
                SourceItem(
                    source_kind=self.kind,
                    source_label=f"r/{self.subreddit}",
                    source_item_id=str(data.get("name") or data.get("id") or title[:24]),
                    title=title,
                    url=permalink,
                    author=str(data.get("author") or ""),
                    published_at=created,
                    snippet=excerpt,
                    claim_text=title or excerpt or "Reddit post",
                    novelty_score=0.0,
                    relevance_score=0.0,
                    extract=excerpt[:500],
                    metadata={"subreddit": self.subreddit, "score": data.get("score")},
                )
            )
        return items


class GithubAdapter(LiveSourceAdapter):
    kind = "github"
    label = "GitHub"

    def fetch(self, query: str, limit: int = 5) -> list[SourceItem]:
        html = http_get("https://github.com/trending?since=daily", timeout=15.0).decode("utf-8", errors="replace")
        repo_paths: list[str] = []
        for owner, repo in re.findall(r'href="/([^"/]+)/([^"/]+)/?"', html):
            path = f"{owner}/{repo}"
            if path not in repo_paths:
                repo_paths.append(path)
            if len(repo_paths) >= limit:
                break
        items: list[SourceItem] = []
        for path in repo_paths[:limit]:
            release_url = f"https://api.github.com/repos/{path}/releases/latest"
            release_title = ""
            release_tag = ""
            release_html = ""
            try:
                release = http_get_json(release_url, timeout=12.0, headers={"Accept": "application/vnd.github+json"})
                release_title = str(release.get("name") or release.get("tag_name") or "")
                release_tag = str(release.get("tag_name") or "")
                release_html = str(release.get("html_url") or "")
            except Exception:
                release_html = f"https://github.com/{path}"
            items.append(
                SourceItem(
                    source_kind=self.kind,
                    source_label="GitHub trending",
                    source_item_id=path,
                    title=path,
                    url=release_html or f"https://github.com/{path}",
                    author="",
                    published_at=now(),
                    snippet=release_title or release_tag or path,
                    claim_text=f"Trending repo: {path}",
                    novelty_score=0.0,
                    relevance_score=0.0,
                    extract=release_title or release_tag or path,
                    metadata={"release_tag": release_tag, "query": query},
                )
            )
        return items


ADAPTERS: dict[str, LiveSourceAdapter] = {
    "hn": HackerNewsAdapter(),
    "arxiv": ArxivAdapter(),
    "reddit": RedditAdapter(),
    "github": GithubAdapter(),
}


def choose_adapter(text: str, source: str | None = None) -> tuple[str, LiveSourceAdapter, str]:
    low = text.lower()
    if source and source != "auto":
        adapter = ADAPTERS.get(source.lower())
        if not adapter:
            raise ValueError(f"unknown source adapter: {source}")
        return source.lower(), adapter, _query_for_source(text, source.lower())
    if any(token in low for token in ("arxiv", "paper", "preprint")):
        return "arxiv", ADAPTERS["arxiv"], _query_for_source(text, "arxiv")
    if any(token in low for token in ("reddit", "subreddit", "localllama")):
        return "reddit", ADAPTERS["reddit"], _query_for_source(text, "reddit")
    if any(token in low for token in ("github", "release", "issues", "trending")):
        return "github", ADAPTERS["github"], _query_for_source(text, "github")
    return "hn", ADAPTERS["hn"], _query_for_source(text, "hn")


def _query_for_source(text: str, source_kind: str) -> str:
    low = text.lower()
    if source_kind == "arxiv":
        for hint in ("arxiv", "paper", "preprint"):
            low = low.replace(hint, " ")
        q = " ".join(tok for tok in re.split(r"\W+", low) if tok)
        return q or "machine learning systems"
    if source_kind == "reddit":
        if "localllama" in low or "local llama" in low:
            return "LocalLLaMA"
        return "LocalLLaMA"
    if source_kind == "github":
        return "trending"
    return "top stories"


def score_item(prompt: str, item: SourceItem) -> dict[str, Any]:
    prompt_terms = {t for t in re.split(r"\W+", prompt.lower()) if len(t) > 2}
    item_terms = {t for t in re.split(r"\W+", f"{item.title} {item.snippet} {item.extract}".lower()) if len(t) > 2}
    overlap = len(prompt_terms & item_terms)
    relevance = min(1.0, 0.2 + 0.18 * overlap)
    novelty = 1.0
    if item.published_at and "Z" in item.published_at:
        try:
            ts = datetime.fromisoformat(item.published_at.replace("Z", "+00:00"))
            age_hours = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0)
            novelty = max(0.1, min(1.0, 1.0 - (age_hours / 48.0)))
        except Exception:
            novelty = 0.6
    item.relevance_score = round(relevance, 3)
    item.novelty_score = round(novelty, 3)
    return {
        "gain": round(relevance * 0.7 + novelty * 0.3, 3),
        "cost": 0.12,
        "risk": 0.1 if item.source_kind in {"github", "reddit"} else 0.08,
        "reversibility": "high",
        "score": round(max(0.0, relevance * 0.7 + novelty * 0.3 - 0.12), 3),
        "verdict": "promote" if relevance >= 0.35 else "stall",
    }


def build_source_items(text: str, adapter_kind: str, adapter: LiveSourceAdapter, query: str, limit: int = 5) -> tuple[list[SourceItem], dict[str, Any]]:
    try:
        items = adapter.fetch(query, limit=limit)
        probe = {"passed": True, "reason": "", "item_count": len(items), "query": query}
    except Exception as exc:
        items = []
        probe = {"passed": False, "reason": f"{type(exc).__name__}:{exc}", "item_count": 0, "query": query}
    return items, probe


def choose_focus_item(prompt: str, items: list[SourceItem]) -> SourceItem | None:
    if not items:
        return None
    scored = [(score_item(prompt, item), item) for item in items]
    scored.sort(key=lambda pair: (pair[0]["score"], pair[0]["gain"]), reverse=True)
    return scored[0][1]


def write_db_rows(
    conn: psycopg.Connection,
    *,
    run_id: str,
    text: str,
    adapter_kind: str,
    adapter_label: str,
    items: list[SourceItem],
    focus: SourceItem | None,
    board: BoardState,
    candidate: dict[str, Any],
    probe: dict[str, Any],
    score: dict[str, Any],
    receipt_path: str,
) -> dict[str, str]:
    with conn.cursor(row_factory=dict_row) as cur:
        event_id = sha256_text(json.dumps({"run_id": run_id, "source_kind": adapter_kind, "text": text}, sort_keys=True))
        raw_ref = f"inline://luci-source/{sha256_text(adapter_kind + adapter_label)[:16]}/{run_id}"
        raw_row = cur.execute(
            """
            INSERT INTO lucidota_control.raw_artifact(raw_ref, raw_sha256, hash_algo, source, actor, byte_count, char_count, mime_type, storage_hint, detail)
            VALUES (%s, %s, 'sha256', 'luci_source_slice', 'operator', %s, %s, 'application/json', 'receipt_or_artifact', %s::jsonb)
            ON CONFLICT (raw_ref) DO UPDATE SET
              raw_sha256 = EXCLUDED.raw_sha256,
              hash_algo = EXCLUDED.hash_algo,
              source = EXCLUDED.source,
              actor = EXCLUDED.actor,
              byte_count = EXCLUDED.byte_count,
              char_count = EXCLUDED.char_count,
              mime_type = EXCLUDED.mime_type,
              storage_hint = EXCLUDED.storage_hint,
              detail = EXCLUDED.detail
            RETURNING raw_artifact_uuid::text
            """,
            (
                raw_ref,
                sha256_text(stable_json({"run_id": run_id, "adapter_kind": adapter_kind, "items": [item.source_item_id for item in items]})),
                len(stable_json(items).encode("utf-8", errors="replace")),
                len(stable_json(items)),
                json.dumps({"adapter_kind": adapter_kind, "adapter_label": adapter_label}),
            ),
        ).fetchone()
        raw_artifact_uuid = raw_row["raw_artifact_uuid"] if isinstance(raw_row, dict) else raw_row[0]
        cur.execute(
            """
            INSERT INTO lucidota_control.event_envelope(event_id, ts, source, actor, raw_ref, raw_artifact_uuid, verbatim_hash, hash_algo, text, entities, claims, actions_requested, artifacts_referenced, risk_flags, route_candidates, board_features, embedding_ref, detail)
            VALUES (%s, now(), 'luci_source_slice', 'operator', %s, %s::uuid, %s, 'sha256', %s, '[]'::jsonb, '[]'::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, NULL, %s::jsonb)
            ON CONFLICT (event_id) DO UPDATE SET
              ts = EXCLUDED.ts,
              source = EXCLUDED.source,
              actor = EXCLUDED.actor,
              raw_ref = EXCLUDED.raw_ref,
              raw_artifact_uuid = EXCLUDED.raw_artifact_uuid,
              verbatim_hash = EXCLUDED.verbatim_hash,
              hash_algo = EXCLUDED.hash_algo,
              text = EXCLUDED.text,
              actions_requested = EXCLUDED.actions_requested,
              artifacts_referenced = EXCLUDED.artifacts_referenced,
              risk_flags = EXCLUDED.risk_flags,
              route_candidates = EXCLUDED.route_candidates,
              board_features = EXCLUDED.board_features,
              detail = EXCLUDED.detail
            RETURNING event_id
            """,
            (
                event_id,
                raw_ref,
                raw_artifact_uuid,
                sha256_text(text),
                text,
                json.dumps([candidate["candidate_kind"]]),
                json.dumps([focus.url if focus else adapter_kind]),
                json.dumps([f"probe_passed={probe.get('passed')}"]),
                json.dumps([f"{adapter_kind}_live_source"]),
                json.dumps(asdict(board)),
                json.dumps({
                    "adapter_kind": adapter_kind,
                    "adapter_label": adapter_label,
                    "focus_item": asdict(focus) if focus else None,
                    "item_count": len(items),
                }),
            ),
        )
        work_order_row = cur.execute(
            """
            INSERT INTO lucidota_control.work_order(event_id, lane, work_kind, status, payload, idempotency_key)
            VALUES (%s, 'audit', %s, %s, %s::jsonb, %s)
            ON CONFLICT (idempotency_key) DO UPDATE SET
              event_id = EXCLUDED.event_id,
              lane = EXCLUDED.lane,
              work_kind = EXCLUDED.work_kind,
              status = EXCLUDED.status,
              payload = EXCLUDED.payload,
              updated_at = now()
            RETURNING work_order_uuid::text
            """,
            (
                event_id,
                "luci_current_world_source_slice",
                "succeeded" if probe.get("passed") else "failed",
                json.dumps({
                    "adapter_kind": adapter_kind,
                    "adapter_label": adapter_label,
                    "query": probe.get("query"),
                    "items": [asdict(item) for item in items],
                    "focus_item": asdict(focus) if focus else None,
                    "candidate": candidate,
                    "score": score,
                }),
                f"luci-source:{run_id}:{adapter_kind}:{sha256_text(text)[:16]}",
            ),
        ).fetchone()
        work_order_uuid = work_order_row["work_order_uuid"] if isinstance(work_order_row, dict) else work_order_row[0]
        receipt_row = cur.execute(
            """
            INSERT INTO lucidota_control.work_receipt(event_id, work_order_uuid, receipt_path, receipt_sha256, verdict, cost, gain, artifact_refs, canonical_graph_writes_performed, graph_write_mode, detail)
            VALUES (%s, %s::uuid, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, false, 'staged_only', %s::jsonb)
            RETURNING work_receipt_uuid::text
            """,
            (
                event_id,
                work_order_uuid,
                receipt_path,
                sha256_text(stable_json({"run_id": run_id, "adapter_kind": adapter_kind, "focus_item": asdict(focus) if focus else None, "score": score})),
                score["verdict"],
                json.dumps({"cost": score["cost"], "risk": score["risk"]}),
                json.dumps({"gain": score["gain"], "score": score["score"]}),
                json.dumps([raw_ref] + ([focus.url] if focus and focus.url else [])),
                json.dumps({
                    "adapter_kind": adapter_kind,
                    "adapter_label": adapter_label,
                    "board_state": asdict(board),
                    "items": [asdict(item) for item in items],
                    "focus_item": asdict(focus) if focus else None,
                    "probe": probe,
                    "candidate": candidate,
                }),
            ),
        ).fetchone()
        work_receipt_uuid = receipt_row["work_receipt_uuid"] if isinstance(receipt_row, dict) else receipt_row[0]
        conn.commit()
    return {
        "event_id": event_id,
        "raw_artifact_uuid": raw_artifact_uuid,
        "work_order_uuid": work_order_uuid,
        "work_receipt_uuid": work_receipt_uuid,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    text = args.text or ""
    adapter_kind, adapter, query = choose_adapter(text, args.source)
    board = map_board_state(text, adapter_kind)
    items, probe = build_source_items(text, adapter_kind, adapter, query, limit=args.limit)
    focus = choose_focus_item(text, items)
    candidate = classify_candidate(text, adapter_kind, adapter.label)
    score = score_item(text, focus) if focus else {"gain": 0.0, "cost": 0.2, "risk": 0.2, "reversibility": "high", "score": 0.0, "verdict": "stall"}
    run_id = args.run_id or "luci-source:" + sha256_text(stable_json({"text": text, "source": adapter_kind, "query": query}))[:24]
    receipt_dir = Path(args.receipt_dir or OUT)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"luci_source_{stamp()}.json"
    receipt = {
        "schema": "lucidota.luci.source_slice.receipt.v1",
        "generated_at": now(),
        "run_id": run_id,
        "adapter_kind": adapter_kind,
        "adapter_label": adapter.label,
        "query": query,
        "board_state": asdict(board),
        "items": [asdict(item) for item in items],
        "focus_item": asdict(focus) if focus else None,
        "candidate": candidate,
        "probe": probe,
        "score": score,
        "status": "PASS" if probe.get("passed") else "DEGRADED",
        "promotion_decision": score["verdict"],
        "visible_response": (
            f"Indy_READs: studied {adapter.label} via {query}; "
            f"found {len(items)} item(s), scored the board, and wrote the ledger."
        ),
    }
    receipt["receipt_path"] = rel(receipt_path)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    db_result: dict[str, str] = {}
    try:
        with psycopg.connect(db_url(args.database_url), row_factory=dict_row) as conn:
            db_result = write_db_rows(
                conn,
                run_id=run_id,
                text=text,
                adapter_kind=adapter_kind,
                adapter_label=adapter.label,
                items=items,
                focus=focus,
                board=board,
                candidate=candidate,
                probe=probe,
                score=score,
                receipt_path=receipt["receipt_path"],
            )
    except Exception as exc:
        receipt["db_error"] = f"{type(exc).__name__}:{exc}"
        receipt["status"] = "DEGRADED"
    receipt["db_write"] = db_result
    receipt["visible_response"] = {
        "summary": receipt["visible_response"],
        "work_order_id": db_result.get("work_order_uuid", ""),
        "work_receipt_id": db_result.get("work_receipt_uuid", ""),
        "attempt_id": db_result.get("work_order_uuid", ""),
        "raw_artifact_id": db_result.get("raw_artifact_uuid", ""),
        "artifact": rel(focus.url if focus else adapter_kind),
        "probe": probe.get("query") or query,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url")
    ap.add_argument("--text", default="")
    ap.add_argument("--source", default="auto", help="source adapter: auto, hn, arxiv, reddit, github")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--run-id")
    ap.add_argument("--receipt-dir", default=str(OUT))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)
    receipt = run(args)
    if args.json:
        print(json.dumps(receipt, sort_keys=True, default=str))
    else:
        print(f"RECEIPT_PATH={receipt['receipt_path']}")
        print(f"SOURCE={receipt['status']}")
        print(json.dumps(receipt, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
