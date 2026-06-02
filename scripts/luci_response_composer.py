#!/usr/bin/env python3
"""LUCI response composer: fast signal + template + math + quotes + slow lane + review.

This is the shared class-handler for the operator surface. It takes already
computed routing/context and deterministically weaves the visible response into
ordered lanes so the system owns composition instead of ad hoc prompt glue.
"""
from __future__ import annotations

from functools import lru_cache
import importlib
import json
import os
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row

from scripts.lucidota_indy_corpus import build_corpus


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", (text or "").lower())


@lru_cache(maxsize=1)
def _indy_units() -> tuple[dict[str, Any], ...]:
    corpus = build_corpus()
    units = corpus.get("units") or []
    return tuple(u for u in units if isinstance(u, dict))


def pick_quotes(text: str, *, limit: int = 3, units: Iterable[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    haystack = set(_tokens(text))
    scored: list[tuple[int, dict[str, Any]]] = []
    for unit in list(units or _indy_units()):
        label = str(unit.get("label") or "")
        unit_text = str(unit.get("text") or "")
        source = str(unit.get("source") or "")
        needle = set(_tokens(unit_text)) | set(_tokens(label)) | set(_tokens(source))
        score = len(haystack & needle)
        if score > 0:
            scored.append((score, unit))
    if not scored:
        scored = [(0, unit) for unit in list(units or _indy_units())[:limit]]
    scored.sort(key=lambda item: (-item[0], str(item[1].get("label") or ""), str(item[1].get("text") or "")))
    picked: list[dict[str, Any]] = []
    for _, unit in scored[:limit]:
        picked.append(
            {
                "lane": "quotes",
                "label": unit.get("label") or "indy",
                "text": str(unit.get("text") or "")[:240],
                "source": unit.get("source") or "",
            }
        )
    return picked


def _latest_json_receipt(dir_name: str) -> dict[str, Any] | None:
    root = Path(__file__).resolve().parents[1] / "05_OUTPUTS" / dir_name
    if not root.exists():
        return None
    files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data["_receipt_file"] = str(path.relative_to(root.parent.parent))
                return data
        except Exception:
            continue
    return None


def _recent_self_receipts(context: dict[str, Any]) -> list[dict[str, Any]]:
    supplied = context.get("recent_self_receipts")
    if isinstance(supplied, list):
        return [row for row in supplied if isinstance(row, dict)]
    receipts = _recent_self_receipts_from_db(context)
    if receipts:
        return receipts
    receipts = []
    for label, dir_name in [
        ("attempt", "luci_attempt_engine"),
        ("learning", "luci_learning"),
        ("source", "luci_source"),
        ("delegate", "luci_delegate"),
    ]:
        receipt = _latest_json_receipt(dir_name)
        if not receipt:
            continue
        status = str(receipt.get("status") or receipt.get("verdict") or receipt.get("promotion_decision") or "unknown")
        note = (
            receipt.get("visible_response", {}).get("summary")
            if isinstance(receipt.get("visible_response"), dict)
            else receipt.get("summary")
        )
        receipts.append(
            {
                "label": label,
                "status": status,
                "note": str(note or "")[:180],
                "receipt_path": str(receipt.get("receipt_path") or receipt.get("_receipt_file") or ""),
                "source": dir_name,
            }
        )
    return receipts


def _recent_self_receipts_from_db(context: dict[str, Any], *, limit: int = 4) -> list[dict[str, Any]]:
    db_url = (
        context.get("database_url")
        or os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL")
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
    )
    if not db_url:
        return []
    try:
        with psycopg.connect(db_url, connect_timeout=3, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  wr.work_receipt_uuid::text AS work_receipt_uuid,
                  wr.receipt_path AS receipt_path,
                  wr.verdict AS verdict,
                  wr.detail AS receipt_detail,
                  wo.work_order_uuid::text AS work_order_uuid,
                  wo.work_kind AS work_kind,
                  wo.status AS work_status,
                  wo.payload AS work_payload,
                  mi.model_name AS model_name,
                  mi.raw_output AS model_raw_output,
                  mi.detail AS model_detail
                FROM lucidota_control.work_receipt wr
                JOIN lucidota_control.work_order wo ON wo.work_order_uuid = wr.work_order_uuid
                LEFT JOIN lucidota_control.model_invocation mi ON mi.event_id = wr.event_id
                ORDER BY wr.created_at DESC
                LIMIT %s
                """,
                (max(1, limit * 3),),
            )
            rows = cur.fetchall()
    except Exception:
        return []

    receipts: list[dict[str, Any]] = []
    for row in rows:
        work_kind = str(row.get("work_kind") or "")
        label = "attempt"
        if "learning" in work_kind:
            label = "learning"
        elif "source" in work_kind:
            label = "source"
        elif "delegate" in work_kind:
            label = "delegate"
        elif "provider" in work_kind or "model" in work_kind:
            label = "model"
        detail = row.get("receipt_detail") or {}
        note = ""
        if isinstance(detail, dict):
            note = str(detail.get("summary") or detail.get("note") or detail.get("work_kind") or "")
        if not note:
            model_detail = row.get("model_detail") or {}
            if isinstance(model_detail, dict):
                note = str(model_detail.get("summary") or model_detail.get("note") or "")
        if not note:
            raw_output = str(row.get("model_raw_output") or "")
            if raw_output.strip():
                note = raw_output
        if not note:
            payload = row.get("work_payload") or {}
            if isinstance(payload, dict):
                note = str(payload.get("summary") or payload.get("prompt_preview") or payload.get("text_preview") or "")
        if not note:
            model_name = str(row.get("model_name") or "")
            if model_name:
                note = f"{model_name} invocation"
        receipts.append(
            {
                "label": label,
                "status": str(row.get("verdict") or row.get("work_status") or "unknown"),
                "note": note[:180],
                "receipt_path": str(row.get("receipt_path") or ""),
                "source": f"db:{work_kind or 'work'}",
                "work_order_id": str(row.get("work_order_uuid") or ""),
                "work_receipt_id": str(row.get("work_receipt_uuid") or ""),
            }
        )
        if len(receipts) >= limit:
            break
    return receipts


def _file_excerpt(path: Path, *, limit: int = 260) -> str:
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return " ".join(text.split())[:limit]


def _recent_system_maps(context: dict[str, Any]) -> list[dict[str, Any]]:
    supplied = context.get("system_maps")
    if isinstance(supplied, list):
        return [row for row in supplied if isinstance(row, dict)]
    root = Path(__file__).resolve().parents[1]
    sources = [
        ("handoff", root / "GOALS" / "CURRENT_HANDOFF.md"),
        ("agent_policy", root / "GOALS" / "AGENT_ORCHESTRATION_POLICY.md"),
        ("routing_fabric", root / "00_PROJECT_BRAIN" / "ACTIVE_SPEC" / "LUCI_ROUTING_FABRIC_DECISIONS_20260601.md"),
    ]
    maps: list[dict[str, Any]] = []
    for label, path in sources:
        excerpt = _file_excerpt(path)
        if not excerpt:
            continue
        maps.append(
            {
                "label": label,
                "path": str(path.relative_to(root)),
                "excerpt": excerpt,
            }
        )
    return maps


def build_math_lane(context: dict[str, Any]) -> dict[str, Any]:
    text = str(context.get("text") or "")
    terms = list(context.get("ontology_terms") or [])
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    digits = re.findall(r"\d+", text)
    return {
        "lane": "math",
        "text": (
            f"math: chars={len(text)} words={len(words)} "
            f"unique_terms={len(set(_tokens(text)))} ontology_terms={len(terms)} digits={len(digits)}"
        ),
        "values": {
            "chars": len(text),
            "words": len(words),
            "unique_terms": len(set(_tokens(text))),
            "ontology_terms": len(terms),
            "digits": len(digits),
        },
    }


def build_template_lane(context: dict[str, Any]) -> dict[str, Any]:
    intent = str(context.get("intent") or "ops")
    lane = str(context.get("lane") or "FASTLANE")
    rendered = str(context.get("language_rendered") or context.get("rendered") or "").strip()
    text = str(context.get("text") or "")
    if not rendered:
        rendered = f"INTENT={intent} LANE={lane} TASK={text[:160]}"
    return {
        "lane": "template",
        "text": rendered,
        "values": {"intent": intent, "lane": lane},
    }


def build_fast_lane(context: dict[str, Any]) -> dict[str, Any]:
    intent = str(context.get("intent") or "ops")
    lane = str(context.get("lane") or "FASTLANE")
    user = str(context.get("operator") or context.get("user") or "operator")
    return {
        "lane": "fast",
        "text": f"Indy_READs: got it, {user} — routed {intent} through {lane}.",
    }


def build_review_lane(context: dict[str, Any]) -> dict[str, Any]:
    review = context.get("external_review")
    provider_lanes = context.get("provider_lanes") or {}
    groq_status = ((provider_lanes.get("groq") or {}).get("status")) if isinstance(provider_lanes, dict) else None
    vibes_status = ((provider_lanes.get("vibes") or {}).get("status")) if isinstance(provider_lanes, dict) else None
    if isinstance(review, dict):
        groq_findings = ((review.get("groq") or {}).get("findings") or [])
        vibes_findings = ((review.get("vibes") or {}).get("findings") or [])
        parts = []
        if groq_findings:
            parts.append("Groq: " + "; ".join(str(x) for x in groq_findings[:2]))
        if vibes_findings:
            parts.append("Vibes: " + "; ".join(str(x) for x in vibes_findings[:2]))
        if not parts:
            parts.append("Groq/Vibes review lane executed but no findings were returned.")
        return {"lane": "review", "text": "review: " + " | ".join(parts), "values": review}
    note = "review: Groq/Vibes review lane available and receipt-backed"
    if groq_status or vibes_status:
        note += f" (groq={groq_status or 'unknown'}, vibes={vibes_status or 'unknown'})"
    note += "; deferred unless the operator asks for an external audit."
    return {"lane": "review", "text": note, "values": {"groq": groq_status, "vibes": vibes_status}}


def build_map_lane(context: dict[str, Any]) -> dict[str, Any]:
    maps = _recent_system_maps(context)
    if not maps:
        return {
            "lane": "map",
            "text": "map: no recent system maps found; operating from live router state only.",
            "values": [],
        }
    summary_bits = []
    for item in maps[:3]:
        excerpt = item.get("excerpt") or ""
        summary_bits.append(f"{item['label']} -> {excerpt}")
    return {
        "lane": "map",
        "text": "map: " + " | ".join(summary_bits),
        "values": maps,
    }


def build_improve_lane(context: dict[str, Any]) -> dict[str, Any]:
    receipts = _recent_self_receipts(context)
    if not receipts:
        return {
            "lane": "improve",
            "text": "improve: no recent self-improvement receipts found; the machine is live but has no fresh local learning loop evidence yet.",
            "values": [],
        }
    parts = []
    for receipt in receipts[:4]:
        note = receipt.get("note") or ""
        if note:
            parts.append(f"{receipt['label']}={receipt['status']} {note}")
        else:
            parts.append(f"{receipt['label']}={receipt['status']}")
    return {
        "lane": "improve",
        "text": "improve: recent self-learning receipts -> " + " | ".join(parts),
        "values": receipts,
    }


def build_slow_lane(context: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    text = str(context.get("text") or "")
    adhd_context = context.get("adhd_slow_lane")
    if isinstance(adhd_context, dict) and adhd_context.get("synthesis"):
        return {
            "lane": "slow",
            "text": "slow: " + str(adhd_context["synthesis"]),
            "values": adhd_context,
        }

    candidate_branches = [
        {"frame": "evidence", "lane": "deepseek", "text": segments[0]["text"] if segments else text},
        {"frame": "template", "lane": "mamba_cpu", "text": segments[1]["text"] if len(segments) > 1 else text},
        {"frame": "math", "lane": "bonsai", "text": segments[2]["text"] if len(segments) > 2 else text},
        {"frame": "quotes", "lane": "deepseek", "text": "\n".join(seg["text"] for seg in segments if seg["lane"] == "quotes") or text},
    ]

    try:
        slow = importlib.import_module("adhd_slow_lane_divergence")
        scored = slow.score_branches(candidate_branches)
        kept = slow.prune(scored, survivors=2)
        synthesis = ""
        if kept:
            synthesis = kept[0].get("text") or kept[0].get("frame") or ""
        if not synthesis and scored:
            synthesis = scored[0].get("text") or scored[0].get("frame") or ""
        if not synthesis:
            synthesis = " ".join(branch["text"] for branch in candidate_branches[:2]).strip()
        return {
            "lane": "slow",
            "text": f"slow: {synthesis}",
            "values": {"scored": scored, "survivors": kept, "source": "adhd_slow_lane_divergence"},
        }
    except Exception:
        synthesis = "Resource management, learning, precognition, and hyperplexing belong in the slow lane."
        return {
            "lane": "slow",
            "text": f"slow: {synthesis}",
            "values": {"source": "deterministic_fallback"},
        }


def compose_response(context: dict[str, Any]) -> dict[str, Any]:
    fast = build_fast_lane(context)
    template = build_template_lane(context)
    map_lane = build_map_lane(context)
    math = build_math_lane(context)
    quotes = pick_quotes(str(context.get("text") or ""), units=context.get("indy_corpus_units"))
    quote_segment = {
        "lane": "quotes",
        "text": "quote: " + (" | ".join(f"[{q['label']}] {q['text']}" for q in quotes) if quotes else "no matching quote candidates"),
        "values": quotes,
    }
    improve = build_improve_lane(context)
    review = build_review_lane(context)
    slow = build_slow_lane(context, [fast, template, map_lane, math, quote_segment, improve, review])
    segments = [fast, template, map_lane, math, quote_segment, improve, slow, review]
    summary = "\n".join(segment["text"] for segment in segments)
    visible_response = {
        "summary": summary,
        "next": str(context.get("next_hint") or ("receipt written; slow work queued" if str(context.get("lane") or "").upper() == "SLOWLANE" else "fast route completed")),
        "work_order_id": str(context.get("work_order_id") or ""),
        "attempt_id": str(context.get("attempt_id") or ""),
        "work_receipt_id": str(context.get("work_receipt_id") or ""),
        "artifact": str(context.get("artifact") or ""),
        "receipt_path": str(context.get("receipt_path") or ""),
        "segments": segments,
    }
    return {
        "schema": "lucidota.luci.response_composer.v1",
        "generated_at": now(),
        "intent": context.get("intent") or "ops",
        "lane": context.get("lane") or "FASTLANE",
        "segments": segments,
        "composition": {
            "json_safe": True,
            "lane_count": len(segments),
            "quote_count": len(quotes),
            "improve_count": len(improve.get("values") or []),
            "map_count": len(map_lane.get("values") or []),
            "has_slow_lane": True,
            "has_review_lane": True,
            "has_improve_lane": True,
            "has_map_lane": True,
            "source": "system_owned_composition",
        },
        "visible_response": visible_response,
    }
