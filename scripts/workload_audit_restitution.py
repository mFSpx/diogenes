#!/usr/bin/env python3
"""Backfill the DB-backed workload audit ledger from receipt evidence."""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import tomllib

import psycopg
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "KRAMPUSCHEWING" / "Script_Corpses"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

from swarm_usage_ledger_core import discover_receipts, load, rel  # noqa: E402

BASE = Path("05_OUTPUTS/model_invocations")
REPORT_DIR = ROOT / "05_OUTPUTS/goals"
NAMESPACE = uuid.UUID("7b201c9d-9ff1-4583-9948-e85092a22ffc")
VIBE_HOME = Path.home() / ".vibe"

ONTOLOGY_INDEX: dict[str, Any] = {
    "tier_1_go": {"universal_primitives": ["telemetry", "duplex", "allocation"]},
    "tier_2_stable": {
        "code_ontology": ["token_ledger_surface", "local_model_sovereignty"],
        "indy_ontology": ["production_prize_race", "investigative_skepticism", "clue_tracking"],
        "unhinged_414": ["POST_NUT_CLARITY_ACCOUNTING", "ANTI_HANDWAVE_PROTOCOL"],
    },
    "tier_3_fungible_souls": {
        "percyphon_village_seed": "0x7b201c9d9ff14583",
        "active_telemetry_tokens": ["laptop_vs_cloud_race", "budget_pressure_rung_3"],
    },
    "tier_4_now_ontology": {
        "vuuid_source": "129-row-scorecard-flash",
        "procedural_flash_100": ["SCOREBOARD", "NOIR_METRICS", "SILICON_SPEEDOMETER"],
    },
}

FUNCTIONALITY_EXPLANATION = (
    "Tracks receipt-backed workload accounting plus explicit UNKNOWN debt "
    "so no actor, provider, or model is credited without DB truth."
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def classify_actor_class(provider: str, model: str, payload: dict[str, Any]) -> str:
    provider = provider.lower()
    model_lower = model.lower()
    if provider == "local" or "llama.cpp" in str(payload.get("backend") or ""):
        return "local_llm"
    if provider == "groq":
        return "groq"
    if provider == "gemini":
        if "paid" in model_lower:
            return "gemini_paid"
        return "gemini"
    if provider == "vibe":
        return "vibe"
    return "unknown"


def caller_for_receipt(payload: dict[str, Any], actor_class: str) -> str:
    caller = str(payload.get("caller") or payload.get("caller_class") or payload.get("owner") or "").strip().lower()
    if caller in {"codex", "indy_reads", "operator", "daemon", "unknown"}:
        return caller
    if actor_class in {"codex_main", "codex_agent"}:
        return "codex"
    if actor_class == "indy_reads":
        return "indy_reads"
    return "unknown"


def token_counts(payload: dict[str, Any]) -> tuple[int | None, int | None, str]:
    token_accounting = payload.get("token_accounting")
    if isinstance(token_accounting, dict):
        prompt = token_accounting.get("prompt_tokens")
        completion = token_accounting.get("completion_tokens")
        total = token_accounting.get("total_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            return prompt, completion, "local_counter"
        if isinstance(total, int):
            return None, None, "local_counter"

    usage = payload.get("usage")
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            return prompt, completion, "provider_api"
        if isinstance(total, int) and isinstance(prompt, int):
            return prompt, max(total - prompt, 0), "provider_api"
        tokens = usage.get("tokens")
        if isinstance(tokens, dict):
            prompt = tokens.get("input_tokens") or tokens.get("prompt_tokens")
            completion = tokens.get("output_tokens") or tokens.get("completion_tokens")
            if isinstance(prompt, int) and isinstance(completion, int):
                return prompt, completion, "provider_api"
        billed = usage.get("billed_units")
        if isinstance(billed, dict):
            prompt = billed.get("input_tokens") or billed.get("prompt_tokens")
            completion = billed.get("output_tokens") or billed.get("completion_tokens")
            if isinstance(prompt, int) and isinstance(completion, int):
                return prompt, completion, "provider_api"

    raw_response = payload.get("raw_response")
    if isinstance(raw_response, dict):
        usage = raw_response.get("usage")
        if isinstance(usage, dict):
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            total = usage.get("total_tokens")
            if isinstance(prompt, int) and isinstance(completion, int):
                return prompt, completion, "provider_api"
            if isinstance(total, int) and isinstance(prompt, int):
                return prompt, max(total - prompt, 0), "provider_api"
            tokens = usage.get("tokens")
            if isinstance(tokens, dict):
                prompt = tokens.get("input_tokens") or tokens.get("prompt_tokens")
                completion = tokens.get("output_tokens") or tokens.get("completion_tokens")
                if isinstance(prompt, int) and isinstance(completion, int):
                    return prompt, completion, "provider_api"
        if payload.get("provider") == "gemini":
            usage = payload.get("response", {}).get("usageMetadata") if isinstance(payload.get("response"), dict) else None
            if isinstance(usage, dict):
                prompt = usage.get("promptTokenCount")
                total = usage.get("totalTokenCount")
                if isinstance(prompt, int) and isinstance(total, int):
                    return prompt, max(total - prompt, 0), "provider_api"

    response = payload.get("response")
    if isinstance(response, dict):
        usage = response.get("usageMetadata")
        if isinstance(usage, dict):
            prompt = usage.get("promptTokenCount")
            total = usage.get("totalTokenCount")
            if isinstance(prompt, int) and isinstance(total, int):
                return prompt, max(total - prompt, 0), "provider_api"

    return None, None, "unknown"


def receipt_uuid_for(source_ref: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE, f"receipt:{source_ref}")


def workload_uuid_for(actor_id: str, source_ref: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE, f"workload:{actor_id}:{source_ref}")


def evidence_refs_for(source_ref: str, extra_refs: list[str] | None = None) -> list[str]:
    refs = [source_ref]
    if extra_refs:
        refs.extend(extra_refs)
    seen: set[str] = set()
    ordered: list[str] = []
    for ref in refs:
        if ref not in seen:
            seen.add(ref)
            ordered.append(ref)
    return ordered


def discover_vibe_sessions() -> list[Path]:
    session_root = VIBE_HOME / "logs" / "session"
    if not session_root.exists():
        return []
    return sorted(session_root.glob("*/meta.json"))


def vibe_model_id() -> str:
    config_path = VIBE_HOME / "config.toml"
    if not config_path.exists():
        return "mistral-medium-3.5"
    try:
        config = load_toml(config_path)
    except Exception:
        return "mistral-medium-3.5"
    return str(config.get("active_model") or "mistral-medium-3.5")


def receipt_rows(receipt_paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in receipt_paths:
        payload = load(path)
        provider = str(payload.get("provider") or "unknown")
        model = str(payload.get("model") or "")
        actor_class = classify_actor_class(provider, model, payload)
        caller = caller_for_receipt(payload, actor_class)
        prompt_in, completion_out, token_source = token_counts(payload)
        status = str(payload.get("status") or "").upper()
        proof_status = "PROVEN" if status == "PASS" and prompt_in is not None and completion_out is not None else "PARTIAL"
        source_ref = rel(path)
        receipt_uuid = receipt_uuid_for(source_ref)
        actor_id = f"{provider}:{model}:{payload.get('receipt_key') or path.stem}"
        action_summary = f"{provider} {model} {status.lower()} receipt"
        if payload.get("prompt_text"):
            action_summary = f"{provider} {model}: {str(payload['prompt_text']).strip()[:120]}"
        evidence_refs = evidence_refs_for(source_ref, [str(payload.get("report_path") or source_ref)])
        rows.append({
            "workload_audit_uuid": workload_uuid_for(actor_id, source_ref),
            "actor_id": actor_id,
            "actor_class": actor_class,
            "caller": caller,
            "provider": provider,
            "model_id": model,
            "action_summary": action_summary,
            "tokens_in": prompt_in,
            "tokens_out": completion_out,
            "token_source": token_source,
            "receipt_uuid": receipt_uuid,
            "evidence_refs": Jsonb(evidence_refs),
            "proof_status": proof_status,
            "debt_reason": "",
            "created_at": now(),
            "refreshed_at": now(),
            "functionality_explanation": FUNCTIONALITY_EXPLANATION,
            "ontology_index": Jsonb(ONTOLOGY_INDEX),
        })
    return rows


def debt_rows() -> list[dict[str, Any]]:
    debt_specs = [
        ("codex_main", "codex", "scripts/swarm_usage_ledger.py", "ledger wrapper and GOALS claims were not receipt-backed"),
        ("codex_agent", "codex", "GOALS/GOAL_LOG.md", "agent claims were not receipt-backed"),
        ("indy_reads", "indy_reads", "GOALS/CURRENT_HANDOFF.md", "indy claims were not receipt-backed"),
        ("gemini_paid", "unknown", "05_OUTPUTS/goals", "paid Gemini usage was not receipt-backed"),
        ("vibe", "unknown", "05_OUTPUTS/goals", "vibe lane usage was not receipt-backed"),
    ]
    rows: list[dict[str, Any]] = []
    for actor_class, caller, evidence_ref, reason in debt_specs:
        actor_id = f"debt:{actor_class}"
        rows.append({
            "workload_audit_uuid": workload_uuid_for(actor_id, evidence_ref),
            "actor_id": actor_id,
            "actor_class": actor_class,
            "caller": caller,
            "provider": "unknown",
            "model_id": "",
            "action_summary": "claimed or expected work not proven",
            "tokens_in": None,
            "tokens_out": None,
            "token_source": "unknown",
            "receipt_uuid": None,
            "evidence_refs": Jsonb(evidence_refs_for(evidence_ref, [
                "KRAMPUSCHEWING/Script_Corpses/swarm_usage_ledger_core.py",
                "GOALS/GOAL_LOG.md",
                "GOALS/CURRENT_HANDOFF.md",
            ])),
            "proof_status": "UNKNOWN",
            "debt_reason": "no receipt-backed workload/token evidence",
            "created_at": now(),
            "refreshed_at": now(),
            "functionality_explanation": FUNCTIONALITY_EXPLANATION,
            "ontology_index": Jsonb(ONTOLOGY_INDEX),
        })
    return rows


def vibe_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_id = vibe_model_id()
    for meta_path in discover_vibe_sessions():
        try:
            meta = load_json(meta_path)
        except Exception:
            continue
        stats = meta.get("stats") if isinstance(meta.get("stats"), dict) else {}
        prompt_tokens = stats.get("session_prompt_tokens")
        completion_tokens = stats.get("session_completion_tokens")
        session_id = str(meta.get("session_id") or meta_path.parent.name)
        title = str(meta.get("title") or "vibe session").strip()
        actor_id = f"vibe:{session_id}"
        rows.append({
            "workload_audit_uuid": workload_uuid_for(actor_id, rel(meta_path)),
            "actor_id": actor_id,
            "actor_class": "vibe",
            "caller": "codex",
            "provider": "vibe",
            "model_id": model_id,
            "action_summary": f"vibe cli session: {title}",
            "tokens_in": prompt_tokens if isinstance(prompt_tokens, int) else None,
            "tokens_out": completion_tokens if isinstance(completion_tokens, int) else None,
            "token_source": "receipt_file",
            "receipt_uuid": uuid.UUID(session_id),
            "evidence_refs": Jsonb(evidence_refs_for(
                rel(meta_path),
                [
                    rel(meta_path.parent / "messages.jsonl"),
                    rel(VIBE_HOME / "config.toml"),
                ],
            )),
            "proof_status": "PROVEN" if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int) else "PARTIAL",
            "debt_reason": "",
            "created_at": now(),
            "refreshed_at": now(),
            "functionality_explanation": FUNCTIONALITY_EXPLANATION,
            "ontology_index": Jsonb(ONTOLOGY_INDEX),
        })
    return rows


def rows_by_status(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        counts[row["proof_status"]] += 1
    return dict(counts)


def insert_rows(conn: psycopg.Connection[Any], rows: list[dict[str, Any]]) -> int:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO lucidota_audit.workload_audit_ledger (
                workload_audit_uuid,
                actor_id,
                actor_class,
                caller,
                provider,
                model_id,
                action_summary,
                tokens_in,
                tokens_out,
                token_source,
                receipt_uuid,
                evidence_refs,
                proof_status,
                debt_reason,
                created_at,
                refreshed_at,
                functionality_explanation,
                ontology_index
            ) VALUES (
                %(workload_audit_uuid)s,
                %(actor_id)s,
                %(actor_class)s,
                %(caller)s,
                %(provider)s,
                %(model_id)s,
                %(action_summary)s,
                %(tokens_in)s,
                %(tokens_out)s,
                %(token_source)s,
                %(receipt_uuid)s,
                %(evidence_refs)s,
                %(proof_status)s,
                %(debt_reason)s,
                %(created_at)s::timestamptz,
                %(refreshed_at)s::timestamptz,
                %(functionality_explanation)s,
                %(ontology_index)s
            )
            ON CONFLICT (workload_audit_uuid) DO UPDATE SET
                actor_id = EXCLUDED.actor_id,
                actor_class = EXCLUDED.actor_class,
                caller = EXCLUDED.caller,
                provider = EXCLUDED.provider,
                model_id = EXCLUDED.model_id,
                action_summary = EXCLUDED.action_summary,
                tokens_in = EXCLUDED.tokens_in,
                tokens_out = EXCLUDED.tokens_out,
                token_source = EXCLUDED.token_source,
                receipt_uuid = EXCLUDED.receipt_uuid,
                evidence_refs = EXCLUDED.evidence_refs,
                proof_status = EXCLUDED.proof_status,
                debt_reason = EXCLUDED.debt_reason,
                refreshed_at = EXCLUDED.refreshed_at,
                functionality_explanation = EXCLUDED.functionality_explanation,
                ontology_index = EXCLUDED.ontology_index
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill workload audit ledger rows and debt rows from receipt evidence.")
    ap.add_argument("--database-url", default=os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    receipt_paths = discover_receipts(root=ROOT)
    receipt_backfill = receipt_rows(receipt_paths)
    vibe_backfill = vibe_rows()
    debt_backfill = debt_rows()
    rows = receipt_backfill + vibe_backfill + debt_backfill
    status_counts = rows_by_status(rows)
    provider_counts = Counter(row["provider"] for row in rows)

    report = {
        "generated_at": now(),
        "database_url": "postgresql:///lucidota_state" if args.database_url.endswith("/lucidota_state") else "<redacted>",
        "receipt_count": len(receipt_paths),
        "vibe_session_count": len(vibe_backfill),
        "row_count": len(rows),
        "status_counts": dict(status_counts),
        "provider_counts": dict(provider_counts),
        "report_path": None,
    }

    if not args.dry_run:
        with psycopg.connect(args.database_url, autocommit=False) as conn:
            insert_rows(conn, rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"workload_audit_restitution_{stamp()}.json"
    report["report_path"] = rel(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(report, sort_keys=True))
    print("REPORT_PATH=" + rel(report_path))
    print("WORKLOAD_AUDIT_RESTITUTION=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
