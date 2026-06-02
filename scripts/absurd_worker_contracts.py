#!/usr/bin/env python3
"""ABSURD worker contract validation helpers.

A queued job is executable only when a live contract row names its queue and
job kind. This is intentionally small and boring so worker code can call it
before side effects.
"""
from __future__ import annotations

import json
import hashlib
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from ALGOS.decision_hygiene import counts as decision_counts, score_features
except Exception:  # pragma: no cover - deterministic fallback when ALGOS isn't on sys.path
    import re

    EVIDENCE_RE = re.compile(r"\b(?:evidence|verified|confirmed|confirm|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|facts|check|checked|audit)\b", re.I)
    PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
    DELAY_RE = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de-escalate|not now|before i|first|after|review)\b", re.I)
    SUPPORT_RE = re.compile(r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b", re.I)
    BOUNDARY_RE = re.compile(r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b", re.I)
    OUTCOME_RE = re.compile(r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b", re.I)
    IMPULSIVE_RE = re.compile(r"\b(?:rage|impulsive|panic|spiral|doom|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b", re.I)
    SCARCITY_RE = re.compile(r"\b(?:broke|homeless|last\\s+\\$?\\d+|\\$\\s?\\d+\\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b", re.I)
    RISK_RE = re.compile(r"\b(?:kill|die|suicide|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b", re.I)

    def decision_counts(text: str) -> dict[str, int]:
        return {
            "evidence_count": len(EVIDENCE_RE.findall(text or "")),
            "planning_count": len(PLANNING_RE.findall(text or "")),
            "delay_count": len(DELAY_RE.findall(text or "")),
            "support_count": len(SUPPORT_RE.findall(text or "")),
            "boundary_count": len(BOUNDARY_RE.findall(text or "")),
            "outcome_count": len(OUTCOME_RE.findall(text or "")),
            "impulsive_count": len(IMPULSIVE_RE.findall(text or "")),
            "scarcity_count": len(SCARCITY_RE.findall(text or "")),
            "risk_count": len(RISK_RE.findall(text or "")),
        }

    def score_features(c: dict[str, int]) -> tuple[int, str]:
        positive = (
            c["evidence_count"] * 1600
            + c["planning_count"] * 1200
            + c["delay_count"] * 1400
            + c["support_count"] * 1000
            + c["boundary_count"] * 1200
            + c["outcome_count"] * 800
        )
        negative = c["impulsive_count"] * 1500 + c["scarcity_count"] * 700 + c["risk_count"] * 1200
        score = max(-10000, min(10000, positive - negative))
        if c["risk_count"] and score < 2500:
            label = "critical_risk_or_pain_signal"
        elif score >= 7000:
            label = "high_decision_hygiene"
        elif score >= 3000:
            label = "improving_decision_hygiene"
        elif score <= -2500:
            label = "strained_decision_context"
        else:
            label = "neutral_or_unclear"
        return score, label


DEFAULT_HYGIENE_MIN_SCORE = 250
DEFAULT_REQUIRED_PAYLOAD_HINTS = (
    "ok",
    "status",
    "outcome",
    "command",
    "returncode",
    "error",
    "error_kind",
    "result",
    "health",
    "workflow_event_id",
    "event_id",
    "run_uuid",
    "command_uuid",
)


@dataclass(frozen=True)
class WorkerContractDecision:
    ok: bool
    error_kind: str = ""
    error_message: str = ""
    worker_key: str | None = None
    queue_name: str | None = None
    job_kind: str | None = None
    script_path: str | None = None
    allowed_job_kinds: tuple[str, ...] = ()

    def as_result(self) -> dict[str, Any]:
        return asdict(self)


def gate_worker_payload_hygiene(
    payload: Any,
    *,
    queue_name: str | None = None,
    worker_key: str | None = None,
    job_kind: str | None = None,
    required_keys: tuple[str, ...] = DEFAULT_REQUIRED_PAYLOAD_HINTS,
    min_score: int = DEFAULT_HYGIENE_MIN_SCORE,
) -> tuple[bool, dict[str, Any]]:
    """Return PASS only when payload shape/text has deterministic hygiene.

    This is intentionally strict and deterministic: payload must be structured,
    non-empty (for object/list), and pass a minimum decision-hygiene score.
    """
    if not isinstance(payload, (dict, list)):
        return False, {
            "hygiene": {
                "ok": False,
                "score": None,
                "label": "invalid_payload_type",
                "required_keys": [],
                "missing_required_keys": list(required_keys),
                "queue_name": queue_name,
                "worker_key": worker_key,
                "job_kind": job_kind,
            },
            "error": "job_result_payload_type_invalid",
            "error_message": "payload for success must be dict/list",
        }
    if isinstance(payload, list) and not payload:
        return False, {
            "hygiene": {
                "ok": False,
                "score": 0,
                "label": "empty_payload",
                "required_keys": list(required_keys),
                "missing_required_keys": list(required_keys),
                "queue_name": queue_name,
                "worker_key": worker_key,
                "job_kind": job_kind,
            },
            "error": "job_result_payload_empty",
            "error_message": "payload for success cannot be empty",
        }

    payload_text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    score, label = score_features(decision_counts(payload_text))
    present: list[str] = []
    missing: list[str] = []
    if required_keys:
        present = [key for key in required_keys if isinstance(payload, dict) and key in payload]
        missing = [key for key in required_keys if key not in present]
    if score < min_score:
        return False, {
            "hygiene": {
                "ok": False,
                "score": score,
                "label": label,
                "required_keys": list(required_keys),
                "missing_required_keys": missing,
                "queue_name": queue_name,
                "worker_key": worker_key,
                "job_kind": job_kind,
            },
            "error": "job_result_hygiene_failed",
            "error_message": f"decision_hygiene_score_below_threshold:{score}<{min_score}",
        }
    if required_keys and not present:
        return False, {
            "hygiene": {
                "ok": False,
                "score": score,
                "label": label,
                "required_keys": list(required_keys),
                "missing_required_keys": missing,
                "queue_name": queue_name,
                "worker_key": worker_key,
                "job_kind": job_kind,
            },
            "error": "job_result_hygiene_failed",
            "error_message": "no_structural_hint_keys_present",
        }

    return True, {
        "hygiene": {
            "ok": True,
            "score": score,
            "label": label,
            "required_keys": list(required_keys),
            "missing_required_keys": missing,
            "queue_name": queue_name,
            "worker_key": worker_key,
            "job_kind": job_kind,
        },
        "error": "none",
        "error_message": "",
    }


def _split_job_kinds(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, list):
        return tuple(str(x).strip() for x in raw if str(x).strip())
    text = str(raw or "")
    return tuple(x.strip() for x in text.replace(",", "|").split("|") if x.strip())


def _script_exists(script_path: str) -> bool:
    p = Path(script_path)
    if not p.is_absolute():
        p = ROOT / p
    return p.exists() and p.is_file()


def _sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def record_worker_contract_rejection(
    cur: Any,
    *,
    job_uuid: str,
    queue_name: str,
    payload: Any,
    contract: WorkerContractDecision,
    event_source: str,
) -> tuple[dict[str, Any], str]:
    """Persist a fatal contract rejection without running the worker handler."""
    error_kind = contract.error_kind or "worker_contract_rejected"
    gate_result = {"error": "worker_contract_rejected", "worker_contract": contract.as_result()}
    payload_obj = payload or {}
    cur.execute(
        """
        UPDATE lucidota_control.absurd_queue_job
        SET status='failed', result=%s::jsonb, completed_at=now(), updated_at=now(), last_error=%s
        WHERE job_uuid=%s::uuid
        """,
        (json.dumps(gate_result, default=str), error_kind, job_uuid),
    )
    cur.execute(
        "INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, event_source, detail) VALUES (%s::uuid,%s,'failed',%s,%s::jsonb)",
        (job_uuid, queue_name, event_source, json.dumps(gate_result, default=str)),
    )
    cur.execute(
        """
        INSERT INTO lucidota_control.absurd_queue_dead_letter
          (job_uuid, queue_name, workflow_name, job_kind, idempotency_key, error_kind, error_message, attempt_count, payload_sha256, context)
        SELECT job_uuid, queue_name, workflow_name, job_kind, idempotency_key, %s, %s, attempt_count, %s, %s::jsonb
        FROM lucidota_control.absurd_queue_job WHERE job_uuid=%s::uuid
        ON CONFLICT (job_uuid) WHERE resolved=false DO UPDATE SET error_message=EXCLUDED.error_message,last_seen_at=now(),context=EXCLUDED.context
        """,
        (error_kind, contract.error_message, _sha256_obj(payload_obj), json.dumps(gate_result, default=str), job_uuid),
    )
    return gate_result, error_kind


@dataclass(frozen=True)
class SchemaVersionDecision:
    ok: bool
    error_kind: str = ""
    error_message: str = ""
    worker_key: str | None = None
    worker_version: str | None = None
    job_version: str | None = None
    is_compatible: bool = True

    def as_result(self) -> dict[str, Any]:
        return asdict(self)


def validate_schema_contract_version(
    cur: Any,
    *,
    worker_key: str,
    payload: dict[str, Any],
    queue_name: str,
    job_kind: str,
) -> SchemaVersionDecision:
    """Check schema contract version compatibility between worker and job.

    Returns PASS if:
    - No version record exists for worker (backwards compatible)
    - No schema_contract_version in job payload
    - Compatibility matrix confirms (job_version, worker_version) is_compatible=true

    Returns FAIL if compatibility matrix says is_compatible=false.
    Fails open (returns PASS) on any DB error.
    """
    try:
        # Look up worker's latest version record
        cur.execute(
            """
            SELECT schema_contract_version, compatible_branches
            FROM lucidota_control.worker_contract_version
            WHERE worker_key=%s
            ORDER BY declared_at DESC
            LIMIT 1
            """,
            (worker_key,),
        )
        version_row = cur.fetchone()
        
        # No version record = backwards compatible, proceed
        if not version_row:
            return SchemaVersionDecision(True, worker_key=worker_key)
        
        worker_version = version_row[0] if not isinstance(version_row, dict) else version_row.get("schema_contract_version")
        
        # No worker version = backwards compatible, proceed
        if not worker_version:
            return SchemaVersionDecision(True, worker_key=worker_key)
        
        job_version = payload.get("schema_contract_version")
        
        # No job version = backwards compatible, proceed
        if not job_version:
            return SchemaVersionDecision(True, worker_key=worker_key, worker_version=str(worker_version))
        
        # Check compatibility matrix
        cur.execute(
            """
            SELECT is_compatible
            FROM lucidota_control.schema_compatibility_matrix
            WHERE job_version=%s AND worker_version=%s
            LIMIT 1
            """,
            (str(job_version), str(worker_version)),
        )
        matrix_row = cur.fetchone()
        
        if matrix_row:
            is_compatible = matrix_row[0] if not isinstance(matrix_row, dict) else matrix_row.get("is_compatible")
            if is_compatible is False:
                return SchemaVersionDecision(
                    False,
                    error_kind="schema_version_mismatch",
                    error_message=f"Job version {job_version} incompatible with worker version {worker_version}",
                    worker_key=worker_key,
                    worker_version=str(worker_version),
                    job_version=str(job_version),
                    is_compatible=False,
                )
        
        # No matrix entry = assume compatible (fail open)
        return SchemaVersionDecision(
            True,
            worker_key=worker_key,
            worker_version=str(worker_version),
            job_version=str(job_version),
            is_compatible=True,
        )
    except Exception as e:
        # DB unavailable or any error: fail open, log warning, proceed
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"schema_contract_version check failed for worker {worker_key}: {e}")
        return SchemaVersionDecision(True, worker_key=worker_key)


def validate_worker_contract(cur: Any, *, queue_name: str, job_kind: str, worker_key: str | None = None) -> WorkerContractDecision:
    """Return PASS only when the DB contract explicitly allows queue/job_kind.

    Missing contract tables are blockers. This prevents the fallback behavior
    where bespoke workers silently invent runtime contracts at dequeue time.
    """
    cur.execute("SELECT to_regclass('lucidota_control.absurd_worker_contract')")
    row = cur.fetchone()
    exists = row[0] if not isinstance(row, dict) else row.get("to_regclass")
    if exists is None:
        return WorkerContractDecision(False, "worker_contract_table_missing", "lucidota_control.absurd_worker_contract is missing", queue_name=queue_name, job_kind=job_kind)

    params: list[Any] = [queue_name]
    worker_clause = ""
    if worker_key:
        worker_clause = " AND worker_key=%s"
        params.append(worker_key)
    cur.execute(
        f"""
        SELECT worker_key, queue_name, script_path, input_contract, canonical_graph_write_allowed, status
        FROM lucidota_control.absurd_worker_contract
        WHERE queue_name=%s{worker_clause}
          AND status IN ('implemented','verified')
        ORDER BY updated_at DESC, created_at DESC
        """,
        params,
    )
    rows = cur.fetchall()
    if not rows:
        return WorkerContractDecision(False, "worker_contract_not_found", f"no active contract for queue {queue_name}", queue_name=queue_name, job_kind=job_kind)

    seen: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            wk, qn, sp, contract = row["worker_key"], row["queue_name"], row["script_path"], row["input_contract"]
        else:
            wk, qn, sp, contract = row[0], row[1], row[2], row[3]
        if isinstance(contract, str):
            try:
                contract = json.loads(contract)
            except Exception:
                contract = {}
        allowed = _split_job_kinds((contract or {}).get("job_kind"))
        seen.extend(allowed)
        if job_kind in allowed:
            if not _script_exists(str(sp)):
                return WorkerContractDecision(False, "worker_contract_script_missing", f"contract script missing: {sp}", worker_key=str(wk), queue_name=str(qn), job_kind=job_kind, script_path=str(sp), allowed_job_kinds=allowed)
            return WorkerContractDecision(True, worker_key=str(wk), queue_name=str(qn), job_kind=job_kind, script_path=str(sp), allowed_job_kinds=allowed)

    return WorkerContractDecision(False, "job_kind_not_in_worker_contract", f"{job_kind} not allowed for {queue_name}; allowed={sorted(set(seen))}", queue_name=queue_name, job_kind=job_kind, allowed_job_kinds=tuple(sorted(set(seen))))
