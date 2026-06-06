#!/usr/bin/env python3
"""Async Matrix -> Reticulum -> Postgres gateway for Indy_READs.

This is a bootstrap script, not a hidden controller:
- Matrix ingress is optional and guarded behind matrix-nio availability.
- Reticulum emission is optional and guarded behind RNS availability.
- Postgres receipts are the durable truth surface.

Default mode is dry-run so the script can be tested on a workstation that
does not yet have matrix-nio or Reticulum installed.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "matrix_gateway"
MODEL_OUT = ROOT / "05_OUTPUTS" / "model_invocations"
WORK_ORDER_UUID = uuid.UUID("58465be6-9ecb-4f71-b86d-e3641c52d2d8")
WORKER_ID = "indy_reads_runtime"
DEFAULT_ROOM = "#indy_command_deck:localhost"
DEFAULT_HOMESERVER = "http://localhost:8008"
DEFAULT_PROVIDER = "auto"
DEFAULT_RETICULUM_APP = "lucidota"
DEFAULT_RETICULUM_ASPECT = "lane_indy_reads_runtime"
DEFAULT_RETICULUM_LANE_ID = "indy_reads_runtime"
RETICULUM_MTU = 471
DEFAULT_ACTIVE_LANES = ("indy_reads_runtime",) + tuple(f"needle_{idx}" for idx in range(20))

try:  # optional runtime dependency
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - venv should provide psycopg for live DB work
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]

try:  # optional runtime dependency
    from nio import AsyncClient, MatrixRoom, RoomMessageText
except Exception:  # pragma: no cover - installed only in the live Matrix environment
    AsyncClient = None  # type: ignore[assignment]
    MatrixRoom = None  # type: ignore[assignment]
    RoomMessageText = None  # type: ignore[assignment]

try:  # optional runtime dependency
    import RNS
except Exception:  # pragma: no cover - installed only on the Reticulum host
    RNS = None  # type: ignore[assignment]

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from indy_conduit_driver import clean_text  # noqa: E402


def now_z() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_obj(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def chunk_bytes(payload: bytes, mtu: int = RETICULUM_MTU) -> list[bytes]:
    if mtu <= 0:
        raise ValueError("mtu_must_be_positive")
    if not payload:
        return [b""]
    return [payload[i : i + mtu] for i in range(0, len(payload), mtu)]


def infer_ontology_tags(text: str) -> list[str]:
    tags: list[str] = []
    explicit = sorted(set(re.findall(r"@(\d{2})", text)))
    tags.extend([f"@{item}" for item in explicit])
    if re.search(r"\bmatrix\b", text, re.I):
        tags.append("matrix")
    if re.search(r"\breticulum\b|\brns\b", text, re.I):
        tags.append("reticulum")
    if re.search(r"\bindy\b", text, re.I):
        tags.append("indy_reads")
    return sorted(dict.fromkeys(tags))


def select_provider(provider: str, env: dict[str, str] | None = None, which: Callable[[str], str | None] = shutil.which) -> str:
    env = env or os.environ
    choice = (provider or DEFAULT_PROVIDER).strip().lower()
    if choice != "auto":
        return choice
    vibe_bin = which("vibe")
    if not vibe_bin:
        vibe_path = ROOT / ".venv/bin/vibe"
        vibe_bin = str(vibe_path) if vibe_path.exists() else None
    if vibe_bin and env.get("MISTRAL_API_KEY"):
        return "vibes"
    if env.get("GROQ_API_KEY"):
        return "groq"
    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        return "gemini"
    return "dry-run"


def provider_model(provider: str) -> str:
    return {
        "vibes": os.environ.get("MISTRAL_MODEL", "codestral-2508"),
        "groq": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        "gemini": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        "dry-run": "dry-run",
    }.get(provider, "dry-run")


def build_prompt_receipt(
    *,
    room_id: str,
    event_id: str,
    sender: str,
    body: str,
    target_model: str,
    provider: str,
    work_order_uuid: uuid.UUID = WORK_ORDER_UUID,
) -> dict[str, Any]:
    normalized = clean_text(body)
    prompt_id = uuid.uuid5(
        uuid.NAMESPACE_URL,
        json.dumps(
            {
                "room_id": room_id,
                "event_id": event_id,
                "sender": sender,
                "body_hash": sha256_text(normalized),
                "provider": provider,
                "target_model": target_model,
                "work_order_uuid": str(work_order_uuid),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    return {
        "schema": "lucidota.matrix_reticulum_gateway.prompt.v1",
        "prompt_id": str(prompt_id),
        "work_order_uuid": str(work_order_uuid),
        "source": "matrix",
        "source_model": "matrix-nio",
        "receiving_model": "indy_reads",
        "target_model": target_model,
        "provider": provider,
        "conversation_session_id": room_id,
        "parent_prompt_id": None,
        "linked_work_order_uuid": [str(work_order_uuid)],
        "linked_receipt_uuid": [],
        "linked_goal_id": "SYSTEMIC_SWARM_HARDEN_V050",
        "ontology_tags": infer_ontology_tags(normalized),
        "subsystem_tags": ["matrix", "reticulum", "indy_reads"],
        "status": "filed",
        "notes": "matrix ingress filed through gateway",
        "blockers": "",
        "idempotency_key": sha256_text(f"{room_id}:{event_id}:{normalized}"),
        "source_path": f"matrix://{room_id}",
        "received_at": now_z(),
        "received_at_confidence": 1.0,
        "received_at_basis": "matrix_event_ts",
        "detail": {
            "raw_prompt_text": body,
            "normalized_prompt_text": normalized,
            "sender": sender,
            "event_id": event_id,
            "body_sha256": sha256_text(normalized),
            "room_ref": room_id,
            "work_order_uuid": str(work_order_uuid),
        },
    }


def _db_url() -> str:
    return (
        os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL")
        or os.environ.get("ABSURD_SYSTEM_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or "postgresql:///lucidota_state"
    )


def file_prompt_row(receipt: dict[str, Any]) -> dict[str, Any]:
    if psycopg is None:
        return {"performed": False, "blockers": ["psycopg_unavailable"]}
    raw_prompt_text = receipt["detail"].get("raw_prompt_text") or receipt["detail"].get("body") or ""
    normalized_prompt_text = receipt["detail"].get("normalized_prompt_text") or clean_text(str(raw_prompt_text))
    params = (
        receipt["source"],
        receipt["source_model"],
        receipt["receiving_model"],
        receipt["target_model"],
        raw_prompt_text,
        normalized_prompt_text,
        receipt["conversation_session_id"],
        None,
        receipt["linked_work_order_uuid"],
        receipt["linked_receipt_uuid"],
        receipt["linked_goal_id"],
        receipt["ontology_tags"],
        receipt["subsystem_tags"],
        receipt["status"],
        receipt["notes"],
        receipt["blockers"],
        receipt["idempotency_key"],
        receipt["source_path"],
        dt.datetime.fromisoformat(receipt["received_at"].replace("Z", "+00:00")),
        receipt["received_at_confidence"],
        receipt["received_at_basis"],
        json.dumps(receipt["detail"]),
    )
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            row = cur.execute(
                """
                SELECT * FROM lucidota_canon.file_prompt(
                    source => %s,
                    source_model => %s,
                    receiving_model => %s,
                    target_model => %s,
                    raw_prompt_text => %s,
                    normalized_prompt_text => %s,
                    conversation_session_id => %s,
                    parent_prompt_id => %s,
                    linked_work_order_uuid => %s,
                    linked_receipt_uuid => %s,
                    linked_goal_id => %s,
                    ontology_tags => %s,
                    subsystem_tags => %s,
                    status => %s,
                    notes => %s,
                    blockers => %s,
                    idempotency_key => %s,
                    source_path => %s,
                    received_at => %s,
                    received_at_confidence => %s,
                    received_at_basis => %s,
                    detail => %s::jsonb
                )
                """,
                params,
            ).fetchone()
        conn.commit()
    return {"performed": True, "prompt_row": row}


def upsert_partial_invocation(
    *,
    provider: str,
    model_id: str,
    body: str,
    room_id: str,
    event_id: str,
    prompt_id: str | None,
    proof_status: str,
    status: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
    receipt_uuid: uuid.UUID | None = None,
    answer_hash: str = "",
) -> dict[str, Any]:
    if psycopg is None:
        return {"performed": False, "blockers": ["psycopg_unavailable"]}
    receipt_uuid = receipt_uuid or uuid.uuid5(
        uuid.NAMESPACE_URL,
        json.dumps(
            {
                "provider": provider,
                "model_id": model_id,
                "room_id": room_id,
                "event_id": event_id,
                "body_hash": sha256_text(clean_text(body)),
                "prompt_id": prompt_id,
                "work_order_uuid": str(WORK_ORDER_UUID),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    ontology_index = {
        "risk_tier": "T3",
        "actor_role": "control_plane",
        "claim_type": "model_invocation",
        "primitive_refs": infer_ontology_tags(body),
        "subsystem_refs": ["matrix", "reticulum", "indy_reads"],
        "next_route": ["workload_audit_current", "visible_status_layer"],
        "proof_status": proof_status,
    }
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            row = cur.execute(
                """
                INSERT INTO lucidota_audit.workload_audit_ledger (
                    actor_id, actor_class, caller, provider, model_id, action_summary,
                    tokens_in, tokens_out, token_source, receipt_uuid, evidence_refs,
                    proof_status, debt_reason, functionality_explanation, ontology_index,
                    model_identifier_uuid, work_order_uuid, work_order_attempt_uuid, worker_id,
                    refreshed_at
                )
                VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s::uuid, %s::jsonb,
                    %s, %s, %s, %s::jsonb,
                    %s::uuid, %s::uuid, NULL, %s,
                    now()
                )
                ON CONFLICT (receipt_uuid) DO UPDATE SET
                    actor_id = EXCLUDED.actor_id,
                    actor_class = EXCLUDED.actor_class,
                    caller = EXCLUDED.caller,
                    provider = EXCLUDED.provider,
                    model_id = EXCLUDED.model_id,
                    action_summary = EXCLUDED.action_summary,
                    tokens_in = EXCLUDED.tokens_in,
                    tokens_out = EXCLUDED.tokens_out,
                    token_source = EXCLUDED.token_source,
                    evidence_refs = EXCLUDED.evidence_refs,
                    proof_status = EXCLUDED.proof_status,
                    debt_reason = EXCLUDED.debt_reason,
                    functionality_explanation = EXCLUDED.functionality_explanation,
                    ontology_index = EXCLUDED.ontology_index,
                    model_identifier_uuid = EXCLUDED.model_identifier_uuid,
                    work_order_uuid = EXCLUDED.work_order_uuid,
                    worker_id = EXCLUDED.worker_id,
                    refreshed_at = now()
                RETURNING receipt_uuid::text
                """,
                (
                    WORKER_ID,
                    "indy_reads",
                    "matrix_reticulum_gateway",
                    provider,
                    model_id,
                    f"matrix ingress routed via {provider} for {room_id}",
                    tokens_in,
                    tokens_out,
                    "matrix_event",
                    str(receipt_uuid),
                    json.dumps(
                        {
                            "room_id": room_id,
                            "event_id": event_id,
                            "prompt_id": prompt_id,
                            "work_order_uuid": str(WORK_ORDER_UUID),
                            "provider": provider,
                            "model_id": model_id,
                            "answer_hash": answer_hash,
                        }
                    ),
                    proof_status,
                    "matrix ingress awaiting reticulum round-trip" if proof_status != "PROVEN" else "",
                    "matrix ingress gateway receipt",
                    json.dumps(ontology_index),
                    "b0f0a0b0-0000-4000-8000-000000000001",
                    str(WORK_ORDER_UUID),
                    WORKER_ID,
                ),
            ).fetchone()
        conn.commit()
    return {"performed": True, "receipt_uuid": row["receipt_uuid"] if isinstance(row, dict) else row[0]}


def mark_proven(receipt_uuid: str, *, answer_hash: str, tokens_out: int, provider: str, model_id: str) -> dict[str, Any]:
    if psycopg is None:
        return {"performed": False, "blockers": ["psycopg_unavailable"]}
    with psycopg.connect(_db_url(), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE lucidota_audit.workload_audit_ledger
                SET proof_status = 'PROVEN',
                    debt_reason = '',
                    tokens_out = %s,
                    functionality_explanation = 'matrix_reticulum_gateway proof round-trip completed',
                    refreshed_at = now(),
                    evidence_refs = COALESCE(evidence_refs, '{}'::jsonb) || %s::jsonb
                WHERE receipt_uuid = %s::uuid
                """,
                (
                    tokens_out,
                    json.dumps(
                        {
                            "answer_hash": answer_hash,
                            "provider": provider,
                            "model_id": model_id,
                            "state_bus": "INDY_LANE_A_PROVED",
                        }
                    ),
                    receipt_uuid,
                ),
            )
            cur.execute("SELECT pg_notify('state_bus', 'INDY_LANE_A_PROVED');")
        conn.commit()
    return {"performed": True, "receipt_uuid": receipt_uuid}


def reticulum_support() -> bool:
    return RNS is not None


def build_reticulum_payload(prompt: str, reply: str, *, room_id: str, event_id: str, prompt_id: str | None) -> bytes:
    packet = {
        "schema": "lucidota.matrix_reticulum_gateway.packet.v1",
        "room_id": room_id,
        "event_id": event_id,
        "prompt_id": prompt_id,
        "prompt_sha256": sha256_text(prompt),
        "reply_sha256": sha256_text(reply),
        "reply_text": reply,
        "ontology_tags": infer_ontology_tags(prompt + "\n" + reply),
        "canon_status": "not_truth_runtime_only",
        "generated_at": now_z(),
    }
    return json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def build_reticulum_destination(identity: Any, *, lane_id: str, app_name: str = DEFAULT_RETICULUM_APP) -> Any:
    aspect = f"lane_{lane_id}"
    return RNS.Destination(identity, RNS.Destination.OUT, RNS.Destination.PLAIN, app_name, aspect)


def build_reticulum_destination_map(active_lanes: tuple[str, ...] = DEFAULT_ACTIVE_LANES, *, app_name: str = DEFAULT_RETICULUM_APP) -> dict[str, Any]:
    if RNS is None:
        return {}
    RNS.Reticulum()
    identity = RNS.Identity()
    return {
        lane_id: build_reticulum_destination(identity, lane_id=lane_id, app_name=app_name)
        for lane_id in active_lanes
    }


def send_reticulum_frames(payload: bytes, *, lane_id: str = DEFAULT_RETICULUM_LANE_ID, app_name: str = DEFAULT_RETICULUM_APP) -> dict[str, Any]:
    if RNS is None:
        return {"performed": False, "blockers": ["reticulum_unavailable"], "frames": []}
    destination_map = build_reticulum_destination_map(DEFAULT_ACTIVE_LANES, app_name=app_name)
    destination = destination_map[lane_id]
    frames = []
    for idx, frame in enumerate(chunk_bytes(payload, RETICULUM_MTU)):
        packet = RNS.Packet(destination, frame)
        packet.send()
        frames.append({"index": idx, "size": len(frame)})
    return {
        "performed": True,
        "frame_count": len(frames),
        "frames": frames,
        "lane_id": lane_id,
        "aspect": f"lane_{lane_id}",
        "destination_map_size": len(destination_map),
        "active_lane_ids": list(destination_map.keys()),
    }


def build_provider_prompt(*, body: str, room_id: str, event_id: str, prompt_id: str | None, work_order_uuid: uuid.UUID) -> str:
    return "\n".join(
        [
            "Return compact JSON only.",
            f"room_id: {room_id}",
            f"event_id: {event_id}",
            f"prompt_id: {prompt_id or ''}",
            f"work_order_uuid: {work_order_uuid}",
            f"ontology_tags: {', '.join(infer_ontology_tags(body))}",
            f"user_text: {body}",
            "fields: reply_text, response_kind, ontology_tags, next_route, blockers",
        ]
    )


def run_provider_lane(provider: str, prompt_text: str, system_text: str = "") -> dict[str, Any]:
    provider = provider.lower()
    if provider == "dry-run":
        return {
            "performed": False,
            "provider": provider,
            "model": "dry-run",
            "text": prompt_text[:400],
            "blockers": [],
        }
    if provider == "vibes":
        vibe_bin = ROOT / ".venv/bin/vibe"
        cmd = [
            str(vibe_bin),
            "-p",
            prompt_text,
            "--agent",
            "auto-approve",
            "--trust",
            "--workdir",
            str(ROOT),
        ]
        if system_text.strip():
            cmd.extend(["--system", system_text])
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
        return {
            "performed": proc.returncode == 0,
            "provider": provider,
            "model": provider_model(provider),
            "text": proc.stdout.strip()[-4000:],
            "stderr_tail": proc.stderr.strip()[-2000:],
            "returncode": proc.returncode,
            "blockers": [] if proc.returncode == 0 else [f"vibe_exit_{proc.returncode}"],
        }
    if provider in {"groq", "gemini"}:
        subcmd = "groq-chat" if provider == "groq" else "gemini-chat"
        cmd = [
            sys.executable,
            "scripts/model_runner_cli.py",
            subcmd,
            "--prompt",
            prompt_text,
            "--system",
            system_text,
            "--model",
            provider_model(provider),
            "--max-tokens",
            "256",
            "--temperature",
            "0",
            "--json",
            "--execute",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
        output = ""
        blocker = f"{provider}_delegate_failed"
        for line in reversed([line.strip() for line in proc.stdout.splitlines() if line.strip()]):
            if line.startswith("{"):
                try:
                    output = json.loads(line).get("text", "")
                    blocker = ""
                except Exception:
                    pass
                break
        return {
            "performed": proc.returncode == 0 and not blocker,
            "provider": provider,
            "model": provider_model(provider),
            "text": output or proc.stdout.strip()[-4000:],
            "stderr_tail": proc.stderr.strip()[-2000:],
            "returncode": proc.returncode,
            "blockers": [blocker] if blocker else [],
        }
    return {"performed": False, "provider": provider, "model": "dry-run", "text": prompt_text[:400], "blockers": [f"unknown_provider:{provider}"]}


async def send_matrix_reply(client: Any, room_id: str, reply_text: str) -> None:
    for idx, frame in enumerate(chunk_bytes(reply_text.encode("utf-8"), RETICULUM_MTU)):
        notice = frame.decode("utf-8", errors="replace")
        await client.room_send(
            room_id=room_id,
            message_type="m.room.message",
            content={"msgtype": "m.notice", "body": notice},
        )


@dataclass
class GatewayConfig:
    homeserver: str = DEFAULT_HOMESERVER
    user_id: str = ""
    device_id: str = ""
    password: str = ""
    access_token: str = ""
    room_alias: str = DEFAULT_ROOM
    provider: str = DEFAULT_PROVIDER
    execute: bool = False
    listen: bool = True
    once: bool = False
    retain_receipt_dir: Path = OUT


class MatrixReticulumGateway:
    def __init__(self, config: GatewayConfig):
        self.config = config
        self.client: Any = None
        self.receipts: list[dict[str, Any]] = []
        self.user_id = config.user_id

    def _has_matrix_credentials(self) -> bool:
        return bool(self.config.user_id and (self.config.password or self.config.access_token))

    async def login(self) -> None:
        if AsyncClient is None:
            raise SystemExit("matrix_nio_unavailable")
        self.client = AsyncClient(self.config.homeserver, self.config.user_id)
        self.client.add_event_callback(self.on_room_message, RoomMessageText)
        if self.config.access_token:
            device_id = self.config.device_id or "matrix_reticulum_gateway"
            self.client.restore_login(self.config.user_id, device_id, self.config.access_token)
        elif self.config.password:
            await self.client.login(self.config.password)
        else:
            raise SystemExit("matrix_credentials_required")
        self.user_id = getattr(self.client, "user_id", self.config.user_id)
        if self.config.room_alias:
            await self.client.join(self.config.room_alias)

    async def on_room_message(self, room: Any, event: Any) -> None:
        sender = str(getattr(event, "sender", ""))
        if sender and sender == self.user_id:
            return
        body = str(getattr(event, "body", "") or "")
        room_id = str(getattr(room, "room_id", self.config.room_alias))
        event_id = str(getattr(event, "event_id", ""))
        normalized = clean_text(body)
        provider = select_provider(self.config.provider)
        chosen_model = provider_model(provider)
        prompt_receipt = build_prompt_receipt(
            room_id=room_id,
            event_id=event_id,
            sender=sender,
            body=body,
            target_model=chosen_model,
            provider=provider,
        )
        prompt_result = file_prompt_row(prompt_receipt)
        prompt_id = None
        if prompt_result.get("performed"):
            row = prompt_result.get("prompt_row") or {}
            prompt_id = str(row.get("prompt_id") or row.get("prompt_id::text") or "")
        partial = upsert_partial_invocation(
            provider=provider,
            model_id=chosen_model,
            body=body,
            room_id=room_id,
            event_id=event_id,
            prompt_id=prompt_id,
            proof_status="PARTIAL",
            status="queued",
        )
        provider_prompt = build_provider_prompt(
            body=normalized,
            room_id=room_id,
            event_id=event_id,
            prompt_id=prompt_id,
            work_order_uuid=WORK_ORDER_UUID,
        )
        response = run_provider_lane(provider, provider_prompt, system_text="Return compact JSON only.")
        reply_text = response.get("text") or json.dumps(
            {
                "reply_text": normalized[:400],
                "response_kind": "dry_run",
                "ontology_tags": infer_ontology_tags(normalized),
                "next_route": ["workload_audit_current"],
                "blockers": response.get("blockers", []),
            },
            sort_keys=True,
        )
        reticulum_payload = build_reticulum_payload(body, reply_text, room_id=room_id, event_id=event_id, prompt_id=prompt_id)
        reticulum = send_reticulum_frames(reticulum_payload, lane_id=DEFAULT_RETICULUM_LANE_ID, app_name=DEFAULT_RETICULUM_APP)
        if self.config.execute and self.client is not None and not response.get("blockers"):
            await send_matrix_reply(self.client, room_id, reply_text)
            if partial.get("performed") and prompt_id is not None:
                mark_proven(
                    partial["receipt_uuid"],
                    answer_hash=sha256_text(reply_text),
                    tokens_out=max(1, len(reply_text.split())),
                    provider=provider,
                    model_id=chosen_model,
                )
        receipt = {
            "schema": "lucidota.matrix_reticulum_gateway.receipt.v1",
            "generated_at": now_z(),
            "room_id": room_id,
            "event_id": event_id,
            "sender": sender,
            "body": body,
            "normalized_body": normalized,
            "prompt_id": prompt_id,
            "work_order_uuid": str(WORK_ORDER_UUID),
            "provider": provider,
            "model_id": chosen_model,
            "reticulum_lane_id": DEFAULT_RETICULUM_LANE_ID,
            "reticulum_aspect": f"lane_{DEFAULT_RETICULUM_LANE_ID}",
            "prompt_receipt": prompt_result,
            "partial_receipt": partial,
            "reticulum": reticulum,
            "response": response,
            "reply_text": reply_text,
            "proof_status": "PROVEN" if self.config.execute and not response.get("blockers") else "PARTIAL",
            "canon_status": "not_truth_runtime_only",
            "ontology_index": {
                "primitive_refs": infer_ontology_tags(body + "\n" + reply_text),
                "subsystem_refs": ["matrix", "reticulum", "indy_reads"],
                "target_room": room_id,
            },
            "blockers": (prompt_result.get("blockers") or []) + (partial.get("blockers") or []) + (response.get("blockers") or []) + (reticulum.get("blockers") or []),
        }
        self.receipts.append(receipt)
        self._write_receipt(receipt)

    def _write_receipt(self, receipt: dict[str, Any]) -> Path:
        self.config.retain_receipt_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.retain_receipt_dir / f"matrix_reticulum_gateway_{stamp()}.json"
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    async def run(self) -> int:
        if not self.config.listen and self.config.provider == "dry-run" and not self._has_matrix_credentials():
            receipt = {
                "schema": "lucidota.matrix_reticulum_gateway.receipt.v1",
                "generated_at": now_z(),
                "blockers": [],
                "canon_status": "not_truth_runtime_only",
                "proof_status": "PARTIAL",
                "provider_choice": "dry-run",
                "reticulum_supported": reticulum_support(),
                "matrix_transport": "bootstrap_only",
            }
            self.config.retain_receipt_dir.mkdir(parents=True, exist_ok=True)
            path = self.config.retain_receipt_dir / f"matrix_reticulum_gateway_bootstrap_{stamp()}.json"
            path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print("REPORT_PATH=" + rel(path))
            print(json.dumps(receipt, sort_keys=True))
            return 0
        await self.login()
        if not self.config.listen:
            return 0
        if self.client is None:
            raise SystemExit("matrix_client_missing")
        if self.config.once:
            await self.client.sync(timeout=30000)
        else:
            await self.client.sync_forever(timeout=30000)
        return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Matrix -> Reticulum gateway for Indy_READs.")
    ap.add_argument("--homeserver", default=os.environ.get("MATRIX_HOMESERVER", DEFAULT_HOMESERVER))
    ap.add_argument("--user-id", default=os.environ.get("MATRIX_USER_ID", ""))
    ap.add_argument("--device-id", default=os.environ.get("MATRIX_DEVICE_ID", "matrix_reticulum_gateway"))
    ap.add_argument("--password", default=os.environ.get("MATRIX_PASSWORD", ""))
    ap.add_argument("--access-token", default=os.environ.get("MATRIX_ACCESS_TOKEN", ""))
    ap.add_argument("--room-alias", default=os.environ.get("MATRIX_ROOM_ALIAS", DEFAULT_ROOM))
    ap.add_argument("--provider", default=os.environ.get("MATRIX_GATEWAY_PROVIDER", DEFAULT_PROVIDER), choices=["auto", "vibes", "groq", "gemini", "dry-run"])
    ap.add_argument("--execute", action="store_true", help="Send replies and promote ledger rows to PROVEN when possible.")
    ap.add_argument("--once", action="store_true", help="Perform a single sync cycle and exit.")
    ap.add_argument("--no-listen", dest="listen", action="store_false", help="Login and return after initial setup.")
    ap.add_argument("--receipt-dir", default=str(OUT))
    return ap


def main() -> int:
    args = build_parser().parse_args()
    config = GatewayConfig(
        homeserver=args.homeserver,
        user_id=args.user_id,
        device_id=args.device_id,
        password=args.password,
        access_token=args.access_token,
        room_alias=args.room_alias,
        provider=args.provider,
        execute=bool(args.execute),
        listen=bool(args.listen),
        once=bool(args.once),
        retain_receipt_dir=Path(args.receipt_dir),
    )
    gateway = MatrixReticulumGateway(config)
    if AsyncClient is None:
        receipt = {
            "schema": "lucidota.matrix_reticulum_gateway.receipt.v1",
            "generated_at": now_z(),
            "blockers": ["matrix_nio_unavailable"],
            "canon_status": "not_truth_runtime_only",
            "proof_status": "PARTIAL",
            "provider_choice": select_provider(config.provider),
            "reticulum_supported": reticulum_support(),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"matrix_reticulum_gateway_blocked_{stamp()}.json"
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("REPORT_PATH=" + rel(path))
        print(json.dumps(receipt, sort_keys=True))
        return 3
    return asyncio.run(gateway.run())


if __name__ == "__main__":
    raise SystemExit(main())
