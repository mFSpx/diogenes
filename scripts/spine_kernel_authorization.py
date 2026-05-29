#!/usr/bin/env python3
"""ABSURD/Postgres kernel-authorization enforcement helpers.

This is not a status/report layer. Workers import this before executing jobs whose
payload can cause downstream custody, graph, parser, OCR, or ledger effects.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from kernel_control_packet import verify_control_packet
from spine_authority_checker import decide_authority

# Initial hard gate: corpus lane jobs are the first path wired through CKDOG1.
# More job kinds should be added here only when their enqueue path emits packets.
KERNEL_AUTH_REQUIRED_JOB_KINDS = {
    "boring_beast_work_item",
    "korpus_lane_job",
    "surface_instruction_compiled_command",
    "conversation_command_work_order",
}

KERNEL_AUTH_REQUIRED_QUEUES = {
    "korpus",
}


@dataclass(frozen=True)
class KernelAuthorizationVerdict:
    required: bool
    ok: bool
    error_kind: str | None = None
    error_message: str | None = None
    packet_hash: str | None = None
    authorized_by: str | None = None

    def as_result(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "ok": self.ok,
            "error_kind": self.error_kind,
            "error_message": self.error_message,
            "packet_hash": self.packet_hash,
            "authorized_by": self.authorized_by,
        }


def kernel_authorization_required(queue_name: str, job_kind: str, payload: dict[str, Any]) -> bool:
    if job_kind in KERNEL_AUTH_REQUIRED_JOB_KINDS:
        return True
    if queue_name in KERNEL_AUTH_REQUIRED_QUEUES and payload.get("bridge_version") in {"v2", "v3"}:
        return True
    if payload.get("canonical_mutation_allowed") is True:
        return True
    return False


def validate_job_kernel_authorization(
    *,
    queue_name: str,
    job_kind: str,
    payload: dict[str, Any],
) -> KernelAuthorizationVerdict:
    required = kernel_authorization_required(queue_name, job_kind, payload)
    if not required:
        return KernelAuthorizationVerdict(required=False, ok=True)
    packet = payload.get("kernel_authorization")
    if not isinstance(packet, dict):
        return KernelAuthorizationVerdict(
            required=True,
            ok=False,
            error_kind="missing_kernel_authorization",
            error_message="job requires kernel_authorization control packet",
        )
    ok, error = verify_control_packet(packet)
    if not ok:
        return KernelAuthorizationVerdict(
            required=True,
            ok=False,
            error_kind="invalid_kernel_authorization",
            error_message=error or "invalid kernel_authorization control packet",
            packet_hash=packet.get("packet_hash"),
            authorized_by=packet.get("authorized_by"),
        )
    expected_queue = packet.get("payload", {}).get("queue_name")
    if expected_queue and expected_queue != queue_name:
        return KernelAuthorizationVerdict(
            required=True,
            ok=False,
            error_kind="kernel_authorization_queue_mismatch",
            error_message=f"packet queue_name={expected_queue} does not match job queue_name={queue_name}",
            packet_hash=packet.get("packet_hash"),
            authorized_by=packet.get("authorized_by"),
        )
    expected_lane = packet.get("payload", {}).get("lane")
    payload_lane = payload.get("lane")
    if expected_lane and payload_lane and expected_lane != payload_lane:
        return KernelAuthorizationVerdict(
            required=True,
            ok=False,
            error_kind="kernel_authorization_lane_mismatch",
            error_message=f"packet lane={expected_lane} does not match payload lane={payload_lane}",
            packet_hash=packet.get("packet_hash"),
            authorized_by=packet.get("authorized_by"),
        )
    expected_source_path = packet.get("payload", {}).get("source_path")
    payload_source_path = payload.get("source_path")
    if expected_source_path and payload_source_path and expected_source_path != payload_source_path:
        return KernelAuthorizationVerdict(
            required=True,
            ok=False,
            error_kind="kernel_authorization_source_path_mismatch",
            error_message=f"packet source_path={expected_source_path} does not match payload source_path={payload_source_path}",
            packet_hash=packet.get("packet_hash"),
            authorized_by=packet.get("authorized_by"),
        )
    expected_idempotency_key = packet.get("payload", {}).get("idempotency_key")
    payload_idempotency_key = payload.get("idempotency_key")
    if expected_idempotency_key and payload_idempotency_key and expected_idempotency_key != payload_idempotency_key:
        return KernelAuthorizationVerdict(
            required=True,
            ok=False,
            error_kind="kernel_authorization_idempotency_key_mismatch",
            error_message="packet idempotency_key does not match payload idempotency_key",
            packet_hash=packet.get("packet_hash"),
            authorized_by=packet.get("authorized_by"),
        )
    if job_kind == "conversation_command_work_order" or payload.get("authority_class"):
        evidence_refs: list[str] = []
        for key in ("evidence_refs", "source_artifact_refs", "target_refs"):
            value = payload.get(key)
            if isinstance(value, list):
                evidence_refs.extend(str(x) for x in value if str(x).strip())
            elif value:
                evidence_refs.append(str(value))
        env = payload.get("command_envelope") if isinstance(payload.get("command_envelope"), dict) else {}
        for key in ("evidence_refs", "artifact_refs", "target_refs"):
            value = env.get(key)
            if isinstance(value, list):
                evidence_refs.extend(str(x) for x in value if str(x).strip())
            elif value:
                evidence_refs.append(str(value))
        authority_lane = queue_name if queue_name == "surface_cep" else (payload.get("lane") or job_kind)
        authority_decision = decide_authority(
            authority_class=payload.get("authority_class"),
            effect=payload.get("allowed_effect"),
            lane=authority_lane,
            evidence_refs=evidence_refs,
        )
        if not authority_decision["allowed"]:
            return KernelAuthorizationVerdict(
                required=True,
                ok=False,
                error_kind="spine_authority_denied",
                error_message=",".join(authority_decision["blockers"]),
                packet_hash=packet.get("packet_hash"),
                authorized_by=packet.get("authorized_by"),
            )
    return KernelAuthorizationVerdict(
        required=True,
        ok=True,
        packet_hash=packet.get("packet_hash"),
        authorized_by=packet.get("authorized_by"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate ABSURD/Postgres job payload kernel authorization before worker execution.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--queue-name", required=True)
    v.add_argument("--job-kind", required=True)
    src = v.add_mutually_exclusive_group(required=True)
    src.add_argument("--payload-json")
    src.add_argument("--payload-file")
    args = ap.parse_args()
    if args.cmd == "validate":
        if args.payload_json is not None:
            payload = json.loads(args.payload_json)
        else:
            from pathlib import Path as _Path
            payload = json.loads(_Path(args.payload_file).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            print(json.dumps({"ok": False, "error_kind": "payload_not_object"}, sort_keys=True))
            return 2
        verdict = validate_job_kernel_authorization(queue_name=args.queue_name, job_kind=args.job_kind, payload=payload)
        print(json.dumps(verdict.as_result(), sort_keys=True))
        return 0 if verdict.ok else 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
