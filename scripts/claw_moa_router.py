#!/usr/bin/env python3
"""Claw operator-front-door router: deterministic lane gate + draft hyperplex.

This is a thin adapter, not a new agent framework. It reuses the existing
fast/slow lane gate, Groq bridge, and language membrane so Claw can expose one
operator command that preserves route provenance before any model work.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from core.language_membrane import weave_output  # noqa: E402
from fast_slow_lane_gate import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_RECEIPT_ROOT,
    build_flow,
    cache_bit,
    flush_fastlane,
    importance_score,
    route_packet,
    status_payload,
)
from spine_job_adapter import ABSURDJobAdapter  # noqa: E402
from spine_common import now, receipt, rel, sha256_json  # noqa: E402

SCHEMA = "lucidota.claw_moa_router.v1"
DEFAULT_RECEIPT_ROOT = ROOT / "05_OUTPUTS" / "claw_moa"
DEFAULT_ABSURD_DIR = ROOT / "09_STORAGE" / "absurd" / "claw_moa"


def load_text(value: str | None, path: str | None) -> str:
    if path:
        source = Path(path)
        if not source.is_absolute():
            source = ROOT / source
        return source.read_text(encoding="utf-8", errors="replace")
    return value or ""


def read_json_object(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("metadata JSON must decode to an object")
    return data


def build_local_model_admission() -> dict[str, Any]:
    """Describe strict resident-model admission without probing external API keys."""
    try:
        from scripts.lucidota_strict_model_stack_admission import build_strict_stack_plan

        plan = build_strict_stack_plan(root=ROOT)
        services = [
            {
                "name": service.get("name"),
                "required": bool(service.get("required")),
                "device_lane": service.get("device_lane"),
                "port": service.get("port"),
                "port_range": service.get("port_range"),
                "switch_group": service.get("switch_group"),
                "switch_role": service.get("switch_role"),
                "gpu_switch_env": service.get("gpu_switch_env"),
            }
            for service in plan.get("services", [])
        ]
        return {
            "schema": "lucidota.luci.local_model_admission_ref.v1",
            "mode": "strict_fail_closed",
            "source": "scripts/lucidota_strict_model_stack_admission.py",
            "startup_blocked_by_missing_provider_keys": False,
            "services": services,
        }
    except Exception as exc:  # pragma: no cover - defensive runtime boundary
        return {
            "schema": "lucidota.luci.local_model_admission_ref.v1",
            "mode": "strict_fail_closed",
            "source": "scripts/lucidota_strict_model_stack_admission.py",
            "startup_blocked_by_missing_provider_keys": False,
            "services": [],
            "status": "unverified",
            "error": f"{type(exc).__name__}: {exc}",
        }


def build_provider_lanes(env: dict[str, str] | None = None) -> dict[str, Any]:
    """External API/tool lanes are capability checks, not local model admission."""
    env = dict(os.environ if env is None else env)
    vibe_path = ROOT / ".venv" / "bin" / "vibe"
    vibe_found = vibe_path.exists() or bool(shutil.which("vibe"))
    return {
        "groq": {
            "lane_kind": "external_provider",
            "provider": "groq",
            "secret_ref": "GROQ_API_KEY",
            "status": "available" if env.get("GROQ_API_KEY") else "skipped",
            "reason": "GROQ_API_KEY_present" if env.get("GROQ_API_KEY") else "missing_GROQ_API_KEY",
            "startup_blocker": False,
        },
        "vibes": {
            "lane_kind": "external_provider",
            "provider": "mistral_vibes",
            "tool": ".venv/bin/vibe",
            "status": "available" if vibe_found else "skipped",
            "reason": "vibe_cli_present" if vibe_found else "vibe_cli_missing",
            "startup_blocker": False,
        },
        "promptflow": {
            "lane_kind": "sidecar_operator_tool",
            "provider": "promptflow",
            "status": "sidecar_only",
            "startup_blocker": False,
        },
    }


def build_lane_plan(decision: dict[str, Any], *, execute_groq: bool) -> dict[str, Any]:
    base_lane = decision["lane"]
    slow = base_lane == "SLOWLANE"
    external_lanes = ["groq", "vibes"] if slow else (["groq"] if execute_groq else [])
    return {
        "schema": "lucidota.claw_moa_router.lane_plan.v1",
        "base_lane": base_lane,
        "external_lanes": external_lanes,
        "deterministic_lanes": ["fast_slow_lane_gate", "language_membrane"],
        "execution_state": "executed" if execute_groq else "planned_not_executed",
        "local_model_admission": build_local_model_admission(),
        "provider_lanes": build_provider_lanes(),
        "groq_contract": {
            "provider": "groq",
            "api_key_lane": "GROQ_API_KEY",
            "token_window_note": "separate_from_mistral_vibes",
        },
        "vibes_contract": {
            "provider": "mistral_vibes",
            "tool": ".venv/bin/vibe -p",
            "budget_note": "operator_note_200k_session_32k_window_applies_here_only",
        },
        "tasks": [
            {
                "lane": "fast_slow_lane_gate",
                "state": "executed",
                "model_calls_performed": False,
            },
            {
                "lane": "groq",
                "state": "execute_requested" if execute_groq else "planned",
                "purpose": "bounded synthesis/summary, not hidden controller",
            },
            {
                "lane": "vibes",
                "state": "planned" if slow else "not_needed_for_fastlane",
                "purpose": "bounded code/deep-work delegation when explicitly executed",
            },
        ],
    }


def build_route_targets(packet: dict[str, Any], decision: dict[str, Any], lane_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Router-of-routers packet targets: route may target routers/workflows/providers."""
    packet_id = packet["packet_id"]
    targets: list[dict[str, Any]] = [
        {
            "schema": "lucidota.luci.route_target.v1",
            "decision_packet_id": packet_id,
            "target_kind": "router",
            "target_id": "language_router",
            "state": "executed",
        },
        {
            "schema": "lucidota.luci.route_target.v1",
            "decision_packet_id": packet_id,
            "target_kind": "router",
            "target_id": "fast_slow_lane_gate",
            "state": "executed",
        },
    ]
    if decision["lane"] == "SLOWLANE":
        targets.append(
            {
                "schema": "lucidota.luci.route_target.v1",
                "decision_packet_id": packet_id,
                "target_kind": "workflow",
                "target_id": "absurd_workflow_queue",
                "state": "planned",
            }
        )
    for provider in lane_plan.get("external_lanes", []):
        targets.append(
            {
                "schema": "lucidota.luci.route_target.v1",
                "decision_packet_id": packet_id,
                "target_kind": "provider_lane",
                "target_id": provider,
                "state": lane_plan.get("provider_lanes", {}).get(provider, {}).get("status", "planned"),
            }
        )
    return targets


def groq_prompt(text: str, decision: dict[str, Any], lane_plan: dict[str, Any]) -> str:
    return (
        "You are the Groq synthesis lane inside LUCIDOTA Claw. "
        "The deterministic route already happened; do not override it. "
        "Return a terse operator-facing draft with: route, reason, next action.\n\n"
        f"TEXT:\n{text[:4000]}\n\n"
        f"ROUTE:\n{json.dumps(decision, sort_keys=True)}\n\n"
        f"LANE_PLAN:\n{json.dumps(lane_plan, sort_keys=True)}\n"
    )


def run_groq_synthesis(
    text: str,
    decision: dict[str, Any],
    lane_plan: dict[str, Any],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    timeout_sec: float,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/groq_chat_cli.py",
        "--prompt",
        groq_prompt(text, decision, lane_plan),
        "--system",
        "Return concise plain text only.",
        "--model",
        model,
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--timeout-sec",
        str(timeout_sec),
        "--execute",
    ]
    started = now()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout_sec + 15)
    receipt_path = None
    status = None
    text_lines: list[str] = []
    for line in proc.stdout.splitlines():
        if line.startswith("RECEIPT_PATH="):
            receipt_path = line.split("=", 1)[1]
        elif line.startswith("GROQ_CHAT="):
            status = line.split("=", 1)[1]
        else:
            text_lines.append(line)
    return {
        "provider": "groq",
        "model": model,
        "started_at": started,
        "returncode": proc.returncode,
        "status": status or ("PASS" if proc.returncode == 0 else "FAILED"),
        "receipt_path": receipt_path,
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
        "text": "\n".join(text_lines).strip(),
        "execute_performed": proc.returncode == 0,
    }


def write_task_chain_jsonl(path: str | Path, nodes: list[dict[str, Any]]) -> str:
    out = Path(path)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(node, sort_keys=True, separators=(",", ":")) + "\n" for node in nodes),
        encoding="utf-8",
    )
    return rel(out)


def redact_database_url(database_url: str) -> str:
    if database_url.startswith("postgresql:///"):
        return "postgresql:///<database>"
    if "@" in database_url:
        return "postgresql://<redacted>@" + database_url.split("@", 1)[1]
    return "set_redacted"


def enqueue_db_chain_nodes(
    nodes: list[dict[str, Any]],
    *,
    database_url: str,
    queue_name: str = "luci_operator",
    workflow_name: str = "luci_operator_chain",
) -> dict[str, Any]:
    """Persist planned chain nodes into the canonical ABSURD queue tables."""
    planned = [node for node in nodes if node.get("state") == "planned"]
    result: dict[str, Any] = {
        "schema": "lucidota.claw_moa_router.db_chain_enqueue.v1",
        "database_url": redact_database_url(database_url),
        "queue_name": queue_name,
        "workflow_name": workflow_name,
        "db_writes_performed": False,
        "jobs": [],
        "blockers": [],
    }
    if not planned:
        return result
    try:
        import psycopg  # type: ignore

        with psycopg.connect(database_url, connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO lucidota_control.absurd_queue(queue_name, owner_subsystem, notes)
                VALUES (%s, %s, %s)
                ON CONFLICT (queue_name) DO UPDATE SET updated_at=now()
                """,
                (queue_name, "luci", "LUCI operator slow-lane chain queue"),
            )
            node_to_job: dict[str, str] = {}
            for node in planned:
                deps = [node_to_job[dep] for dep in node.get("depends_on", []) if dep in node_to_job]
                payload = {
                    "schema": "lucidota.claw_moa_router.chain_job_payload.v1",
                    "node": {**node, "resolved_depends_on_job_uuid": deps},
                    "operator_instruction": "execute this bounded LUCI chain node through its named lane contract",
                    "route_to_routing": True,
                }
                idempotency_key = sha256_json(
                    {
                        "component": "claw_moa_router",
                        "queue": queue_name,
                        "node_id": node["node_id"],
                        "node": node,
                    }
                )
                cur.execute(
                    """
                    INSERT INTO lucidota_control.absurd_queue_job
                      (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
                    VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb)
                    ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at=now()
                    RETURNING job_uuid::text, (xmax = 0) AS inserted_new
                    """,
                    (
                        queue_name,
                        workflow_name,
                        node["node_id"],
                        idempotency_key,
                        json.dumps(payload, sort_keys=True),
                        100,
                        3,
                        json.dumps(
                            {
                                "source": "claw_moa_router",
                                "lane": node.get("lane"),
                                "depends_on_job_uuid": deps,
                            },
                            sort_keys=True,
                        ),
                    ),
                )
                job_uuid, inserted_new = cur.fetchone()
                event_kind = "enqueued" if inserted_new else "audit"
                cur.execute(
                    """
                    INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, event_source, detail)
                    VALUES (%s::uuid, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        job_uuid,
                        queue_name,
                        event_kind,
                        "claw_moa_router",
                        json.dumps(
                            {
                                "workflow_name": workflow_name,
                                "job_kind": node["node_id"],
                                "idempotency_key": idempotency_key,
                                "inserted_new": bool(inserted_new),
                                "depends_on_job_uuid": deps,
                            },
                            sort_keys=True,
                        ),
                    ),
                )
                node_to_job[node["node_id"]] = str(job_uuid)
                result["jobs"].append(
                    {
                        "node_id": node["node_id"],
                        "job_uuid": str(job_uuid),
                        "inserted_new": bool(inserted_new),
                        "depends_on_job_uuid": deps,
                    }
                )
            conn.commit()
        result["db_writes_performed"] = True
    except Exception as exc:  # pragma: no cover - runtime DB boundary
        result["blockers"].append(f"db_enqueue_exception:{type(exc).__name__}:{str(exc)[:240]}")
    return result


def enqueue_planned_chain_nodes(
    nodes: list[dict[str, Any]],
    absurd_dir: str | Path,
    *,
    database_url: str | None = None,
    db_queue_name: str = "luci_operator",
) -> dict[str, Any]:
    """Queue only not-yet-executed chain nodes in the existing ABSURD adapter."""
    adapter = ABSURDJobAdapter(absurd_dir)
    planned = [node for node in nodes if node.get("state") == "planned"]
    node_to_job: dict[str, str] = {}
    jobs: list[dict[str, Any]] = []
    for node in planned:
        deps = [node_to_job[dep] for dep in node.get("depends_on", []) if dep in node_to_job]
        payload = {
            "schema": "lucidota.claw_moa_router.chain_job_payload.v1",
            "node": node,
            "operator_instruction": "execute this bounded Claw MOA chain node through its named lane contract",
        }
        job = adapter.create_job(
            lane=f"claw_moa.{node['node_id']}",
            payload=payload,
            idempotency_key=sha256_json(
                {
                    "component": "claw_moa_router",
                    "node_id": node["node_id"],
                    "node": node,
                }
            ),
            depends_on=deps,
        )
        if job["state"] == "CREATED":
            job = adapter.transition(job["job_id"], "QUEUED")
        node_to_job[node["node_id"]] = job["job_id"]
        jobs.append(job)
    db_enqueue = (
        enqueue_db_chain_nodes(nodes, database_url=database_url, queue_name=db_queue_name)
        if database_url
        else {
            "schema": "lucidota.claw_moa_router.db_chain_enqueue.v1",
            "db_writes_performed": False,
            "reason": "database_url_not_provided",
            "jobs": [],
            "blockers": [],
        }
    )
    return {
        "schema": "lucidota.claw_moa_router.chain_enqueue.v1",
        "status": "PASSED" if not db_enqueue.get("blockers") else "DEGRADED",
        "execute_performed": True,
        "job_count": len(jobs),
        "jobs": jobs,
        "absurd_state_path": rel(adapter.state_path),
        "absurd_jobs_path": rel(adapter.jobs_path),
        "canonical_graph_writes_performed": False,
        "db_writes_performed": bool(db_enqueue.get("db_writes_performed")),
        "db_enqueue": db_enqueue,
    }


def build_task_chain(
    *,
    decision: dict[str, Any],
    lane_plan: dict[str, Any],
    cache_path: str | None,
    model_result: dict[str, Any] | None,
    chain_jsonl_out: str | Path | None,
    enqueue_chain: bool,
    absurd_dir: str | Path,
    database_url: str | None,
    db_queue_name: str,
    include_promptflow_prototype: bool,
    promptflow_flow: str,
    promptflow_data: str | None,
    promptflow_run_id: str | None,
) -> dict[str, Any]:
    """Build an inspectable chain; execution stays bounded to already-run nodes."""
    nodes: list[dict[str, Any]] = [
        {
            "node_id": "ingest_packet",
            "lane": "operator_front_door",
            "state": "executed",
            "depends_on": [],
            "model_calls_performed": False,
            "canonical_graph_writes_performed": False,
        },
        {
            "node_id": "deterministic_route",
            "lane": "fast_slow_lane_gate",
            "state": "executed",
            "depends_on": ["ingest_packet"],
            "route": decision["lane"],
            "route_reason": decision.get("route_reason", []),
            "model_calls_performed": False,
            "canonical_graph_writes_performed": False,
        },
        {
            "node_id": "cache_packet",
            "lane": "fast_slow_cache",
            "state": "executed",
            "depends_on": ["deterministic_route"],
            "receipt_expected": cache_path,
            "model_calls_performed": False,
            "canonical_graph_writes_performed": False,
        },
    ]

    hyperplex_deps = ["cache_packet"]
    if decision["lane"] == "SLOWLANE":
        nodes.append(
            {
                "node_id": "slow_queue_plan",
                "lane": "slowlane",
                "state": "planned",
                "depends_on": ["deterministic_route"],
                "command": "./claw work-order next-actions --actions-json ...",
                "model_calls_performed": False,
                "canonical_graph_writes_performed": False,
            }
        )

    if "groq" in lane_plan.get("external_lanes", []) or model_result:
        groq_state = "planned"
        if model_result:
            groq_state = "executed" if model_result.get("execute_performed") else "failed"
        nodes.append(
            {
                "node_id": "groq_synthesis",
                "lane": "groq",
                "state": groq_state,
                "depends_on": ["deterministic_route"],
                "command": "./claw operate --execute-groq --text ...",
                "receipt_expected": (model_result or {}).get("receipt_path"),
                "model_calls_performed": groq_state == "executed",
                "canonical_graph_writes_performed": False,
            }
        )
        hyperplex_deps.append("groq_synthesis")

    if include_promptflow_prototype:
        pf_run_id = promptflow_run_id or ("claw_moa_pf_" + sha256_json({"cache_path": cache_path, "lane": decision["lane"]})[:16])
        nodes.append(
            {
                "node_id": "promptflow_visual_prototype",
                "lane": "promptflow_visual",
                "state": "planned",
                "depends_on": ["deterministic_route"],
                "command": "./claw flow run <flow-dir> --batch-data DATA --run-id ID",
                "flow": promptflow_flow,
                "data": promptflow_data,
                "run_id": pf_run_id,
                "role": "visual_prototype_only_not_correctness_gate",
                "model_calls_performed": False,
                "canonical_graph_writes_performed": False,
            }
        )

    if "vibes" in lane_plan.get("external_lanes", []):
        vibes_dep = "slow_queue_plan" if decision["lane"] == "SLOWLANE" else "deterministic_route"
        nodes.append(
            {
                "node_id": "vibes_delegate",
                "lane": "mistral_vibes",
                "state": "planned",
                "depends_on": [vibes_dep],
                "command": ".venv/bin/vibe -p <bounded code-work prompt> --agent auto-approve --trust",
                "budget_note": "200k/session and ~32k active window applies to this Mistral/Vibes lane only",
                "model_calls_performed": False,
                "canonical_graph_writes_performed": False,
            }
        )

    nodes.append(
        {
            "node_id": "hyperplex_output",
            "lane": "language_membrane",
            "state": "executed",
            "depends_on": hyperplex_deps,
            "outbound_state": "draft_only",
            "model_calls_performed": False,
            "canonical_graph_writes_performed": False,
        }
    )

    chain_jsonl_path = write_task_chain_jsonl(chain_jsonl_out, nodes) if chain_jsonl_out else None
    enqueue = enqueue_planned_chain_nodes(
        nodes,
        absurd_dir,
        database_url=database_url,
        db_queue_name=db_queue_name,
    ) if enqueue_chain else {
        "schema": "lucidota.claw_moa_router.chain_enqueue.v1",
        "status": "DRY_RUN",
        "execute_performed": False,
        "job_count": 0,
        "db_writes_performed": False,
    }
    return {
        "schema": "lucidota.claw_moa_router.task_chain.v1",
        "generated_at": now(),
        "base_lane": decision["lane"],
        "node_count": len(nodes),
        "nodes": nodes,
        "chain_jsonl_path": chain_jsonl_path,
        "enqueue": enqueue,
        "execution_policy": "only deterministic nodes and explicitly requested provider nodes execute",
    }


def build_hyperplex(
    *,
    text: str,
    decision: dict[str, Any],
    lane_plan: dict[str, Any],
    flow: list[dict[str, Any]],
    model_result: dict[str, Any] | None,
) -> dict[str, Any]:
    template = "\n".join(
        [
            f"Route: {decision['lane']}",
            "Reason: " + "; ".join(decision.get("route_reason") or []),
            "Next: " + ("queue/execute slow-lane work" if decision["lane"] == "SLOWLANE" else "answer fast or keep cached"),
        ]
    )
    quotes = [
        {
            "doc_id": "fast_slow_lane_gate",
            "quote": f"deterministic route={decision['lane']} reasons={decision.get('route_reason')}",
        },
        {
            "doc_id": "vibes_groq_contract",
            "quote": "Groq is a separate OpenAI-compatible lane; Mistral/Vibes budget notes apply only to vibe.",
        },
        {
            "doc_id": "flow_edges",
            "quote": json.dumps(flow[:2], sort_keys=True)[:400],
        },
    ]
    synthesis = (
        model_result.get("text", "") if model_result and model_result.get("text") else "No model synthesis executed; deterministic draft only."
    )
    woven = weave_output(
        deterministic_template=template,
        rag_quotes=quotes,
        deepseek_synthesis=synthesis,
        fairyfuse_context={"base_lane": decision["lane"], "input_chars": len(text)},
    )
    woven["lane_name_note"] = (
        "language_membrane uses legacy synthesis lane name `deepseek_q4`; actual provider is recorded in model_synthesis."
    )
    woven["lane_plan"] = lane_plan
    return woven


def orchestrate_text(
    text: str,
    *,
    metadata: dict[str, Any] | None = None,
    lane_hint: str = "auto",
    cache_key: str = "claw_operator",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    receipt_root: str | Path = DEFAULT_RECEIPT_ROOT,
    no_receipt: bool = False,
    store_text: bool = False,
    execute_groq: bool = False,
    groq_model: str = "llama-3.1-8b-instant",
    groq_max_tokens: int = 180,
    groq_temperature: float = 0.0,
    timeout_sec: float = 60.0,
    slow_char_threshold: int = 1800,
    chain_jsonl_out: str | Path | None = None,
    enqueue_chain: bool = False,
    absurd_dir: str | Path = DEFAULT_ABSURD_DIR,
    database_url: str | None = None,
    db_queue_name: str = "luci_operator",
    include_promptflow_prototype: bool = False,
    include_promptflow_gate: bool = False,
    promptflow_flow: str = "04_RUNTIME/promptflow_smoke_flow",
    promptflow_data: str | None = "04_RUNTIME/promptflow_smoke_flow/data.jsonl",
    promptflow_run_id: str | None = None,
    emit_receipt: bool = True,
) -> dict[str, Any]:
    metadata = {"operator_front_door": True, "moa_style_local_metaphor": True, **(metadata or {})}
    packet = {
        "source": "CLAW_OPERATOR_INPUT",
        "target": "CLAW_MOA_ROUTER",
        "text": text,
        "text_sha256": sha256_json({"text": text}),
        "metadata": metadata,
        "lane_hint": lane_hint,
    }
    packet["packet_id"] = sha256_json(packet)
    decision = route_packet(packet, slow_char_threshold=slow_char_threshold)
    importance = importance_score(text, metadata, explicit=None)
    flow = build_flow(decision, "all", "allways")
    cache_args = SimpleNamespace(
        cache_dir=str(cache_dir),
        cache_key=cache_key,
        store_text=store_text,
        flush=False,
        flush_count=5,
        flush_importance=1.0,
    )
    cache = cache_bit(cache_args, packet, decision, flow, importance)
    flush = flush_fastlane(Path(cache_dir), cache_key, force=False, flush_count=5, flush_importance=1.0)
    lane_plan = build_lane_plan(decision, execute_groq=execute_groq)
    route_targets = build_route_targets(packet, decision, lane_plan)
    model_result = None
    blockers: list[str] = []
    if execute_groq:
        groq_provider = lane_plan.get("provider_lanes", {}).get("groq", {})
        if groq_provider.get("status") == "skipped":
            model_result = {
                "provider": "groq",
                "status": "skipped",
                "reason": groq_provider.get("reason", "provider_unavailable"),
                "execute_performed": False,
                "startup_blocker": False,
            }
        else:
            try:
                model_result = run_groq_synthesis(
                    text,
                    decision,
                    lane_plan,
                    model=groq_model,
                    max_tokens=groq_max_tokens,
                    temperature=groq_temperature,
                    timeout_sec=timeout_sec,
                )
                if model_result["returncode"] != 0:
                    blockers.append("groq_synthesis_failed")
            except Exception as exc:  # pragma: no cover - defensive runtime boundary
                blockers.append(f"groq_synthesis_exception:{type(exc).__name__}:{exc}")
                model_result = {"provider": "groq", "execute_performed": False, "error": str(exc)}
    hyperplex = build_hyperplex(
        text=text,
        decision=decision,
        lane_plan=lane_plan,
        flow=flow,
        model_result=model_result,
    )
    task_chain = build_task_chain(
        decision=decision,
        lane_plan=lane_plan,
        cache_path=cache.get("cache_path"),
        model_result=model_result,
        chain_jsonl_out=chain_jsonl_out,
        enqueue_chain=enqueue_chain,
        absurd_dir=absurd_dir,
        database_url=database_url,
        db_queue_name=db_queue_name,
        include_promptflow_prototype=include_promptflow_prototype or include_promptflow_gate,
        promptflow_flow=promptflow_flow,
        promptflow_data=promptflow_data,
        promptflow_run_id=promptflow_run_id,
    )
    payload = {
        "schema": SCHEMA,
        "generated_at": now(),
        "packet": {
            "packet_id": packet["packet_id"],
            "source": packet["source"],
            "target": packet["target"],
            "text_sha256": packet["text_sha256"],
            "text_chars": len(text),
            "text_preview": text[:240],
            "metadata": metadata,
        },
        "input_route": decision,
        "importance": importance,
        "flow": flow,
        "cache": {k: v for k, v in cache.items() if k != "bit"},
        "flush": flush,
        "status": status_payload(Path(cache_dir), cache_key),
        "lane_plan": lane_plan,
        "route_targets": route_targets,
        "routing_fabric": {
            "route_targets": route_targets,
            "provider_lanes": lane_plan.get("provider_lanes", {}),
            "local_model_admission": lane_plan.get("local_model_admission", {}),
        },
        "workflow": {
            "async_capable": True,
            "chain_enqueue": task_chain["enqueue"],
        },
        "task_chain": task_chain,
        "model_synthesis": model_result,
        "hyperplex": hyperplex,
        "model_calls_performed": bool(model_result and model_result.get("execute_performed")),
        "network_calls_performed": bool(model_result and model_result.get("execute_performed")),
        "canonical_graph_writes_performed": False,
        "blockers": blockers,
        "verdict": "PASS" if not blockers else "BLOCKED",
    }
    if store_text:
        payload["packet"]["text"] = text
    if not no_receipt:
        receipt("claw_moa_router", payload, root=Path(receipt_root), emit=emit_receipt)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route one Claw operator packet through deterministic MOA-style lanes.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file")
    parser.add_argument("--metadata-json")
    parser.add_argument("--lane-hint", choices=["auto", "fastlane", "slowlane"], default="auto")
    parser.add_argument("--cache-key", default="claw_operator")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--receipt-root", default=str(DEFAULT_RECEIPT_ROOT))
    parser.add_argument("--store-text", action="store_true")
    parser.add_argument("--no-receipt", action="store_true")
    parser.add_argument("--chain-jsonl-out", help="Optional JSONL export path for the generated task chain.")
    parser.add_argument("--enqueue-chain", action="store_true", help="Queue planned chain nodes in the ABSURD-compatible adapter.")
    parser.add_argument("--absurd-dir", default=str(DEFAULT_ABSURD_DIR), help="ABSURD adapter directory for --enqueue-chain.")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("LUCIDOTA_CONTROL_DATABASE_URL") or os.environ.get("ABSURD_SYSTEM_DATABASE_URL"),
        help="Optional state DB URL for canonical ABSURD queue writes when --enqueue-chain is used.",
    )
    parser.add_argument("--db-queue-name", default="luci_operator")
    parser.add_argument("--include-promptflow-prototype", action="store_true", help="Add a planned PromptFlow visual-prototype node; never a correctness gate.")
    parser.add_argument("--include-promptflow-gate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--promptflow-flow", default="04_RUNTIME/promptflow_smoke_flow")
    parser.add_argument("--promptflow-data", default="04_RUNTIME/promptflow_smoke_flow/data.jsonl")
    parser.add_argument("--promptflow-run-id")
    parser.add_argument("--execute-groq", action="store_true")
    parser.add_argument("--groq-model", default="llama-3.1-8b-instant")
    parser.add_argument("--groq-max-tokens", type=int, default=180)
    parser.add_argument("--groq-temperature", type=float, default=0.0)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = orchestrate_text(
        load_text(args.text, args.text_file),
        metadata=read_json_object(args.metadata_json),
        lane_hint=args.lane_hint,
        cache_key=args.cache_key,
        cache_dir=args.cache_dir,
        receipt_root=args.receipt_root,
        no_receipt=args.no_receipt,
        store_text=args.store_text,
        execute_groq=args.execute_groq,
        groq_model=args.groq_model,
        groq_max_tokens=args.groq_max_tokens,
        groq_temperature=args.groq_temperature,
        timeout_sec=args.timeout_sec,
        chain_jsonl_out=args.chain_jsonl_out,
        enqueue_chain=args.enqueue_chain,
        absurd_dir=args.absurd_dir,
        database_url=args.database_url,
        db_queue_name=args.db_queue_name,
        include_promptflow_prototype=args.include_promptflow_prototype,
        include_promptflow_gate=args.include_promptflow_gate,
        promptflow_flow=args.promptflow_flow,
        promptflow_data=args.promptflow_data,
        promptflow_run_id=args.promptflow_run_id,
        emit_receipt=not args.json,
    )
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print("CLAW_MOA=" + payload["verdict"])
        print("BASE_LANE=" + payload["input_route"]["lane"])
        print("EXECUTION_STATE=" + payload["lane_plan"]["execution_state"])
        if payload.get("receipt_path"):
            print("REPORT_PATH=" + payload["receipt_path"])
    return 0 if payload["verdict"] == "PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
