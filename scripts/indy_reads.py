#!/usr/bin/env python3
"""INDY_READs — GO-25 page-by-page reading game.

INDY_READs is a she: reading companion, margin-noter, judgment collector.

Dynamic library: /home/mfspx/LUCIDOTA/BOOKS
State/data:      /home/mfspx/LUCIDOTA/BOOKS/.indy_reads

No page rewind. One page at a time. Fast heuristic v0.50 parser notes.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import html
import json
import os
import re
import pickle
import resource
import select
import socket
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BOOKS = ROOT / "BOOKS"
DATA = BOOKS / ".indy_reads"
PAGES = DATA / "pages"
WIKI_DIR = DATA / "wiki"
WIKI_PAGES_DIR = WIKI_DIR / "pages"
JOURNAL_DIR = DATA / "private_journal"
CACHE = DATA / "parser_cache"
STATE_PATH = DATA / "state.json"
CSV_PATH = DATA / "indy_reads_judgments.csv"
SCHEMA_PATH = BOOKS / "GO_GAME_GRADING_SCHEMA.json"
ONTOLOGY_PATH = BOOKS / "GO_ACTIVE_TERMS.json"
RIVER_MODEL_PATH = DATA / "indy_reads_attention_model.pkl"
TRANSPORT_SOCKET = Path("/tmp/lucidota_ego.sock")
INDY_CONDUIT_RECEIPT_DIR = ROOT / "05_OUTPUTS" / "indy_conduit"
INDY_OPERATOR_RESPONSE_OUTBOX = ROOT / "05_OUTPUTS" / "indy_conduit" / "indy_operator_responses.jsonl"
INDY_DIRECTIVE_OUTBOX = ROOT / "05_OUTPUTS" / "indy_conduit" / "indy_directives.jsonl"
GOALS_HANDOFF_MD = ROOT / "GOALS" / "CURRENT_HANDOFF.md"
GOALS_NEXT_GOAL_QUEUE = ROOT / "GOALS" / "NEXT_GOAL_QUEUE.json"
INDY_BOOT_PACKET_PATH = ROOT / "04_RUNTIME" / "indy_reads_boot_packet.json"
INDY_ORCHESTRATION_INTENT_PATH = ROOT / "04_RUNTIME" / "indy_reads_orchestration_intent.json"
INDY_BOOT_RECEIPT_DIR = ROOT / "05_OUTPUTS" / "indy_reads_boot"
DB_URL = os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"
ATTENTION_TIMEOUT_SECONDS = 45.0
AUTONOMOUS_TICK_SECONDS = 1.0
PARSER_VERSION = "go_fast_indy_reads_v0.1"
PERSONA_ID = "INDY_READs"
DAEMON_NAME = "indy_reads"
PERSONA_DISPLAY = "INDY_READs"
PERSONA_PRONOUNS = "she/her"
MAIN_AI_PERSONA = True
PERSONA_CONFIG_PATH = ROOT / "04_RUNTIME" / "indy_reads_persona_config.json"
ADAPTER_REGISTRY_PATH = ROOT / "04_RUNTIME" / "indy_reads_adapter_registry.json"
CHARS_PER_PAGE = 2200
SUPPORTED = {".pdf", ".epub", ".mobi", ".azw", ".azw3", ".txt", ".md"}
CANONICAL_BPS = [0, 2, 4, 6, 10, 50, 69, 150]


def _repo_venv_site_packages() -> list[Path]:
    candidates: list[Path] = []
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for lib_root in (ROOT / ".venv" / "lib", ROOT / ".venv" / "lib64"):
        candidate = lib_root / py_version / "site-packages"
        if candidate.exists():
            candidates.append(candidate)
    return candidates


def _bootstrap_psycopg() -> Any | None:
    try:
        import psycopg as psycopg_module  # type: ignore[import-not-found]

        globals()["_PSYCOPG_BOOTSTRAP_MODE"] = "system"
        return psycopg_module
    except Exception:
        for site_packages in _repo_venv_site_packages():
            site_packages_str = str(site_packages)
            if site_packages_str not in sys.path:
                sys.path.insert(0, site_packages_str)
        try:
            import psycopg as psycopg_module  # type: ignore[import-not-found]

            globals()["_PSYCOPG_BOOTSTRAP_MODE"] = "repo_venv_site_packages"
            return psycopg_module
        except Exception:
            globals()["_PSYCOPG_BOOTSTRAP_MODE"] = "unavailable"
            return None


psycopg = _bootstrap_psycopg()  # type: ignore[assignment]
_PSYCOPG_BOOTSTRAP_MODE = globals().get("_PSYCOPG_BOOTSTRAP_MODE", "unavailable")
_STARTUP_WARNING_EMITTED = False


def emit_startup_warning_once(message: str) -> dict[str, Any]:
    global _STARTUP_WARNING_EMITTED
    if _STARTUP_WARNING_EMITTED:
        return {"ok": True, "emitted": False}
    _STARTUP_WARNING_EMITTED = True
    warning = {
        "schema": "lucidota.indy_reads.startup_warning.v1",
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "severity": "warning",
        "source": "indy_reads",
        "message": message,
        "bootstrap_mode": _PSYCOPG_BOOTSTRAP_MODE,
    }
    INDY_CONDUIT_RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    warning_path = INDY_CONDUIT_RECEIPT_DIR / "indy_startup_warnings.jsonl"
    with warning_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(warning, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    print(f"[indy_reads warning] {message}", file=sys.stderr)
    if psycopg is not None:
        try:
            with psycopg.connect(DB_URL) as conn:  # type: ignore[union-attr]
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO ironclaw.waking_dialogue_stream
                          (comms_channel, sender_id, room_id, event_id, raw_text, clean_text, extracted_entities, processed_status, receipt_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                        ON CONFLICT (comms_channel, event_id) DO UPDATE SET
                          raw_text = EXCLUDED.raw_text,
                          clean_text = EXCLUDED.clean_text,
                          extracted_entities = EXCLUDED.extracted_entities,
                          processed_status = EXCLUDED.processed_status,
                          receipt_id = EXCLUDED.receipt_id,
                          updated_at = now()
                        """,
                        (
                            "matrix",
                            "indy_reads_bootstrap",
                            "!indy_command_deck:localhost",
                            "indy-startup:" + sha_text(json.dumps(warning, sort_keys=True, ensure_ascii=False, default=str))[:24],
                            f"[BOOTSTRAP WARNING] {message}",
                            f"[BOOTSTRAP WARNING] {message}",
                            json.dumps({"warning": ["psycopg_bootstrap"]}, sort_keys=True),
                            "queued",
                            "indy-startup-receipt:" + sha_text(message)[:16],
                        ),
                    )
                conn.commit()
        except Exception:
            pass
    return {"ok": True, "emitted": True, "warning_path": str(warning_path), "bootstrap_mode": _PSYCOPG_BOOTSTRAP_MODE}

DEFAULT_PERSONA_CONFIG: dict[str, Any] = {
    "schema": "lucidota.indy_reads.persona_config.v1",
    "persona_id": PERSONA_ID,
    "display_name": PERSONA_DISPLAY,
    "pronouns": PERSONA_PRONOUNS,
    "main_ai_persona": MAIN_AI_PERSONA,
    "boot_packet_path": "04_RUNTIME/indy_reads_boot_packet.json",
    "active_ontology": {
        "name": "GO",
        "expanded_name": "Global Ontology",
        "terms_path": "BOOKS/GO_ACTIVE_TERMS.json",
    },
    "mission": "Page-locked reading companion, margin-noter, and judgment collector for the GO reading game.",
    "runtime_truth_refs": [
        "lucidota_control.active_operation_mode",
        "lucidota_canon.manual_current",
        "lucidota_canon.root_orchestrator_current",
        "lucidota_canon.workload_audit_current",
        "lucidota_canon.workload_audit_telemetry_current",
        "lucidota_canon.indy_reads_self_model",
        "lucidota_canon.indy_reads_llmwiki_entry",
        "lucidota_canon.indy_reads_metacognition_current",
    ],
    "permissions": {
        "read_paths": [
            "BOOKS",
            "BOOKS/.indy_reads",
            "04_RUNTIME/indy_reads_adapter_registry.json",
            "04_RUNTIME/indy_reads_boot_packet.json",
            "04_RUNTIME/INDY_READS/indy_reads_service_manifest.json",
        ],
        "write_paths": [
            "BOOKS/.indy_reads",
            "04_RUNTIME/indy_reads_persona_config.json",
            "04_RUNTIME/indy_reads_adapter_registry.json",
            "04_RUNTIME/indy_reads_boot_packet.json",
        ],
        "may_update_adapter_registry": True,
        "may_edit_active_go_terms": False,
        "may_touch_graph_core_sql": False,
        "may_create_doctrine_markdown": False,
    },
    "memory_boundaries": {
        "page_locked_reading": True,
        "no_forward_book_claims": True,
        "persistent_memory_paths": ["BOOKS/.indy_reads/state.json", "BOOKS/.indy_reads/indy_reads_judgments.csv"],
        "cache_paths": ["BOOKS/.indy_reads/pages", "BOOKS/.indy_reads/parser_cache"],
        "external_truth_default": "unverified_until_evidence",
    },
}

DEFAULT_ADAPTER_REGISTRY: dict[str, Any] = {
    "schema": "lucidota.indy_reads.adapter_registry.v1",
    "registry_id": "indy_reads_lora_adapter_candidates",
    "owner_persona": PERSONA_ID,
    "active_ontology": "GO / Global Ontology",
    "write_policy": "append_or_update_candidates_only; no graph-core SQL writes; runtime bootstrap lanes are receipt-bound",
    "default_base_model": "deepseek-1.5b-indy_reads-reads",
    "runtime_fabric": {
        "primary_language_cortex": {
            "model_lane": "bonsai_q1_0",
            "shared_weight": True,
            "logical_slots": ["slot_0", "slot_1"],
            "slot_0_role": "synthesis",
            "slot_1_role": "skeptic_verifier",
            "default_context_tokens": 10_000,
            "max_context_tokens_after_proof": 16_000,
            "prefix_cache_required": True,
            "quantized_kv_preferred": True,
        },
        "reflex_bank": {
            "model_family": "needle-26m",
            "logical_lanes": 6,
            "shared_weight": True,
        },
        "state_watcher": {
            "model_lane": "mamba_cpu",
            "role": "state_flow_watcher",
        },
        "recursive_bank": {
            "model_family": "trm",
            "logical_workers": 20,
        },
        "rolling_language": {
            "model_lane": "rwkv_world_400m",
            "role": "journal_and_continuity",
        },
    },
    "candidates": [
        {
            "adapter_id": "indy_reads_bootstrap_bonsai_v0",
            "kind": "prompt_or_lora",
            "target_model_id": "bonsai_q1_0",
            "status": "active",
            "training_sources": [
                "04_RUNTIME/indy_reads_boot_packet.json",
                "04_RUNTIME/INDY_READS/indy_reads_service_manifest.json",
                "GOALS/CURRENT_HANDOFF.md",
                "GOALS/GOAL_LOG.md",
            ],
            "permission_scope": "runtime_bootstrap_only",
            "memory_boundary": "two-slot synthesis + skeptic verifier; no giant context dump",
            "notes": "Primary Indy_READs bootstrap lane on the GTX 1650: one shared Bonsai 8B weight, two logical slots, quantized KV preferred.",
        },
        {
            "adapter_id": "indy_reads_go_margin_v0",
            "kind": "lora",
            "target_model_id": "deepseek-1.5b-indy_reads-reads",
            "status": "planned",
            "training_sources": ["BOOKS/.indy_reads/indy_reads_judgments.csv"],
            "permission_scope": "private_local_only",
            "memory_boundary": "page_locked_go_margin_notes",
            "notes": "Candidate adapter for INDY_READs GO margin-note style; not trained yet.",
        },
        {
            "adapter_id": "indy_reads_go_router_v0",
            "kind": "prompt_or_lora",
            "target_model_id": "deepseek-1.5b-indy_reads-reads",
            "status": "watch",
            "training_sources": ["BOOKS/GO_ACTIVE_TERMS.json"],
            "permission_scope": "terms_read_only",
            "memory_boundary": "term-routing only; no doctrine expansion",
            "notes": "Lightweight candidate for GO term routing and adapter browsing.",
        },
    ],
}

DEFAULT_BOOT_PACKET: dict[str, Any] = {
    "schema": "lucidota.indy_reads.boot_packet.v1",
    "owner": "indy_reads_runtime",
    "runtime": "ironclaw_local_models",
    "model_lane": "bonsai_q1_0",
    "cloud_models": {
        "allowed_during_build": True,
        "receipt_required": True,
        "uses": [
            "architecture_synthesis",
            "contradiction_checks",
            "integration_planning",
            "document_review",
            "ontology_mapping",
            "slop_pruning",
        ],
    },
    "model_fabric": {
        "primary_cortex": {
            "model_lane": "bonsai_q1_0",
            "shared_weight": True,
            "logical_slots": ["slot_0", "slot_1"],
            "slot_0_role": "synthesis",
            "slot_1_role": "skeptic_verifier",
            "default_context_tokens": 10_000,
            "max_context_tokens_after_proof": 16_000,
            "prefix_cache_required": True,
            "quantized_kv_preferred": True,
            "usable_vram_budget_mb": 3_500,
            "stable_usable_vram_mb": 3_300,
        },
        "reflex_bank": {
            "model_family": "needle-26m",
            "logical_lanes": 6,
            "shared_weight": True,
        },
        "state_watcher": {
            "model_family": "mamba",
            "model_lane": "mamba_cpu",
            "role": "state_flow_watcher",
        },
        "recursive_bank": {
            "model_family": "trm",
            "logical_workers": 20,
            "role": "constraint_and_reconciliation",
        },
        "rolling_language": {
            "model_family": "rwkv",
            "model_lane": "rwkv_world_400m",
            "role": "journal_and_continuity",
        },
    },
    "boot_target": "indy_reads_self_boot",
    "boot_objective": "Initialize Indy_READs as her own runtime lane, not as a Codex subagent, using live DB/manual/workload/mode surfaces plus receipt-bound local model output.",
    "surface_refs": {
        "active_operation_mode": "lucidota_control.active_operation_mode",
        "manual_current": "lucidota_canon.manual_current",
        "root_orchestrator_current": "lucidota_canon.root_orchestrator_current",
        "workload_audit_current": "lucidota_canon.workload_audit_current",
        "workload_audit_telemetry_current": "lucidota_canon.workload_audit_telemetry_current",
        "command_registry": "lucidota_canon.command_registry",
        "capability_current": "lucidota_canon.capability_current",
        "capability_registry": "lucidota_canon.capability_registry",
        "ontology_registry": "BOOKS/GO_ACTIVE_TERMS.json",
    },
    "work_orders": [
        "Read the live DB/manual/root/workload/mode surfaces before claiming anything.",
        "Write Indy_READs first entries herself: self_model, LLMWIKI, hunch_log, system_map, mistake_ledger, learning_queue.",
        "Keep receipts for any model/provider call and any agent work.",
        "Mark unknown debt explicitly when proof is missing.",
        "Do not let Codex impersonate Indy_READs.",
        "Use the Bonsai 8B shared weight with two logical slots: synthesis and skeptic/verifier.",
        "Use 10k context by default and only expand to 16k after proof.",
        "Prefer quantized KV and prefix cache if the lane supports it; keep the GTX 1650 budget under control.",
        "Cloud models are allowed during build and verification, and every call needs a receipt.",
        "Exchange ontology packets, not prose blobs, between models and agents.",
    ],
    "receipt_requirements": {
        "agent_work_receipt_required": True,
        "model_invocation_receipt_required": True,
        "provider_call_receipt_required": False,
        "workload_audit_row_required": True,
    },
    "llmwiki": {
        "ownership_statement": "LLMWIKI belongs to Indy_READs as her metacognition notebook. It is not canon truth until promoted with DB + receipt proof.",
        "promotion_rule": "Any operationally important wiki note must be promoted through DB/receipt/graph gates before becoming truth.",
    },
    "first_entries": {
        "self_model": True,
        "llmwiki_entry": True,
        "hunch_log": True,
        "system_map": True,
        "mistake_ledger": True,
        "learning_queue": True,
        "metacognition_current": True,
    },
    "prompt_instructions": [
        "Return compact JSON only.",
        "Do not claim receipt-backed work without receipts.",
        "Use proof_status PROVEN, PARTIAL, UNKNOWN, or CONTRADICTED exactly.",
        "Include evidence_refs and db_refs arrays for every entry.",
        "When uncertain, write UNKNOWN debt instead of guessing.",
    ],
    "evidence_refs": [
        "AGENTS.md",
        "CLAUDE.md",
        "GOALS/CURRENT_HANDOFF.md",
        "GOALS/GOAL_LOG.md",
        "scripts/lucidota_start_indy_reads_watcher.sh",
        "scripts/indy_daemon.py",
        "scripts/indy_reads.py",
    ],
}

DEFAULT_ORCHESTRATION_INTENT: dict[str, Any] = {
    "schema": "lucidota.indy_reads.orchestration_intent.v1",
    "actor_id": PERSONA_ID,
    "provider_key": "local_model",
    "provider_kind": "local_runtime",
    "workload_type": "orchestration",
    "model_id": "bonsai_q1_0",
    "model_family": "bonsai",
    "role": "big_brain_orchestration",
    "takeover_mode": False,
    "fallback_provider_key": "local_model",
    "fallback_model_id": "bonsai_q1_0",
    "fallback_reason": "default_to_local_bonsai_when_cloud_orchestration_is_unavailable",
    "source": "indy_boot_default",
    "updated_at": "",
    "notes": "Local 2x8B Bonsai is the default; explicit cloud keys may override for orchestration only.",
}


def write_json_if_missing(path: Path, data: dict[str, Any]) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json_or_default(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def load_persona_config() -> dict[str, Any]:
    return load_json_or_default(PERSONA_CONFIG_PATH, DEFAULT_PERSONA_CONFIG)


def load_adapter_registry() -> dict[str, Any]:
    return load_json_or_default(ADAPTER_REGISTRY_PATH, DEFAULT_ADAPTER_REGISTRY)


def load_go_terms() -> list[dict[str, str]]:
    try:
        data = json.loads(ONTOLOGY_PATH.read_text(encoding="utf-8"))
        return data.get("terms", [])
    except (OSError, json.JSONDecodeError, TypeError):
        return []

GO_TERMS = load_go_terms()
GO_BY_TERM = {t["term"]: t for t in GO_TERMS}
GO_BY_ID = {t["id"]: t for t in GO_TERMS}

CORE_TERMS = [t["term"] for t in GO_TERMS]
MYTHIC_TERMS = {"NAUGHTY", "NICE"}


def ensure_dirs() -> None:
    for p in [BOOKS, DATA, PAGES, WIKI_DIR, WIKI_PAGES_DIR, JOURNAL_DIR, CACHE, ROOT / "04_RUNTIME"]:
        p.mkdir(parents=True, exist_ok=True)
    write_json_if_missing(PERSONA_CONFIG_PATH, DEFAULT_PERSONA_CONFIG)
    write_json_if_missing(ADAPTER_REGISTRY_PATH, DEFAULT_ADAPTER_REGISTRY)
    write_json_if_missing(INDY_BOOT_PACKET_PATH, DEFAULT_BOOT_PACKET)
    write_json_if_missing(INDY_ORCHESTRATION_INTENT_PATH, DEFAULT_ORCHESTRATION_INTENT)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            writer.writeheader()


def clear() -> None:
    os.system("clear" if os.name == "posix" else "cls")


def pause(msg: str = "ENTER continues...") -> None:
    input(f"\n{msg}")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", s).strip("_").lower()[:96] or "book"


def rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path.resolve())


def write_journal_entry(
    *,
    title: str,
    body: str,
    kind: str = "note",
    journal_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    target_dir = Path(journal_dir) if journal_dir is not None else JOURNAL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_title = slug(title)
    path = target_dir / f"{now().replace(':', '').replace('-', '').replace('T', '_').replace('+', '_')}_{safe_title}.md"
    payload = {
        "schema": "lucidota.indy_reads.journal_entry.v1",
        "kind": kind,
        "title": title,
        "body": body,
        "created_at": now(),
        "source": "indy_reads",
        "path": rel_or_abs(path),
        "abs_path": str(path.resolve()),
    }
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"- kind: {kind}",
                f"- created_at: {payload['created_at']}",
                "",
                body.strip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    payload["sha256"] = sha_file(path)
    return payload


def write_wiki_page(
    *,
    title: str,
    body: str,
    wiki_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    target_dir = Path(wiki_dir) if wiki_dir is not None else WIKI_PAGES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{slug(title)}.md"
    page = {
        "schema": "lucidota.indy_reads.wiki_page.v1",
        "title": title,
        "body": body,
        "created_at": now(),
        "source": "indy_reads",
        "path": rel_or_abs(path),
        "abs_path": str(path.resolve()),
    }
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                body.strip(),
                "",
            ]
        ),
        encoding="utf-8",
    )
    page["sha256"] = sha_file(path)
    return page


@dataclass
class Book:
    id: str
    name: str
    path: str
    ext: str
    size_bytes: int


def library() -> list[Book]:
    ensure_dirs()
    rows = []
    for p in sorted(BOOKS.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            rows.append(Book(slug(p.stem), p.name, str(p), p.suffix.lower(), p.stat().st_size))
    return rows


def load_state() -> dict[str, Any]:
    ensure_dirs()
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    else:
        state = {"books": {}, "active_book_id": ""}
    state.setdefault("books", {})
    state.setdefault("active_book_id", "")
    state.setdefault("slow_lane", {})
    state["slow_lane"].setdefault("ingestion_batch_size", 4)
    state["slow_lane"].setdefault("transport_socket", str(TRANSPORT_SOCKET))
    state["slow_lane"].setdefault("last_autonomous_tick_at", "")
    return state


def save_state(st: dict[str, Any]) -> None:
    ensure_dirs()
    STATE_PATH.write_text(json.dumps(st, indent=2, sort_keys=True), encoding="utf-8")


def get_book_state(st: dict[str, Any], b: Book) -> dict[str, Any]:
    bs = st.setdefault("books", {}).setdefault(b.id, {})
    bs.setdefault("current_page", 1)
    bs.setdefault("completed_pages", [])
    bs.setdefault("source_sha256", sha_file(Path(b.path)))
    bs.setdefault("last_opened", now())
    bs.setdefault("name", b.name)
    bs.setdefault("path", b.path)
    return bs


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def strip_html(raw: str) -> str:
    raw = re.sub(r"<script.*?</script>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"<style.*?</style>", " ", raw, flags=re.I | re.S)
    raw = re.sub(r"</p>|<br\s*/?>|</h\d>|</div>", "\n\n", raw, flags=re.I)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = html.unescape(raw)
    raw = re.sub(r"[ \t\r\f\v]+", " ", raw)
    raw = re.sub(r"\n\s+", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def epub_text(path: Path) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith((".xhtml", ".html", ".htm"))]
        # Sort path-wise; good enough for game reading. Future: parse OPF spine.
        for name in sorted(names):
            try:
                raw = z.read(name).decode("utf-8", errors="ignore")
            except (KeyError, RuntimeError, UnicodeError, zipfile.BadZipFile):
                continue
            txt = strip_html(raw)
            if txt:
                parts.append(txt)
    return "\n\n".join(parts)


def whole_text_for_book(path: Path) -> tuple[str, str]:
    ext = path.suffix.lower()
    if ext == ".epub":
        return epub_text(path), "epub-zip-html"
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore"), "text"
    if ext in {".mobi", ".azw", ".azw3"}:
        if shutil.which("ebook-convert"):
            out = CACHE / (slug(path.stem) + ".txt")
            cp = run(["ebook-convert", str(path), str(out)])
            if cp.returncode == 0 and out.exists() and out.stat().st_size > 0:
                return out.read_text(encoding="utf-8", errors="ignore"), "ebook-convert"
        cp = run(["strings", "-n", "5", str(path)])
        return cp.stdout, "strings-fallback"
    raise ValueError(f"whole_text_for_book unsupported for {ext}")


def split_pages(text: str, chars: int = CHARS_PER_PAGE) -> list[str]:
    paras = re.split(r"\n\s*\n", text)
    pages: list[str] = []
    cur = ""
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if cur and len(cur) + len(para) + 2 > chars:
            pages.append(cur.strip())
            cur = para
        else:
            cur = (cur + "\n\n" + para).strip() if cur else para
    if cur:
        pages.append(cur.strip())
    return pages or [text[:chars]]


def extract_page(book: Book, page: int) -> dict[str, Any]:
    path = Path(book.path)
    book_dir = PAGES / book.id
    book_dir.mkdir(parents=True, exist_ok=True)
    page_file = book_dir / f"p{page:04d}.json"
    if page_file.exists():
        return json.loads(page_file.read_text(encoding="utf-8"))

    if book.ext == ".pdf":
        if not shutil.which("pdftotext"):
            raise RuntimeError("pdftotext missing")
        cp = run(["pdftotext", "-f", str(page), "-l", str(page), book.path, "-"])
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr.strip() or "pdftotext failed")
        text = cp.stdout.strip()
        method = "pdftotext-page"
    else:
        txt, method = whole_text_for_book(path)
        pages = split_pages(txt)
        if page < 1 or page > len(pages):
            raise RuntimeError(f"page {page} out of range 1..{len(pages)} by {method}")
        text = pages[page - 1]

    obj = {
        "book_id": book.id,
        "book_name": book.name,
        "book_path": book.path,
        "page": page,
        "text": text,
        "page_hash": sha_text(text),
        "source_sha256": sha_file(path),
        "extract_method": method,
        "chars": len(text),
        "created_at": now(),
        "do_not_infer_beyond_page": True,
    }
    page_file.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return obj


def sentenceish(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    bits = re.split(r"(?<=[.!?])\s+", cleaned)
    return [b.strip() for b in bits if len(b.strip()) > 20]


def fast_parse(page: dict[str, Any]) -> dict[str, Any]:
    cache = CACHE / f"{page['book_id']}_p{int(page['page']):04d}_{page['page_hash'][:12]}.json"
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        cached.setdefault("persona_id", PERSONA_ID)
        cached.setdefault("main_ai_persona", MAIN_AI_PERSONA)
        return cached
    text = page["text"]
    low = text.lower()
    sents = sentenceish(text)
    local_gates = ["EVIDENCE", "CLAIM"]
    terms = ["EVIDENCE", "CLAIM", "TERM"]
    if any(k in low for k in ["source", "according to", "reported", "archive", "document", "book"]):
        terms.append("SOURCE")
    if any(k in low for k in ["witness", "saw", "observed", "testified"]):
        terms.append("WITNESS")
    if any(k in low for k in ["rumour", "rumor", "alleged", "apparently", "they say"]):
        terms.append("RUMOUR")
    if any(k in low for k in ["threat", "risk", "danger", "coerc", "harm", "exploit"]):
        terms.append("THREAT")
    if any(k in low for k in ["license", "licence", "permit", "certif"]):
        terms.append("LICENSE")
    if any(k in low for k in ["regulator", "government", "ministry", "agency", "bureau"]):
        terms.extend(["REGULATOR", "GOVERNMENT"])
    if any(k in low for k in ["law", "rule", "statute", "regulation", "policy"]):
        terms.extend(["LAW", "RULE"])
    if any(k in low for k in ["where", "street", "avenue", "road", "city", "glasgow", "malta"]):
        terms.append("LOCATION")
    if any(k in low for k in ["said", "asked", "replied", "told"]):
        terms.append("SIGNAL")
    if any(k in low for k in ["because", "therefore", "so that", "result", "caused"]):
        terms.append("RELATIONSHIP")
        local_gates.append("RELATIONSHIP")
    if re.search(r"\b\d{1,2}:\d{2}\b|monday|tuesday|wednesday|thursday|friday|saturday|sunday|september|january|february|march", low):
        terms.append("TIME")
        local_gates.append("TIME")
    if any(k in low for k in ["dream", "like", "as if", "metaphor", "song", "game", "story"]):
        terms.extend(["PATTERN", "GLOW"])
    if page.get("extract_method") == "strings-fallback" or len(re.findall(r"\b[a-zA-Z]{1,2}\b", text)) > 80:
        terms.extend(["SIGNAL", "COMMENT"])
        local_gates.append("SIGNAL")
    # preserve order unique
    seen = set(); terms = [p for p in terms if not (p in seen or seen.add(p))]
    seen = set(); local_gates = [p for p in local_gates if not (p in seen or seen.add(p))]

    notes = []
    notes.append("PAGE_LOCK: interpreting this page/chunk only; no forward-book claims.")
    if page.get("extract_method") == "strings-fallback":
        notes.append("MOBI_STRINGS_HARD_MODE: extraction is noisy; treat as noise-resistance round, not clean prose custody.")
    if sents:
        notes.append("TEXT_SAYS: " + sents[0][:260])
    if len(sents) > 1:
        notes.append("CARRY_FORWARD_THREAD: " + sents[1][:220])
    notes.append("GO_ROUTE: " + " ∩ ".join(terms[:5]) + " → PAGE_LEVEL_READING_PACKET")

    parser = {
        "parser_version": PARSER_VERSION,
        "persona_id": PERSONA_ID,
        "main_ai_persona": MAIN_AI_PERSONA,
        "packet_id": f"indy::{page['book_id']}::p{int(page['page']):04d}",
        "raw_text_anchor": sents[0][:300] if sents else text[:300],
        "local_gates": local_gates,
        "terms": terms,
        "route": {
            "anchor": terms[0],
            "operator": "∩",
            "vector": terms[1:5],
            "resolution": "PAGE_LEVEL_READING_PACKET",
        },
        "ternary_state": {"text_presence": 1, "internal_scope": 1, "external_truth": 0},
        "claim_lifecycle": "CLAIM_UNVERIFIED",
        "confidence_bps": 10 if page.get("extract_method") == "strings-fallback" else 50,
        "falsifier": "Later page or cleaner source extraction contradicts this page-level interpretation.",
        "notes": notes,
        "mythic_terms_available_but_not_forced": sorted(MYTHIC_TERMS),
        "created_at": now(),
    }
    cache.write_text(json.dumps(parser, indent=2, sort_keys=True), encoding="utf-8")
    return parser


CSV_FIELDS = [
    "timestamp", "book_id", "book_name", "page", "page_hash", "extract_method", "parser_version",
    "packet_id", "parser_terms", "parser_bps", "decision", "score", "score_label",
    "term_correction", "notes", "repair_instruction", "favorite_line", "confusion", "raw_csv_json",
]


def score_label(score: int) -> str:
    if score >= 95: return "CAKE"
    if score >= 80: return "COOKED"
    if score >= 60: return "NEEDS_REPAIR"
    if score >= 30: return "SLOP_DETECTED"
    return "ARCHON_BAIT"


def append_csv(row: dict[str, Any]) -> None:
    ensure_dirs()
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def wrap_print(text: str, width: int = 92, max_lines: int | None = None) -> None:
    lines: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=width, replace_whitespace=True))
    if max_lines is None:
        max_lines = len(lines)
    for line in lines[:max_lines]:
        print(line)
    if len(lines) > max_lines:
        print(f"\n…[{len(lines)-max_lines} more lines hidden]")


def banner(subtitle: str) -> None:
    clear()
    print("╔" + "═" * 92 + "╗")
    print("║" + "INDY_READs — main AI persona (she/her)".center(92) + "║")
    print("║" + subtitle.center(92) + "║")
    print("╚" + "═" * 92 + "╝")


def pick_book(st: dict[str, Any]) -> Book | None:
    books = library()
    while True:
        banner("BOOKS — dynamic /LUCIDOTA/BOOKS library")
        if not books:
            print(f"No books in {BOOKS}")
            pause(); return None
        for i, b in enumerate(books, 1):
            bs = st.get("books", {}).get(b.id, {})
            cur = bs.get("current_page", 1)
            mark = " ← active" if st.get("active_book_id") == b.id else ""
            print(f"{i:>2}. {b.name} [{b.ext}] — page {cur}{mark}")
        print("\np. GO term browser   a. adapter candidates   q. quit")
        ans_raw = timed_input("\nPick book: ", ATTENTION_TIMEOUT_SECONDS)
        if ans_raw is TIMEOUT_SENTINEL:
            socket_active = transport_socket_active()
            if socket_active:
                tune_and_record_heartbeat(st, None, None, None, score=None, terminal_active=False)
                print("\nCollaborative companion mode: local transport socket is active.")
            else:
                run_autonomous_slow_lane_tick(st, None, None, None)
                print("\nAutonomous slot claimed while waiting for a book selection.")
            continue
        ans = str(ans_raw).strip().lower()
        if ans == "q": return None
        if ans == "p":
            term_browser(); continue
        if ans == "a":
            adapter_browser(); continue
        try:
            n = int(ans)
            if 1 <= n <= len(books):
                b = books[n-1]
                st["active_book_id"] = b.id
                get_book_state(st, b)["last_opened"] = now()
                save_state(st)
                return b
        except ValueError:
            print("Invalid numeric choice.")


def term_browser() -> None:
    while True:
        banner("GO TERM BROWSER — use @number or #TERM")
        print("Examples: @01, @13, @37, #EVIDENCE, #ANOMALY, search words like law")
        q = input("lookup> ").strip()
        if q.lower() in {"q", "quit", "back", ""}:
            return
        results = []
        if q.startswith("@") and q[1:].isdigit():
            key = "@" + q[1:].zfill(2)
            if key in GO_BY_ID:
                t = GO_BY_ID[key]
                results = [t]
        elif q.startswith("#"):
            target = q[1:].upper()
            results = [t for t in GO_TERMS if target in t["term"]]
        else:
            target = q.upper()
            results = [t for t in GO_TERMS if target in t["term"] or target in t.get("definition", "").upper()]
        if not results:
            print("No hit.")
        else:
            for t in results[:40]:
                print(f"{t['id']} #{t['term']} — {t.get('definition','')}")
        pause()



def adapter_browser() -> None:
    cfg = load_persona_config()
    reg = load_adapter_registry()
    intent = load_orchestration_intent()
    banner("ADAPTER CANDIDATES — INDY_READs browse/update seed")
    print(f"Persona: {cfg.get('display_name', PERSONA_DISPLAY)} ({cfg.get('pronouns', PERSONA_PRONOUNS)})")
    print(f"Main AI persona: {cfg.get('main_ai_persona', MAIN_AI_PERSONA)}")
    ontology = cfg.get("active_ontology", {})
    print(f"Ontology: {ontology.get('name', 'GO')} — {ontology.get('expanded_name', 'Global Ontology')}")
    print(f"Registry: {ADAPTER_REGISTRY_PATH}")
    print(f"Policy: {reg.get('write_policy', '')}\n")
    print(f"Current orchestration intent: {current_orchestration_intent_summary(intent)}")
    print(f"Intent path: {INDY_ORCHESTRATION_INTENT_PATH}\n")
    for c in reg.get("candidates", []):
        print(f"- {c.get('adapter_id')} [{c.get('kind')}/{c.get('status')}]")
        print(f"  target: {c.get('target_model_id', reg.get('default_base_model', ''))}")
        print(f"  scope:  {c.get('permission_scope', '')}")
        print(f"  memory: {c.get('memory_boundary', '')}")
        if c.get("notes"):
            print(f"  notes:  {c.get('notes')}")
    pause()


def db_available() -> bool:
    if _PSYCOPG_BOOTSTRAP_MODE == "repo_venv_site_packages" and not _STARTUP_WARNING_EMITTED:
        emit_startup_warning_once("psycopg was missing from system python; bootstrapped from repo .venv site-packages")
    return psycopg is not None


def import_indy_conduit_driver():
    try:
        import indy_conduit_driver  # type: ignore

        return indy_conduit_driver
    except Exception:
        import scripts.indy_conduit_driver as indy_conduit_driver  # type: ignore

        return indy_conduit_driver


def fetch_one_json(sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any]:
    if not db_available():
        return {}
    try:
        with psycopg.connect(DB_URL) as conn:  # type: ignore[union-attr]
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                if not row:
                    return {}
                value = row[0]
                if isinstance(value, dict):
                    return value
                if isinstance(value, str):
                    return json.loads(value)
                return dict(value) if value is not None else {}
    except Exception:
        return {}


def fetch_rows_json(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    if not db_available():
        return []
    try:
        with psycopg.connect(DB_URL) as conn:  # type: ignore[union-attr]
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            if not row:
                continue
            value = row[0]
            if isinstance(value, dict):
                out.append(value)
            elif isinstance(value, str):
                out.append(json.loads(value))
        return out
    except Exception:
        return []


def fetch_scalar(sql: str, params: tuple[Any, ...] = ()) -> Any:
    if not db_available():
        return None
    try:
        with psycopg.connect(DB_URL) as conn:  # type: ignore[union-attr]
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
                if not row:
                    return None
                return row[0]
    except Exception:
        return None


def execute_sql(sql: str, params: tuple[Any, ...] = ()) -> bool:
    if not db_available():
        return False
    try:
        with psycopg.connect(DB_URL) as conn:  # type: ignore[union-attr]
            with conn.cursor() as cur:
                cur.execute(sql, params)
            conn.commit()
        return True
    except Exception:
        return False


def load_boot_packet() -> dict[str, Any]:
    return load_json_or_default(INDY_BOOT_PACKET_PATH, DEFAULT_BOOT_PACKET)


def load_orchestration_intent() -> dict[str, Any]:
    intent = load_json_or_default(INDY_ORCHESTRATION_INTENT_PATH, DEFAULT_ORCHESTRATION_INTENT)
    merged = dict(DEFAULT_ORCHESTRATION_INTENT)
    merged.update(intent)
    merged.setdefault("updated_at", "")
    return merged


def save_orchestration_intent(intent: dict[str, Any]) -> dict[str, Any]:
    ensure_dirs()
    payload = dict(DEFAULT_ORCHESTRATION_INTENT)
    payload.update(intent)
    payload["updated_at"] = now()
    INDY_ORCHESTRATION_INTENT_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    persist_orchestration_intent_db(payload)
    return payload


def persist_orchestration_intent_db(intent: dict[str, Any]) -> bool:
    if not db_available():
        return False
    try:
        return execute_sql(
            """
            INSERT INTO lucidota_indy.indy_reads_orchestration_intent_state (
                state_key,
                actor_id,
                provider_key,
                provider_kind,
                workload_type,
                model_id,
                model_family,
                role,
                takeover_mode,
                fallback_provider_key,
                fallback_model_id,
                source,
                notes,
                updated_at
            ) VALUES (
                'indy_reads_orchestration_intent',
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now()
            )
            ON CONFLICT (state_key) DO UPDATE SET
                actor_id = EXCLUDED.actor_id,
                provider_key = EXCLUDED.provider_key,
                provider_kind = EXCLUDED.provider_kind,
                workload_type = EXCLUDED.workload_type,
                model_id = EXCLUDED.model_id,
                model_family = EXCLUDED.model_family,
                role = EXCLUDED.role,
                takeover_mode = EXCLUDED.takeover_mode,
                fallback_provider_key = EXCLUDED.fallback_provider_key,
                fallback_model_id = EXCLUDED.fallback_model_id,
                source = EXCLUDED.source,
                notes = EXCLUDED.notes,
                updated_at = now()
            """,
            (
                str(intent.get("actor_id") or PERSONA_ID),
                str(intent.get("provider_key") or "local_model"),
                str(intent.get("provider_kind") or "local_runtime"),
                str(intent.get("workload_type") or "orchestration"),
                str(intent.get("model_id") or "bonsai_q1_0"),
                str(intent.get("model_family") or "bonsai"),
                str(intent.get("role") or "big_brain_orchestration"),
                bool(intent.get("takeover_mode")),
                str(intent.get("fallback_provider_key") or "local_model"),
                str(intent.get("fallback_model_id") or "bonsai_q1_0"),
                str(intent.get("source") or "operator"),
                str(intent.get("notes") or ""),
            ),
        )
    except Exception:
        return False


def normalize_provider_key(provider_key: str | None) -> str:
    key = (provider_key or "").strip().lower().replace(" ", "_").replace("-", "_")
    alias_map = {
        "google": "gemini",
        "vertex": "gemini",
        "gemini_paid": "gemini",
        "gemini_api": "gemini",
        "vibes": "vibe",
        "mistral": "vibe",
        "bonsai": "local_model",
        "local": "local_model",
        "local_bonsai": "local_model",
    }
    return alias_map.get(key, key or "local_model")


def provider_default_model(provider_key: str) -> str:
    provider_key = normalize_provider_key(provider_key)
    return {
        "groq": "llama-3.3-70b-versatile",
        "gemini": "gemini-2.5-flash",
        "vibe": "codestral",
        "codex": "gpt-5.4-mini",
        "local_model": "bonsai_q1_0",
    }.get(provider_key, "bonsai_q1_0")


def provider_for_model_name(model_name: str) -> str | None:
    normalized = (model_name or "").strip().lower()
    if not normalized:
        return None
    model_to_provider = {
        provider_default_model("groq").lower(): "groq",
        provider_default_model("gemini").lower(): "gemini",
        provider_default_model("vibe").lower(): "vibe",
        provider_default_model("codex").lower(): "codex",
        provider_default_model("local_model").lower(): "local_model",
        "gemini-2.5-pro": "gemini",
        "gemini-2.5-flash": "gemini",
        "codestral": "vibe",
        "ministral": "vibe",
        "gpt-5.4-mini": "codex",
        "gpt-5.5": "codex",
        "bonsai_q1_0": "local_model",
        "bonsai-q1-0": "local_model",
    }
    if normalized in model_to_provider:
        return model_to_provider[normalized]
    if normalized.startswith("gemini"):
        return "gemini"
    if normalized.startswith("llama-3.3"):
        return "groq"
    if normalized.startswith("codestral") or normalized.startswith("ministral"):
        return "vibe"
    if normalized.startswith("gpt-5"):
        return "codex"
    if "bonsai" in normalized:
        return "local_model"
    return None


def current_orchestration_intent_summary(intent: dict[str, Any] | None = None) -> str:
    intent = intent or load_orchestration_intent()
    provider_key = normalize_provider_key(str(intent.get("provider_key") or "local_model"))
    model_id = str(intent.get("model_id") or provider_default_model(provider_key))
    takeover_mode = bool(intent.get("takeover_mode"))
    fallback_model = str(intent.get("fallback_model_id") or "bonsai_q1_0")
    if provider_key == "local_model":
        return f"local {model_id} (fallback {fallback_model})"
    takeover = "takeover" if takeover_mode else "orchestration-only"
    return f"{provider_key}::{model_id} ({takeover}; fallback {fallback_model})"


def resolve_orchestration_intent(
    provider_key: str | None = None,
    model_id: str | None = None,
    *,
    takeover_mode: bool | None = None,
    source: str = "operator",
) -> dict[str, Any]:
    provider_key = normalize_provider_key(provider_key or "local_model")
    if provider_key in {"", "unknown"}:
        provider_key = "local_model"
    resolved_model = (model_id or "").strip() or provider_default_model(provider_key)
    fallback_provider_key = "local_model"
    fallback_model_id = provider_default_model(fallback_provider_key)
    provider_kind = {
        "groq": "cloud_provider",
        "gemini": "cloud_provider",
        "vibe": "cloud_orchestrator",
        "codex": "cloud_orchestrator",
        "local_model": "local_runtime",
    }.get(provider_key, "local_runtime")
    takeover = bool(takeover_mode) if takeover_mode is not None else False
    return save_orchestration_intent(
        {
            "actor_id": PERSONA_ID,
            "provider_key": provider_key,
            "provider_kind": provider_kind,
            "workload_type": "orchestration",
            "model_id": resolved_model,
            "model_family": resolved_model.split("_", 1)[0] if "_" in resolved_model else resolved_model.split("-", 1)[0],
            "role": "big_brain_orchestration",
            "takeover_mode": takeover,
            "fallback_provider_key": fallback_provider_key,
            "fallback_model_id": fallback_model_id,
            "source": source,
            "notes": "Explicit orchestration lane intent; resonance/pathing remains in control.",
        }
    )


def parse_orchestration_intent_command(ans: str) -> tuple[dict[str, Any] | None, str]:
    text = ans.strip()
    if not text:
        return None, "empty_command"
    low = text.lower()
    if not any(low.startswith(prefix) for prefix in ("use ", "route ", "model ", "orchestrate ", "set model ", "switch to ")):
        return None, "not_orchestration_command"
    tokens = text.split()
    provider_token = ""
    model_token = ""
    takeover_mode = None
    for tok in tokens[1:]:
        norm = normalize_provider_key(tok)
        if norm in {"groq", "gemini", "vibe", "codex", "local_model"} and not provider_token:
            provider_token = norm
            continue
        if tok.lower() in {"takeover", "bigbrain", "big_brain", "orchestrate", "orchestration"}:
            takeover_mode = tok.lower() != "orchestration"
            continue
        if not model_token and tok.lower() not in {"for", "as", "the", "a", "to", "with", "model", "lane"}:
            model_token = tok
    if not provider_token and "local" in low and "bonsai" in low:
        provider_token = "local_model"
    if not provider_token and model_token:
        inferred = provider_for_model_name(model_token)
        if inferred:
            provider_token = inferred
    if not provider_token:
        inferred = provider_for_model_name(text)
        if inferred:
            provider_token = inferred
    if not provider_token:
        return None, "provider_not_found"
    if not model_token:
        model_token = provider_default_model(provider_token)
    intent = resolve_orchestration_intent(provider_key=provider_token, model_id=model_token, takeover_mode=takeover_mode, source="operator_command")
    return intent, ""


def compose_indy_orchestration_check_message(intent: dict[str, Any]) -> str:
    summary = current_orchestration_intent_summary(intent)
    return (
        "Hello, Indy_READs, it's Northern.Strike, how the fuck are you doing tonight? "
        "Let me know the answer to that question. "
        "Then tell me what model you are running your chat through and confirm that. "
        f"Current orchestration intent: {summary}."
    )


def queue_indy_directive_message(
    message: str,
    *,
    intent: dict[str, Any] | None = None,
    outbox: Path | None = None,
) -> dict[str, Any]:
    outbox = outbox or INDY_DIRECTIVE_OUTBOX
    payload = {
        "schema": "lucidota.indy_reads.indy_directive.v1",
        "queued_at": now(),
        "target_path": "indy_runtime_control_surface",
        "route": "indy_orchestration_directive",
        "actor_id": PERSONA_ID,
        "body": message,
        "body_sha256": sha_text(message),
        "intent": intent or load_orchestration_intent(),
        "delivery_status": "QUEUED_FOR_INDY_RUNTIME",
    }
    outbox.parent.mkdir(parents=True, exist_ok=True)
    with outbox.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    return {
        "ok": True,
        "outbox": str(outbox),
        "delivery_status": payload["delivery_status"],
        "body_sha256": payload["body_sha256"],
    }


def queue_indy_chat_message(
    message: str,
    *,
    sender_id: str = "Northern.Strike",
    room_id: str = "!indy_command_deck:localhost",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or INDY_CONDUIT_RECEIPT_DIR
    event_id = "indy-chat:" + sha_text(
        json.dumps(
            {
                "sender_id": sender_id,
                "room_id": room_id,
                "body": message,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    )[:24]
    raw_text = message
    clean_text = " ".join(message.split())
    extracted_entities = {"urls": [], "emails": [], "slash_commands": [], "hashtags": []}
    if db_available():
        try:
            with psycopg.connect(DB_URL) as conn:  # type: ignore[union-attr]
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO ironclaw.waking_dialogue_stream
                          (comms_channel, sender_id, room_id, event_id, raw_text, clean_text, extracted_entities, processed_status, receipt_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                        ON CONFLICT (comms_channel, event_id) DO UPDATE SET
                          raw_text = EXCLUDED.raw_text,
                          clean_text = EXCLUDED.clean_text,
                          extracted_entities = EXCLUDED.extracted_entities,
                          processed_status = EXCLUDED.processed_status,
                          receipt_id = EXCLUDED.receipt_id,
                          updated_at = now()
                        RETURNING id::text
                        """,
                        (
                            "matrix",
                            sender_id,
                            room_id,
                            event_id,
                            raw_text,
                            clean_text,
                            json.dumps(extracted_entities, sort_keys=True),
                            "queued",
                            "matrix_conduit:" + sha_text(json.dumps({"event_id": event_id, "room_id": room_id, "sender_id": sender_id, "body": raw_text}, sort_keys=True, ensure_ascii=False, default=str))[:16],
                        ),
                    )
                    dialogue_id = cur.fetchone()[0]
                conn.commit()
            receipt = {
                "schema": "lucidota.indy_reads.operator_chat_message.v1",
                "queued_at": now(),
                "dialogue_id": dialogue_id,
                "event_id": event_id,
                "room_id": room_id,
                "sender_id": sender_id,
                "body": raw_text,
                "body_sha256": sha_text(raw_text),
                "delivery_status": "QUEUED_FOR_CHAT_SURFACE",
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            receipt_path = output_dir / f"indy_chat_message_{sha_text(event_id)[:16]}.json"
            receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            return {
                "ok": True,
                "executed": True,
                "receipt_path": str(receipt_path),
                "event_id": event_id,
                "dialogue_row": {
                    "comms_channel": "matrix",
                    "sender_id": sender_id,
                    "room_id": room_id,
                    "event_id": event_id,
                    "raw_text": raw_text,
                    "clean_text": clean_text,
                    "extracted_entities": extracted_entities,
                    "processed_status": "queued",
                    "receipt_id": receipt["dialogue_id"] if isinstance(receipt.get("dialogue_id"), str) else "",
                },
                "absurd_jobs": 0,
                "ui_action": None,
                "error": "",
            }
        except Exception as exc:
            return {
                "ok": False,
                "executed": False,
                "receipt_path": "",
                "event_id": event_id,
                "dialogue_row": {
                    "comms_channel": "matrix",
                    "sender_id": sender_id,
                    "room_id": room_id,
                    "event_id": event_id,
                    "raw_text": raw_text,
                    "clean_text": clean_text,
                    "extracted_entities": extracted_entities,
                    "processed_status": "queued",
                    "receipt_id": "",
                },
                "absurd_jobs": 0,
                "ui_action": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {
        "ok": False,
        "executed": False,
        "receipt_path": "",
        "event_id": event_id,
        "dialogue_row": {
            "comms_channel": "matrix",
            "sender_id": sender_id,
            "room_id": room_id,
            "event_id": event_id,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "extracted_entities": extracted_entities,
            "processed_status": "queued",
            "receipt_id": "",
        },
        "absurd_jobs": 0,
        "ui_action": None,
        "error": "database_unavailable",
    }


def compact_surface_fields(payload: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    return {field: payload.get(field) for field in fields if field in payload}


def compact_list(values: Any, limit: int = 8) -> Any:
    if isinstance(values, list):
        return values[:limit]
    return values


def indy_boot_context_snapshot() -> dict[str, Any]:
    active_operation_mode = fetch_one_json(
        "SELECT COALESCE((SELECT to_jsonb(aom) FROM lucidota_control.active_operation_mode aom LIMIT 1), '{}'::jsonb)"
    )
    manual_current = fetch_one_json(
        "SELECT COALESCE((SELECT to_jsonb(mc) FROM lucidota_canon.manual_current mc LIMIT 1), '{}'::jsonb)"
    )
    root_orchestrator_current = fetch_one_json(
        "SELECT COALESCE((SELECT to_jsonb(roc) FROM lucidota_canon.root_orchestrator_current roc LIMIT 1), '{}'::jsonb)"
    )
    workload_audit_current = fetch_one_json(
        "SELECT COALESCE((SELECT to_jsonb(wac) FROM lucidota_canon.workload_audit_current wac LIMIT 1), '{}'::jsonb)"
    )
    workload_audit_telemetry_current = fetch_one_json(
        "SELECT COALESCE((SELECT to_jsonb(wat) FROM lucidota_canon.workload_audit_telemetry_current wat LIMIT 1), '{}'::jsonb)"
    )
    model_registry_current = fetch_one_json(
        "SELECT COALESCE((SELECT to_jsonb(mrc) FROM lucidota_canon.model_registry_current mrc LIMIT 1), '{}'::jsonb)"
    )
    provider_current = fetch_one_json(
        "SELECT COALESCE((SELECT to_jsonb(pc) FROM lucidota_canon.provider_current pc LIMIT 1), '{}'::jsonb)"
    )
    orchestration_intent = load_orchestration_intent()
    orchestration_current = fetch_one_json(
        "SELECT COALESCE((SELECT to_jsonb(oc) FROM lucidota_canon.indy_reads_orchestration_current oc LIMIT 1), '{}'::jsonb)"
    )
    command_registry = fetch_one_json(
        "SELECT COALESCE((SELECT to_jsonb(cr) FROM lucidota_canon.command_registry cr LIMIT 1), '{}'::jsonb)"
    )
    capability_current = fetch_one_json(
        "SELECT COALESCE((SELECT to_jsonb(cc) FROM lucidota_canon.capability_current cc LIMIT 1), '{}'::jsonb)"
    )
    surface_registry = fetch_rows_json(
        """
        SELECT to_jsonb(sr)
        FROM (
            SELECT surface_id, canonical_owner, surface_kind, active, approval_required
            FROM lucidota_control.surface_registry
            ORDER BY surface_id
            LIMIT 8
        ) sr
        """
    )
    schema_owner_manifest = fetch_rows_json(
        """
        SELECT to_jsonb(som)
        FROM (
            SELECT surface_id, canonical_owner, surface_kind, active, approval_required, approved_by, approval_receipt_uuid
            FROM lucidota_control.schema_owner_manifest
            ORDER BY surface_id
            LIMIT 8
        ) som
        """
    )

    return {
        "active_operation_mode": compact_surface_fields(
            active_operation_mode,
            [
                "current_mode",
                "cloud_policy",
                "swarm_policy",
                "indy_reads_policy",
                "receipt_policy",
                "runtime_default_policy",
                "build_session_policy",
                "operator_override",
                "updated_at",
            ],
        ),
        "manual_current": {
            "route_refs": compact_list(manual_current.get("route_refs", []), 12),
            "next_command_refs": compact_list(manual_current.get("next_command_refs", []), 16),
            "orchestration": manual_current.get("orchestration", {}),
            "db_law": manual_current.get("db_law", {}),
            "work_order_flow": manual_current.get("work_order_flow", {}),
        },
        "root_orchestrator_current": {
            "route_count": root_orchestrator_current.get("route_count"),
            "next_command_refs": compact_list(root_orchestrator_current.get("next_command_refs", []), 16),
            "orchestration": root_orchestrator_current.get("orchestration", {}),
            "db_law": root_orchestrator_current.get("db_law", {}),
            "live_surface_keys": sorted(root_orchestrator_current.get("live_surface", {}).keys())[:16],
        },
        "workload_audit_current": compact_surface_fields(
            workload_audit_current,
            [
                "audit_status",
                "has_unacknowledged_unknown_rows",
                "unknown_row_count",
                "proven_row_count",
                "partial_row_count",
                "contradicted_row_count",
                "can_claim_duplex_race",
                "ledger_row_count",
            ],
        ),
        "workload_audit_telemetry_current": compact_surface_fields(
            workload_audit_telemetry_current,
            [
                "ledger_row_count",
                "receipt_row_count",
                "proven_row_count",
                "partial_row_count",
                "unknown_row_count",
                "contradicted_row_count",
                "codex_row_count",
                "indy_row_count",
                "local_llm_row_count",
                "groq_row_count",
                "gemini_row_count",
                "gemini_paid_row_count",
                "vibe_row_count",
                "tokens_in_total",
                "tokens_out_total",
            ],
        ),
        "model_registry_current": compact_surface_fields(
            model_registry_current,
            ["model_packet_id", "model_id", "role", "slot_name", "loadout_id", "expected_vram_mb", "notes"],
        ),
        "provider_current": compact_surface_fields(
            provider_current,
            ["provider_packet_id", "provider_id", "provider_key", "current_status", "notes"],
        ),
        "indy_reads_orchestration_current": compact_surface_fields(
            orchestration_current,
            ["state_key", "provider_key", "provider_kind", "model_id", "takeover_mode", "summary", "proof_status"],
        ),
        "orchestration_intent": orchestration_intent,
        "command_registry": compact_surface_fields(command_registry, ["command_registry_id", "summary", "next_commands"]),
        "capability_current": compact_surface_fields(
            {
                **capability_current,
                "active_capabilities": compact_list(capability_current.get("active_capabilities", []), 4),
                "next_command_refs": compact_list(capability_current.get("next_command_refs", []), 12),
            },
            ["capability_packet_id", "next_command_refs", "active_capabilities"],
        ),
        "surface_registry": compact_list(surface_registry, 4),
        "schema_owner_manifest": compact_list(schema_owner_manifest, 4),
    }


def boot_prompt_payload(boot_packet: dict[str, Any], context: dict[str, Any], slot_role: str) -> str:
    return json.dumps(
        {
            "boot_packet": boot_packet,
            "live_context": context,
            "slot_role": slot_role,
            "instructions": [
                "Return compact JSON only.",
                "Do not claim any receipt-backed work without receipts.",
                "You are Indy_READs, not Codex.",
                "Use proof_status exactly as PROVEN, PARTIAL, UNKNOWN, or CONTRADICTED.",
                "If you lack evidence, mark UNKNOWN debt instead of inventing facts.",
            ],
            "output_schema": {
                "self_model": "object",
                "llmwiki_entry": "object",
                "hunch_log": "object",
                "system_map": "object",
                "mistake_ledger": "object",
                "learning_queue": "object",
                "metacognition_current": "object",
            },
        },
        indent=2,
        sort_keys=True,
        default=str,
    )


def run_boot_slot(*, lane: str, slot_role: str, boot_packet: dict[str, Any], context: dict[str, Any], max_tokens: int = 512, timeout_sec: float = 180.0) -> dict[str, Any]:
    prompt = boot_prompt_payload(boot_packet, context, slot_role)
    system = (
        "You are Indy_READs, the IronClaw local-model exocortex runtime. "
        f"You are boot slot {slot_role}. "
        "The database is truth; the boot packet and live DB surfaces are the only authority; "
        "LLMWIKI is your notebook, not canon truth."
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "local_model_chat_cli.py"),
            "--lane",
            lane,
            "--prompt",
            prompt,
            "--system",
            system,
            "--max-tokens",
            str(max_tokens),
            "--temperature",
            "0.0",
            "--timeout-sec",
            str(timeout_sec),
            "--execute",
            "--json",
            "--clear-history",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    try:
        receipt = json.loads(proc.stdout.strip() or "{}")
    except json.JSONDecodeError:
        receipt = {
            "schema": "lucidota.indy_reads.bootstrap_slot_receipt.v1",
            "status": "BLOCKED",
            "text": (proc.stdout or proc.stderr or "")[-4000:],
        }
    receipt["returncode"] = proc.returncode
    receipt["stderr"] = (proc.stderr or "")[-4000:]
    receipt["slot_role"] = slot_role
    receipt["lane"] = lane
    return receipt


def parse_json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        match = re.search(r"\{.*\}", value, re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


def default_indy_boot_content(*, slot0_text: str, slot1_text: str, boot_packet: dict[str, Any], context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    live_mode = context.get("active_operation_mode", {})
    manual_refs = context.get("manual_current", {}).get("route_refs", [])
    root_refs = context.get("root_orchestrator_current", {}).get("next_command_refs", [])
    workload_summary = context.get("workload_audit_current", {})
    evidence_refs = [str(INDY_BOOT_PACKET_PATH), "04_RUNTIME/INDY_READS/indy_reads_service_manifest.json"]
    db_refs = [
        "lucidota_control.active_operation_mode",
        "lucidota_canon.manual_current",
        "lucidota_canon.root_orchestrator_current",
        "lucidota_canon.workload_audit_current",
        "lucidota_canon.workload_audit_telemetry_current",
        "lucidota_canon.indy_reads_self_model",
        "lucidota_canon.indy_reads_llmwiki_entry",
        "lucidota_canon.indy_reads_hunch_log",
        "lucidota_canon.indy_reads_learning_queue",
        "lucidota_canon.indy_reads_system_map",
        "lucidota_canon.indy_reads_mistake_ledger",
        "lucidota_canon.indy_reads_research_source",
        "lucidota_canon.indy_reads_metacognition_current",
    ]
    shared_summary = " ".join(part for part in [slot0_text.strip(), slot1_text.strip()] if part).strip()
    if not shared_summary:
        shared_summary = "Indy_READs booted through IronClaw local models and must learn from live DB truth before claiming anything."
    orchestration_intent = load_orchestration_intent()
    orchestration_summary = current_orchestration_intent_summary(orchestration_intent)
    self_model = {
        "actor_id": "indy_reads_runtime",
        "author": "indy_reads",
        "role": "indy_reads_runtime",
        "boundaries": "DB truth first; receipts or UNKNOWN debt; LLMWIKI is notebook, not canon; do not impersonate Codex.",
        "voice": "evidence-oriented, terse, skeptical, curious, operator-aware",
        "relationship_to_operator": "reports compactly, challenges missing proof, follows operator intent through live DB truth",
        "relationship_to_LUCIDOTA": "runtime exocortex lane with DB-backed memory and receipts",
        "relationship_to_northern_strike": "not yet defined; track as UNKNOWN until evidence appears",
        "relationship_to_Krampus": "evidence archive and custody lane; preserve bytes and receipts",
        "relationship_to_Santa": "not yet defined; track as UNKNOWN until evidence appears",
        "investigation_style": "receipt-bound, contradiction-seeking, live-surface first, cheap-before-expensive",
        "learning_style": "local-first, slot split synthesis plus skeptic review, then promote by receipt",
        "preferred_tools": "Postgres/PostgREST, local Bonsai 8B, Needles, Mamba watcher, TRM loops, River, Treelite, DB graphs",
        "evidence_standard": "live DB rows, receipts, route refs, and model/provider invocation artifacts",
        "receipt_standard": "proven or partial only when backed by row/receipt evidence; otherwise UNKNOWN debt",
        "mistake_handling": "write the correction, preserve the proof trail, and downgrade the claim until receipts catch up",
        "curiosity_targets": "system maps, receipts, workload accounting, model lanes, ontology packets, work orders",
        "current_limitations": "first boot, limited proven history, must keep reading live surfaces before claiming state",
        "next_upgrade": "stabilize the boot path, learn from live surfaces, and promote useful wiki notes through DB gates; explicit orchestration intent defaults to local Bonsai unless the operator says to use cloud",
        "summary": shared_summary[:2000],
        "goals_refs": ["GOALS/CURRENT_HANDOFF.md", "GOALS/GOAL_LOG.md"],
        "confidence": 0.84,
        "evidence_refs": evidence_refs,
        "db_refs": db_refs,
        "proof_status": "PARTIAL" if workload_summary.get("unknown_row_count", 1) else "PROVEN",
        "functionality_explanation": "Indy_READs runtime self-model surface; DB truth first, wiki second, prose third.",
        "ontology_index": {
            "primitive_refs": ["state", "duplex", "allocation"],
            "claim_type": "self_model_boot",
            "evidence_type": "boot_packet_and_slot_receipts",
            "actor_role": "indy_reads_runtime",
            "subsystem_refs": ["self_model", "llmwiki", "hunch_log", "system_map", "mistake_ledger", "learning_queue", "research_source"],
            "risk_tier": "T3",
            "proof_status": "PARTIAL" if workload_summary.get("unknown_row_count", 1) else "PROVEN",
            "receipt_refs": ["indy_reads_exocortex_activation_gate"],
            "next_route": ["indy_reads_llmwiki_entry", "indy_reads_hunch_log", "indy_reads_learning_queue", "indy_reads_system_map", "indy_reads_mistake_ledger", "workload_audit_current"],
        },
    }
    llmwiki_entry = {
        "actor_id": "indy_reads_runtime",
        "author": "indy_reads",
        "topic": "What Indy_READs is and what LUCIDOTA is",
        "summary": shared_summary[:1000],
        "body": shared_summary,
        "confidence": 0.79,
        "evidence_refs": evidence_refs,
        "db_refs": db_refs,
        "next_questions": [
            "What live rows prove the current model/workload split?",
            "Which claims still need UNKNOWN debt rows?",
            "Which model lanes are actually resident versus merely planned?",
        ],
        "mistake_risk": "high_if_claiming_unproven_boot_state",
        "promotion_candidate": False,
        "proof_status": "PARTIAL",
        "functionality_explanation": "Indy_READs metacognition notebook entry surface; not canon truth until promoted with receipts.",
        "ontology_index": {
            "primitive_refs": ["state", "duplex", "allocation"],
            "claim_type": "llmwiki_boot_note",
            "evidence_type": "boot_packet_and_slot_receipts",
            "actor_role": "indy_reads_runtime",
            "subsystem_refs": ["self_model", "llmwiki", "hunch_log", "system_map", "mistake_ledger", "learning_queue"],
            "risk_tier": "T3",
            "proof_status": "PARTIAL",
            "receipt_refs": ["indy_reads_exocortex_activation_gate"],
            "next_route": ["indy_reads_metacognition_current", "workload_audit_current"],
        },
    }
    hunch_log = {
        "actor_id": "indy_reads_runtime",
        "topic": "Boot lane and live truth surfaces",
        "hunch": "The Bonsai 8B shared-weight dual-slot lane should be the primary exocortex bootstrap cortex, with Needles as reflexes and Mamba as watcher.",
        "confidence": 0.72,
        "evidence_refs": evidence_refs,
        "db_refs": db_refs,
        "next_questions": [
            "Can the live workload ledger prove the Indy lane after boot?",
            "Do manual and root packets expose the new Indy refs compactly?",
        ],
        "proof_status": "PARTIAL",
        "functionality_explanation": "Indy_READs hunch log; useful for learning and contradiction tracking, not canon truth.",
        "ontology_index": {
            "primitive_refs": ["state", "duplex", "allocation"],
            "claim_type": "hunch_boot",
            "evidence_type": "boot_packet_and_slot_receipts",
            "actor_role": "indy_reads_runtime",
            "subsystem_refs": ["hunch_log", "workload_audit_current", "manual_current", "root_orchestrator_current"],
            "risk_tier": "T3",
            "proof_status": "PARTIAL",
            "receipt_refs": ["indy_reads_exocortex_activation_gate"],
            "next_route": ["indy_reads_learning_queue", "workload_audit_current"],
        },
    }
    system_map = {
        "actor_id": "indy_reads_runtime",
        "topic": "LUCIDOTA runtime topology",
        "summary": "IronClaw body; Postgres law; PostgREST sensory/manual API; Bonsai dual-slot cortex; Needles reflexes; Mamba watcher; TRM constraints; RWKV continuity; receipts or debt.",
        "subsystem_refs": [
            "lucidota_control.active_operation_mode",
            "lucidota_canon.manual_current",
            "lucidota_canon.root_orchestrator_current",
            "lucidota_canon.workload_audit_current",
            "lucidota_canon.indy_reads_self_model",
            "lucidota_canon.indy_reads_llmwiki_entry",
        ],
        "evidence_refs": evidence_refs,
        "db_refs": db_refs,
        "proof_status": "PARTIAL",
        "functionality_explanation": "Indy_READs system map; a compact topology note rather than canon truth.",
        "ontology_index": {
            "primitive_refs": ["state", "duplex", "allocation"],
            "claim_type": "system_map_boot",
            "evidence_type": "boot_packet_and_slot_receipts",
            "actor_role": "indy_reads_runtime",
            "subsystem_refs": ["manual_current", "root_orchestrator_current", "workload_audit_current", "active_operation_mode"],
            "risk_tier": "T3",
            "proof_status": "PARTIAL",
            "receipt_refs": ["indy_reads_exocortex_activation_gate"],
            "next_route": ["workload_audit_current", "active_operation_mode"],
        },
    }
    mistake_ledger = {
        "actor_id": "indy_reads_runtime",
        "mistake_summary": "First boot still needs DB proof before any claim of being online.",
        "mistake_risk": "high_if_claiming_boot_without_workload_rows",
        "correction": "Treat the boot as partial until the workload ledger and the Indy rows are visible in PostgREST.",
        "evidence_refs": evidence_refs,
        "db_refs": db_refs,
        "proof_status": "PARTIAL",
        "functionality_explanation": "Indy_READs mistake ledger; records misses, corrections, and proof debt.",
        "ontology_index": {
            "primitive_refs": ["state", "duplex", "allocation"],
            "claim_type": "mistake_boot",
            "evidence_type": "boot_packet_and_slot_receipts",
            "actor_role": "indy_reads_runtime",
            "subsystem_refs": ["mistake_ledger", "workload_audit_current"],
            "risk_tier": "T3",
            "proof_status": "PARTIAL",
            "receipt_refs": ["indy_reads_exocortex_activation_gate"],
            "next_route": ["workload_audit_current", "unproven_work_debt"],
        },
    }
    learning_queue = {
        "actor_id": "indy_reads_runtime",
        "topic": "Next learning targets for Indy_READs",
        "summary": "Learn the live DB surfaces, prove the workload ledger, and then refine the self-model/wiki via receipt-backed updates.",
        "status": "queued",
        "priority": 10,
        "next_route": "workload_audit_current",
        "evidence_refs": evidence_refs,
        "db_refs": db_refs,
        "proof_status": "PARTIAL",
        "functionality_explanation": "Indy_READs learning queue; tracks what she should learn next and how to route the next investigation.",
        "ontology_index": {
            "primitive_refs": ["state", "duplex", "allocation"],
            "claim_type": "learning_queue_boot",
            "evidence_type": "boot_packet_and_slot_receipts",
            "actor_role": "indy_reads_runtime",
            "subsystem_refs": ["learning_queue", "workload_audit_current", "active_operation_mode"],
            "risk_tier": "T3",
            "proof_status": "PARTIAL",
            "receipt_refs": ["indy_reads_exocortex_activation_gate"],
            "next_route": ["workload_audit_current", "manual_current", "root_orchestrator_current"],
        },
    }
    metacognition_current = {
        "state_key": "indy_reads_metacognition_current",
        "actor_id": "indy_reads_runtime",
        "owner_role": "indy_reads_runtime",
        "what_i_am": "Indy_READs is the IronClaw local-model exocortex lane with DB-backed memory, metacognition, and receipts.",
        "what_i_am_for": "Learn the system, challenge lies, write self-model/wiki/hunch/system-map notes, and keep proof before claims.",
        "operator_model": f"The operator wants live truth over narrative, compact refs over bloat, and receipts or UNKNOWN debt. Current orchestration intent: {orchestration_summary}.",
        "case_model": "Cases are chronology-plus-evidence problems; read the DB and receipts before speaking.",
        "system_model": f"LUCIDOTA is Postgres/PostgREST truth plus local models, workflows, receipts, and compact operator surfaces. Default runtime lane: {orchestration_summary}.",
        "learning_next": "Prove the workload ledger, then promote useful wiki notes through receipt gates; treat explicit provider/model switches as orchestration intent, not takeover.",
        "refusal_standard": "Refuse any claim without a receipt row or explicit UNKNOWN debt.",
        "self_model_ref": "",
        "llmwiki_ref": "",
        "hunch_log_ref": "",
        "system_map_ref": "",
        "mistake_ledger_ref": "",
        "learning_queue_ref": "",
        "research_source_ref": "",
        "boot_packet_ref": str(INDY_BOOT_PACKET_PATH),
        "evidence_refs": evidence_refs,
        "db_refs": db_refs,
        "proof_status": "PARTIAL",
        "functionality_explanation": "Indy_READs metacognition current packet; the current self-understanding surface for the runtime lane.",
        "ontology_index": {
            "primitive_refs": ["state", "duplex", "allocation"],
            "claim_type": "metacognition_boot",
            "evidence_type": "boot_packet_and_slot_receipts",
            "actor_role": "indy_reads_runtime",
            "subsystem_refs": ["self_model", "llmwiki", "hunch_log", "system_map", "mistake_ledger", "learning_queue", "research_source"],
            "risk_tier": "T3",
            "proof_status": "PARTIAL",
            "receipt_refs": ["indy_reads_exocortex_activation_gate"],
            "next_route": ["indy_reads_self_model", "indy_reads_llmwiki_entry", "workload_audit_current"],
        },
    }
    return {
        "self_model": self_model,
        "llmwiki_entry": llmwiki_entry,
        "hunch_log": hunch_log,
        "system_map": system_map,
        "mistake_ledger": mistake_ledger,
        "learning_queue": learning_queue,
        "metacognition_current": metacognition_current,
    }


def hardware_telemetry() -> dict[str, Any]:
    telemetry: dict[str, Any] = {
        "cpu_count": os.cpu_count() or 1,
        "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else [],
        "rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        telemetry.update({
            "memory_total_bytes": int(vm.total),
            "memory_available_bytes": int(vm.available),
            "memory_percent": float(vm.percent),
        })
    except Exception:
        pass
    return telemetry


def transport_socket_active(path: Path = TRANSPORT_SOCKET) -> bool:
    if not path.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.1)
            return sock.connect_ex(str(path)) == 0
    except OSError:
        return False


def load_attention_model() -> Any | None:
    if not RIVER_MODEL_PATH.exists():
        try:
            from river import compose, linear_model, preprocessing  # type: ignore
        except Exception:
            return None
        return compose.Pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression())
    try:
        return pickle.loads(RIVER_MODEL_PATH.read_bytes())
    except Exception:
        return None


def save_attention_model(model: Any) -> None:
    try:
        RIVER_MODEL_PATH.write_bytes(pickle.dumps(model))
    except Exception:
        pass


def attention_features(book: Book | None, page: dict[str, Any] | None, parser: dict[str, Any] | None, socket_active: bool, score: int | None = None) -> dict[str, Any]:
    page_chars = int(page.get("chars", 0)) if page else 0
    parser_bps = int((parser or {}).get("confidence_bps", 0))
    completed_pages = 0
    if page:
        try:
            completed_pages = len(page.get("completed_pages", []))
        except Exception:
            completed_pages = 0
    return {
        "page_chars": page_chars,
        "parser_bps": parser_bps,
        "completed_pages": completed_pages,
        "socket_active": int(socket_active),
        "score": int(score or 0),
        "book_size_bytes": int(getattr(book, "size_bytes", 0) or 0),
    }


def tune_ingestion_batch_size(st: dict[str, Any], book: Book | None, page: dict[str, Any] | None, parser: dict[str, Any] | None, socket_active: bool, score: int | None = None) -> dict[str, Any]:
    model = load_attention_model()
    features = attention_features(book, page, parser, socket_active, score=score)
    proba = 0.5
    if model is not None:
        try:
            probs = model.predict_proba_one(features)
            proba = float(probs.get(True, probs.get(1, 0.5)))
        except Exception:
            proba = 0.5
        try:
            label = bool((score or 0) >= 80)
            model.learn_one(features, label)
            save_attention_model(model)
        except Exception:
            pass
    batch_size = max(1, min(32, int(round(2 + proba * 14 + min(features["page_chars"] / 1200.0, 6.0)))))
    slow_lane = st.setdefault("slow_lane", {})
    slow_lane["ingestion_batch_size"] = batch_size
    slow_lane["river_probability"] = proba
    slow_lane["last_feature_vector"] = features
    slow_lane["transport_socket_active"] = socket_active
    slow_lane["updated_at"] = now()
    return {"batch_size": batch_size, "river_probability": proba, "features": features}


def record_daemon_heartbeat(*, daemon_name: str, socket_active: bool, terminal_active: bool, batch_size: int | None, book: Book | None = None, page: dict[str, Any] | None = None, parser: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> None:
    if not db_available():
        return
    telemetry = hardware_telemetry()
    if page:
        telemetry["page"] = {"book_id": page.get("book_id"), "page": page.get("page"), "page_hash": page.get("page_hash")}
    if parser:
        telemetry["parser"] = {"parser_version": parser.get("parser_version"), "confidence_bps": parser.get("confidence_bps"), "terms": parser.get("terms", [])}
    if book:
        telemetry["book"] = {"book_id": book.id, "book_name": book.name}
    if extra:
        telemetry["extra"] = extra
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ironclaw.daemon_heartbeats
                      (daemon_name, host_name, process_id, transport_socket, socket_active, terminal_active, batch_size, river_state, telemetry, detail)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb)
                    """,
                    (
                        daemon_name,
                        socket.gethostname(),
                        os.getpid(),
                        str(TRANSPORT_SOCKET),
                        socket_active,
                        terminal_active,
                        batch_size,
                        json.dumps({"book_id": getattr(book, "id", ""), "page": page.get("page") if page else None, "score": extra.get("score") if extra else None}),
                        json.dumps(telemetry, default=str),
                        json.dumps({"source": "scripts/indy_reads.py", "attention": "collaborative" if socket_active else "autonomous", **(extra or {})}, default=str),
                    ),
                )
            conn.commit()
    except Exception:
        return


def record_indy_judgment(*, book: Book, page: dict[str, Any], parser: dict[str, Any], decision: str, score: int, score_label_value: str, notes: str, repair_instruction: str, term_correction: str, favorite_line: str, confusion: str, socket_active: bool, terminal_active: bool, batch_size: int | None, extra: dict[str, Any] | None = None) -> None:
    if not db_available():
        return
    telemetry = hardware_telemetry()
    telemetry.update({
        "attention_state": "collaborative" if socket_active else "autonomous",
        "batch_size": batch_size,
        "cpu_count": telemetry.get("cpu_count"),
    })
    source_payload = {
        "page": page,
        "parser": parser,
        "extra": extra or {},
    }
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ironclaw.indy_read_judgments
                      (daemon_name, book_id, book_name, page_number, page_hash, parser_version, decision, score, score_label, term_correction, notes, repair_instruction, favorite_line, confusion, transport_socket, socket_active, terminal_active, batch_size, telemetry, source_payload)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
                    """,
                    (
                        DAEMON_NAME,
                        book.id,
                        book.name,
                        int(page["page"]),
                        page["page_hash"],
                        parser["parser_version"],
                        decision,
                        int(score),
                        score_label_value,
                        term_correction,
                        notes,
                        repair_instruction,
                        favorite_line,
                        confusion,
                        str(TRANSPORT_SOCKET),
                        socket_active,
                        terminal_active,
                        batch_size,
                        json.dumps(telemetry, default=str),
                        json.dumps(source_payload, default=str),
                    ),
                )
            conn.commit()
    except Exception:
        return


def tune_and_record_heartbeat(st: dict[str, Any], book: Book | None, page: dict[str, Any] | None, parser: dict[str, Any] | None, *, score: int | None = None, terminal_active: bool = False) -> dict[str, Any]:
    socket_active = transport_socket_active()
    tune = tune_ingestion_batch_size(st, book, page, parser, socket_active, score=score)
    record_daemon_heartbeat(
        daemon_name=DAEMON_NAME,
        socket_active=socket_active,
        terminal_active=terminal_active,
        batch_size=tune["batch_size"],
        book=book,
        page=page,
        parser=parser,
        extra={"score": score, "river_probability": tune["river_probability"]},
    )
    save_state(st)
    return {"socket_active": socket_active, **tune}


def timed_input(prompt: str, timeout_seconds: float | None = None) -> str | object:
    if timeout_seconds is None or not sys.stdin.isatty():
        return input(prompt)
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    if ready:
        return sys.stdin.readline().rstrip("\n")
    return TIMEOUT_SENTINEL


TIMEOUT_SENTINEL = object()


def run_autonomous_slow_lane_tick(st: dict[str, Any], book: Book | None, page: dict[str, Any] | None, parser: dict[str, Any] | None) -> dict[str, Any]:
    tick = tune_and_record_heartbeat(st, book, page, parser, score=(parser or {}).get("confidence_bps", 0), terminal_active=False)
    slow_lane = st.setdefault("slow_lane", {})
    slow_lane["last_autonomous_tick_at"] = now()
    slow_lane["last_autonomous_reason"] = "terminal_timeout"
    save_state(st)
    return tick


def standard_flow(book: Book, st: dict[str, Any]) -> None:
    bs = get_book_state(st, book)
    while True:
        page_n = int(bs.get("current_page", 1))
        try:
            page = extract_page(book, page_n)
            parser = fast_parse(page)
            # preload next page in background-ish foreground quick cache, ignore failures
            try:
                next_page = extract_page(book, page_n + 1)
                fast_parse(next_page)
            except Exception as exc:
                _ = exc  # next-page cache miss is non-fatal
        except Exception as e:
            banner("EXTRACTION FAILURE")
            print(e); pause(); return

        banner(f"{book.name[:70]} — PAGE {page_n}")
        print(f"extract={page['extract_method']} | chars={page['chars']} | hash={page['page_hash'][:12]} | parser={PARSER_VERSION}")
        print("\nPAGE TEXT")
        print("─" * 96)
        wrap_print(page["text"], max_lines=30)
        print("─" * 96)
        print("\nINDY MARGIN NOTES")
        for note in parser["notes"]:
            print(f"▸ {note}")
        print("\nGO TERMS:", " ".join(f"#{p}" for p in parser.get("terms", parser.get("primitives", []))))
        print(f"BPS: {parser['confidence_bps']} | lifecycle: {parser['claim_lifecycle']}")
        attention = "collaborative" if transport_socket_active() else "autonomous"
        print(f"Attention: {attention} | transport socket: {TRANSPORT_SOCKET}")
        print("\nOptions: [j]udge  [p]terms  [a]dapters  [s]kip/comment  [q]uit to library")
        ans_raw = timed_input("move> ", ATTENTION_TIMEOUT_SECONDS)
        if ans_raw is TIMEOUT_SENTINEL:
            socket_active = transport_socket_active()
            if socket_active:
                tune_and_record_heartbeat(st, book, page, parser, score=parser.get("confidence_bps", 0), terminal_active=False)
                print("\nCollaborative companion mode: local transport socket is active. Waiting for operator input.")
            else:
                run_autonomous_slow_lane_tick(st, book, page, parser)
                print("\nAutonomous slot claimed: tuned batch size and wrote daemon heartbeat.")
            continue
        ans = str(ans_raw).strip().lower()
        if ans == "p":
            term_browser(); continue
        if ans == "a":
            adapter_browser(); continue
        if ans == "q":
            return
        if ans == "s":
            decision, score = "comment", 50
        else:
            judgment = judgment_prompt(st=st, book=book, page=page, parser=parser)
            if judgment is None:
                continue
            decision, score = judgment
        notes = input("Your notes / correction / piss judgment: ").strip()
        repair = input("Repair instruction (optional): ").strip() if decision in {"needs_repair", "rejected"} else ""
        term_correction = input("Term correction (#TERMS or blank): ").strip()
        favorite_line = input("Favorite/important line (optional): ").strip()
        confusion = input("Confusion / carry-forward question (optional): ").strip()
        row = {
            "timestamp": now(),
            "book_id": book.id,
            "book_name": book.name,
            "page": page_n,
            "page_hash": page["page_hash"],
            "extract_method": page["extract_method"],
            "parser_version": PARSER_VERSION,
            "packet_id": parser["packet_id"],
            "parser_terms": "|".join(parser.get("terms", parser.get("primitives", []))),
            "parser_bps": parser["confidence_bps"],
            "decision": decision,
            "score": score,
            "score_label": score_label(score),
            "term_correction": term_correction,
            "notes": notes,
            "repair_instruction": repair,
            "favorite_line": favorite_line,
            "confusion": confusion,
            "raw_csv_json": json.dumps({"page": page, "parser": parser}, sort_keys=True),
        }
        append_csv(row)
        record_indy_judgment(
            book=book,
            page=page,
            parser=parser,
            decision=decision,
            score=score,
            score_label_value=score_label(score),
            notes=notes,
            repair_instruction=repair,
            term_correction=term_correction,
            favorite_line=favorite_line,
            confusion=confusion,
            socket_active=transport_socket_active(),
            terminal_active=True,
            batch_size=int(st.get("slow_lane", {}).get("ingestion_batch_size", 0) or 0) or None,
            extra={"csv_row": row},
        )
        tune_and_record_heartbeat(st, book, page, parser, score=score, terminal_active=True)
        bs.setdefault("completed_pages", []).append(page_n)
        bs["current_page"] = page_n + 1
        bs["last_judgment"] = row
        save_state(st)
        print(f"\nSaved to CSV: {CSV_PATH}")
        print(f"Round result: {score_label(score)} ({score}) — page {page_n} locked. Advancing to page {page_n+1}.")
        pause()


def judgment_prompt(*, st: dict[str, Any], book: Book, page: dict[str, Any], parser: dict[str, Any]) -> tuple[str, int] | None:
    print("\nDecision:")
    opts = ["approved", "needs_repair", "rejected", "comment"]
    for i, o in enumerate(opts, 1): print(f" {i}. {o}")
    while True:
        ans_raw = timed_input("decision> ", ATTENTION_TIMEOUT_SECONDS)
        if ans_raw is TIMEOUT_SENTINEL:
            socket_active = transport_socket_active()
            if socket_active:
                tune_and_record_heartbeat(st, book, page, parser, score=parser.get("confidence_bps", 0), terminal_active=False)
                print("\nCollaborative companion mode detected at decision prompt; waiting for operator.")
            else:
                run_autonomous_slow_lane_tick(st, book, page, parser)
                print("\nDecision prompt timed out; autonomous slot claimed.")
            return None
        ans = str(ans_raw).strip().lower()
        if ans.isdigit() and 1 <= int(ans) <= len(opts):
            decision = opts[int(ans)-1]; break
        if ans in opts:
            decision = ans; break
    while True:
        score_raw = timed_input("score 0-100> ", ATTENTION_TIMEOUT_SECONDS)
        if score_raw is TIMEOUT_SENTINEL:
            socket_active = transport_socket_active()
            if socket_active:
                tune_and_record_heartbeat(st, book, page, parser, score=parser.get("confidence_bps", 0), terminal_active=False)
                print("\nCollaborative companion mode detected at score prompt; waiting for operator.")
            else:
                run_autonomous_slow_lane_tick(st, book, page, parser)
                print("\nScore prompt timed out; autonomous slot claimed.")
            return None
        try:
            score = max(0, min(100, int(str(score_raw).strip())))
            return decision, score
        except ValueError:
            print("number please")


def load_goals_handoff_text() -> str:
    if GOALS_HANDOFF_MD.exists():
        try:
            return GOALS_HANDOFF_MD.read_text(encoding="utf-8")
        except OSError:
            return ""
    return ""


def load_queued_conduit_dialogue(limit: int = 5) -> list[dict[str, Any]]:
    """Read queued Matrix/Conduit rows for the Indy_READs operator chat surface."""
    if not db_available():
        return []
    try:
        indy_conduit_driver = import_indy_conduit_driver()
        read_queued_dialogue_rows = indy_conduit_driver.read_queued_dialogue_rows
        with psycopg.connect(DB_URL) as conn:  # type: ignore[union-attr]
            return read_queued_dialogue_rows(conn, limit=limit)
    except Exception:
        return []


def load_queued_indy_directives(limit: int = 5) -> list[dict[str, Any]]:
    if not INDY_DIRECTIVE_OUTBOX.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with INDY_DIRECTIVE_OUTBOX.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except OSError:
        return []
    return rows[-limit:]


def format_conduit_dialogue_row(row: dict[str, Any], idx: int) -> str:
    clean = str(row.get("clean_text") or row.get("raw_text") or "").replace("\n", " ").strip()
    if len(clean) > 220:
        clean = clean[:217] + "..."
    sender = row.get("sender_id") or "matrix"
    event_id = row.get("event_id") or row.get("id") or ""
    return f"{idx}. {sender} {event_id}: {clean}"


def format_indy_directive_row(row: dict[str, Any], idx: int) -> str:
    body = str(row.get("body") or "").replace("\n", " ").strip()
    if len(body) > 220:
        body = body[:217] + "..."
    intent = row.get("intent") if isinstance(row.get("intent"), dict) else {}
    intent_summary = current_orchestration_intent_summary(intent) if intent else "local bonsai"
    return f"{idx}. {row.get('route', 'indy_orchestration_directive')} [{intent_summary}]: {body}"


def queued_dialogue_context(row: dict[str, Any]) -> tuple[Book, dict[str, Any], dict[str, Any]]:
    text = str(row.get("clean_text") or row.get("raw_text") or "")
    row_id = str(row.get("id") or row.get("event_id") or "")
    book = Book(
        id=f"waking_dialogue::{row_id}",
        name="ironclaw.waking_dialogue_stream",
        path="postgresql:///lucidota_state/ironclaw.waking_dialogue_stream",
        ext=".db",
        size_bytes=len(text.encode("utf-8")),
    )
    page = {
        "page": 1,
        "page_hash": sha_text(json.dumps(row, sort_keys=True, default=str)),
        "text": text,
        "extract_method": "waking_dialogue_stream",
        "chars": len(text),
    }
    entities = row.get("extracted_entities") if isinstance(row.get("extracted_entities"), dict) else {}
    parser = {
        "parser_version": "waking_dialogue_chat_v1",
        "packet_id": f"waking_dialogue::{row_id}",
        "confidence_bps": 10000,
        "terms": list(entities.get("hashtags") or []) + list(entities.get("slash_commands") or []),
        "claim_lifecycle": "WAKING_DIALOGUE_CHAT",
    }
    return book, page, parser


def parse_conduit_response_command(ans: str, dialogue_rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, str]:
    parts = ans.split(maxsplit=2)
    if len(parts) < 2 or parts[0].lower() != "respond":
        return None, "", "not_respond_command"
    idx = int(parts[1]) if parts[1].isdigit() else 1
    if not (1 <= idx <= len(dialogue_rows)):
        return None, "", "dialogue_index_out_of_range"
    reply_text = parts[2].strip() if len(parts) > 2 else ""
    return dialogue_rows[idx - 1], reply_text, ""


def dialogue_row_ref(row: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(row.get("id") or ""),
        "event_id": str(row.get("event_id") or ""),
        "room_id": str(row.get("room_id") or ""),
        "sender_id": str(row.get("sender_id") or ""),
        "receipt_id": str(row.get("receipt_id") or ""),
        "comms_channel": str(row.get("comms_channel") or "matrix"),
    }


def operator_response_id(row: dict[str, Any], reply_text: str) -> str:
    return "indy_response:" + sha_text(
        json.dumps(
            {
                "schema": "lucidota.indy_reads.operator_chat_response.v1",
                "dialogue_row": dialogue_row_ref(row),
                "reply_text": reply_text,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
    )[:16]


def pid_ram_guard() -> dict[str, Any]:
    telemetry = hardware_telemetry()
    return {
        "pid_check_performed": True,
        "process_id": os.getpid(),
        "rss_bytes": int(telemetry.get("rss_bytes") or 0),
        "memory_available_bytes": int(telemetry.get("memory_available_bytes") or 0),
        "memory_percent": float(telemetry.get("memory_percent") or 0.0),
        "cpu_count": int(telemetry.get("cpu_count") or 1),
        "heavy_model_launch_performed": False,
    }


def queue_operator_chat_response(
    row: dict[str, Any],
    reply_text: str,
    outbox: Path | None = None,
    *,
    db_identity: dict[str, Any] | None = None,
    db_api_status: str = "db_api_unavailable_fallback",
) -> dict[str, Any]:
    """Queue Indy_READs' chat response for the active operator surface.

    This is deliberately local and quiet: no Matrix/email/Signal send occurs
    here, and PID/RAM guard facts are captured before any sender can pick the
    packet up.
    """
    outbox = outbox or INDY_OPERATOR_RESPONSE_OUTBOX
    response_id = operator_response_id(row, reply_text)
    guard = pid_ram_guard()
    packet = {
        "schema": "lucidota.indy_reads.operator_chat_response.v1",
        "queued_at": now(),
        "response_id": response_id,
        "persona": PERSONA_DISPLAY,
        "target_path": "active_operator_chat_surface",
        "route": "luci_operator_direct_chat",
        "source_table": "ironclaw.waking_dialogue_stream",
        "dialogue_row": dialogue_row_ref(row),
        "body": reply_text,
        "body_sha256": sha_text(reply_text),
        "operator_delivery_status": "QUEUED_FOR_CHAT_SURFACE",
        "outbound_matrix_send_performed": False,
        "direct_network_send_performed": False,
        "db_api_status": db_api_status,
        "db_identity": db_identity or {},
        "pid_ram_guard": guard,
    }
    outbox.parent.mkdir(parents=True, exist_ok=True)
    with outbox.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(packet, sort_keys=True, ensure_ascii=False, default=str) + "\n")
    return {
        "ok": True,
        "response_id": response_id,
        "operator_delivery_status": packet["operator_delivery_status"],
        "operator_response_outbox": str(outbox),
        "db_api_status": db_api_status,
        "db_identity": db_identity or {},
        "outbound_matrix_send_performed": False,
        "direct_network_send_performed": False,
        "pid_ram_guard": guard,
    }


def mark_conduit_dialogue_done(row: dict[str, Any], response_id: str, reply_text: str) -> dict[str, Any]:
    if not db_available():
        return {"ok": False, "error": "database_unavailable", "processed_status": ""}
    row_id = str(row.get("id") or "")
    event_id = str(row.get("event_id") or "")
    body_sha = sha_text(reply_text)
    try:
        with psycopg.connect(DB_URL) as conn:  # type: ignore[union-attr]
            with conn.cursor() as cur:
                if row_id:
                    cur.execute(
                        """
                        UPDATE ironclaw.waking_dialogue_stream
                        SET processed_status = 'done',
                            receipt_id = CASE WHEN receipt_id = '' THEN %s ELSE receipt_id END,
                            last_response_id = %s,
                            last_response_body = %s,
                            last_response_body_sha256 = %s,
                            response_queued_at = now(),
                            response_delivery_status = 'QUEUED_FOR_CHAT_SURFACE',
                            updated_at = now()
                        WHERE id = %s::uuid
                        RETURNING id::text, processed_status, receipt_id,
                            last_response_id, response_delivery_status,
                            response_queued_at, last_response_body_sha256
                        """,
                        (response_id, response_id, reply_text, body_sha, row_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE ironclaw.waking_dialogue_stream
                        SET processed_status = 'done',
                            receipt_id = CASE WHEN receipt_id = '' THEN %s ELSE receipt_id END,
                            last_response_id = %s,
                            last_response_body = %s,
                            last_response_body_sha256 = %s,
                            response_queued_at = now(),
                            response_delivery_status = 'QUEUED_FOR_CHAT_SURFACE',
                            updated_at = now()
                        WHERE comms_channel = 'matrix'
                          AND event_id = %s
                        RETURNING id::text, processed_status, receipt_id,
                            last_response_id, response_delivery_status,
                            response_queued_at, last_response_body_sha256
                        """,
                        (response_id, response_id, reply_text, body_sha, event_id),
                    )
                rows = cur.fetchall()
            conn.commit()
        if not rows:
            return {
                "ok": False,
                "updated_rows": 0,
                "error": "dialogue_row_not_found",
                "processed_status": "",
                "response_id": response_id,
                "response_body_sha256": body_sha,
            }
        return {
            "ok": True,
            "updated_rows": len(rows),
            "dialogue_id": rows[0][0] if rows else "",
            "processed_status": rows[0][1] if rows else "",
            "receipt_id": rows[0][2] if rows else "",
            "response_id": rows[0][3] if rows else "",
            "response_delivery_status": rows[0][4] if rows else "",
            "response_queued_at": rows[0][5] if rows else "",
            "response_body_sha256": rows[0][6] if rows else body_sha,
        }
    except Exception as exc:
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)[:240], "processed_status": ""}


def default_conduit_reply(row: dict[str, Any]) -> str:
    clean = str(row.get("clean_text") or row.get("raw_text") or "").replace("\n", " ").strip()
    if len(clean) > 180:
        clean = clean[:177] + "..."
    sender = str(row.get("sender_id") or "operator")
    return f"Indy_READs saw queued chat from {sender}: {clean}"


def write_online_once_receipt(payload: dict[str, Any], receipt_dir: Path) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    digest = sha_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))[:16]
    stamp_value = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = receipt_dir / f"indy_reads_online_once_{stamp_value}_{digest}.json"
    payload["receipt_path"] = str(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def process_queued_conduit_once(
    st: dict[str, Any],
    *,
    limit: int = 5,
    response_text: str | None = None,
    receipt_dir: Path | str = INDY_CONDUIT_RECEIPT_DIR,
) -> dict[str, Any]:
    guard = pid_ram_guard()
    rows = load_queued_conduit_dialogue(limit=limit)
    if not rows:
        payload = {
            "schema": "lucidota.indy_reads.online_once_receipt.v1",
            "generated_at": now(),
            "ok": True,
            "status": "IDLE_NO_QUEUED_DIALOGUE",
            "row": None,
            "response": None,
            "pid_ram_guard": guard,
            "model_calls_performed": False,
            "heavy_model_launch_performed": False,
        }
        receipt_path = write_online_once_receipt(payload, Path(receipt_dir))
        return {**payload, "receipt_path": str(receipt_path)}
    row = rows[0]
    reply = response_text if response_text is not None else default_conduit_reply(row)
    response = record_conduit_dialogue_response(row, reply, st)
    payload = {
        "schema": "lucidota.indy_reads.online_once_receipt.v1",
        "generated_at": now(),
        "ok": bool(response.get("ok")),
        "status": "RESPONDED" if response.get("ok") else "RESPONSE_FAILED",
        "row": dialogue_row_ref(row),
        "response": response,
        "pid_ram_guard": guard,
        "model_calls_performed": False,
        "heavy_model_launch_performed": False,
    }
    receipt_path = write_online_once_receipt(payload, Path(receipt_dir))
    return {**payload, "receipt_path": str(receipt_path)}


def record_conduit_dialogue_response(row: dict[str, Any], reply_text: str, st: dict[str, Any]) -> dict[str, Any]:
    book, page, parser = queued_dialogue_context(row)
    response_id = operator_response_id(row, reply_text)
    processed_status_update = mark_conduit_dialogue_done(row, response_id, reply_text)
    db_api_status = "ok" if processed_status_update.get("ok") else "db_api_unavailable_fallback"
    operator_response = queue_operator_chat_response(
        row,
        reply_text,
        db_identity=processed_status_update,
        db_api_status=db_api_status,
    )
    record_indy_judgment(
        book=book,
        page=page,
        parser=parser,
        decision="comment",
        score=100 if reply_text else 50,
        score_label_value=score_label(100 if reply_text else 50),
        notes=reply_text,
        repair_instruction="",
        term_correction="",
        favorite_line="",
        confusion="",
        socket_active=transport_socket_active(),
        terminal_active=True,
        batch_size=int(st.get("slow_lane", {}).get("ingestion_batch_size", 0) or 0) or None,
        extra={
            "dialogue_row": row,
            "reply": reply_text,
            "response_kind": "terminal_conduit_response",
            "operator_response": operator_response,
            "processed_status_update": processed_status_update,
            "db_api_status": db_api_status,
            "outbound_matrix_send_performed": False,
            "direct_network_send_performed": False,
        },
    )
    tune_and_record_heartbeat(st, book, page, parser, score=100 if reply_text else 50, terminal_active=True)
    return {
        "ok": bool(operator_response.get("ok")) and bool(processed_status_update.get("ok")),
        "decision": "comment",
        "score": 100 if reply_text else 50,
        "response_id": operator_response["response_id"],
        "operator_response_queued": bool(operator_response.get("ok")),
        "operator_delivery_status": operator_response["operator_delivery_status"],
        "operator_response_outbox": operator_response["operator_response_outbox"],
        "db_api_status": db_api_status,
        "processed_status_update": processed_status_update,
        "outbound_matrix_send_performed": False,
        "direct_network_send_performed": False,
    }


def load_next_goal_queue() -> list[dict[str, Any]]:
    if not GOALS_NEXT_GOAL_QUEUE.exists():
        return []
    try:
        data = json.loads(GOALS_NEXT_GOAL_QUEUE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    orders = data.get("queue", [])
    if not isinstance(orders, list):
        return []
    return [o for o in orders if isinstance(o, dict)]


def goals_handoff_context(text: str, orders: list[dict[str, Any]]) -> tuple[Book, dict[str, Any], dict[str, Any]]:
    book = Book(
        id="goals_handoff",
        name="GOALS/CURRENT_HANDOFF.md",
        path=str(GOALS_HANDOFF_MD),
        ext=".md",
        size_bytes=len(text.encode("utf-8")),
    )
    page = {
        "page": 1,
        "page_hash": sha_text(text + json.dumps(orders, sort_keys=True, default=str)),
        "text": text,
        "extract_method": "goals_handoff",
        "chars": len(text),
    }
    parser = {
        "parser_version": "goals_handoff_chat_v1",
        "packet_id": "goals::handoff::v1",
        "confidence_bps": 10000 if orders else 7500,
        "terms": ["SESSION", "HANDOFF", "WORK_ORDER", "QUEUE"],
        "claim_lifecycle": "GOALS_CHAT",
    }
    return book, page, parser


def enqueue_goal_work_order(order: dict[str, Any]) -> dict[str, Any]:
    if not db_available():
        return {"ok": False, "error": "database_unavailable", "order_id": order.get("order_id", "")}
    queue = str(order.get("queue") or "control")
    workflow = str(order.get("workflow") or "goal_work_order")
    job_kind = str(order.get("job_kind") or "external_command")
    payload = dict(order.get("payload") or {})
    if not payload:
        payload = {
            "command": [
                ".venv/bin/python",
                "scripts/goal_swarm_dispatch.py",
                "--target",
                "generic",
                "--task",
                str(order.get("objective") or order.get("title") or "goal continuation"),
                "--jobs",
                "1",
                "--json",
            ]
        }
    idempotency_key = str(order.get("order_id") or sha256_obj(order))
    result: dict[str, Any] = {
        "ok": False,
        "queue": queue,
        "workflow": workflow,
        "job_kind": job_kind,
        "idempotency_key": idempotency_key,
    }
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO lucidota_control.absurd_queue_job
                      (queue_name, workflow_name, job_kind, idempotency_key, payload, priority, max_attempts, detail)
                    VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s,%s::jsonb)
                    ON CONFLICT (queue_name, idempotency_key) DO UPDATE SET updated_at=now()
                    RETURNING job_uuid::text, (xmax = 0) AS inserted_new
                    """,
                    (
                        queue,
                        workflow,
                        job_kind,
                        idempotency_key,
                        json.dumps(payload, default=str),
                        int(order.get("priority") or 100),
                        int(order.get("max_attempts") or 3),
                        json.dumps({"source": "indy_reads_chat", "order_id": order.get("order_id", ""), "title": order.get("title", "")}, default=str),
                    ),
                )
                job_uuid, inserted_new = cur.fetchone()
                if inserted_new:
                    cur.execute(
                        """
                        INSERT INTO lucidota_control.absurd_queue_event(job_uuid, queue_name, event_kind, detail)
                        VALUES (%s,%s,'enqueued',%s::jsonb)
                        """,
                        (
                            job_uuid,
                            queue,
                            json.dumps({"workflow": workflow, "job_kind": job_kind, "order_id": order.get("order_id", "")}, default=str),
                        ),
                    )
                conn.commit()
                result.update({"ok": True, "job_uuid": job_uuid, "inserted_new": bool(inserted_new)})
    except Exception as exc:
        result.update({"error": type(exc).__name__, "message": str(exc)})
    return result


def goals_chat_loop(st: dict[str, Any]) -> int:
    while True:
        handoff_text = load_goals_handoff_text()
        orders = load_next_goal_queue()
        book, page, parser = goals_handoff_context(handoff_text, orders)
        banner("GOALS CHAT — handoff / next orders / operator reply")
        print("CURRENT HANDOFF")
        print("─" * 96)
        wrap_print(handoff_text or "(no GOALS/CURRENT_HANDOFF.md found)", max_lines=40)
        print("─" * 96)
        conduit_rows = load_queued_conduit_dialogue(limit=5)
        print("\nMATRIX / CONDUIT QUEUE FOR Indy_READs")
        if not conduit_rows:
            print("(no queued ironclaw.waking_dialogue_stream rows visible to Indy_READs)")
        else:
            for i, row in enumerate(conduit_rows, 1):
                print(format_conduit_dialogue_row(row, i))
        indy_directives = load_queued_indy_directives(limit=5)
        print("\nINDY DIRECTIVE QUEUE")
        if not indy_directives:
            print("(no queued orchestration directives visible to Indy_READs)")
        else:
            for i, row in enumerate(indy_directives, 1):
                print(format_indy_directive_row(row, i))
        print("\nNEXT GOAL QUEUE")
        if not orders:
            print("(no GOALS/NEXT_GOAL_QUEUE.json found)")
        else:
            for i, order in enumerate(orders, 1):
                print(f"{i}. {order.get('title', order.get('order_id', 'goal'))}")
                print(f"   queue={order.get('queue', 'control')} workflow={order.get('workflow', '')} job_kind={order.get('job_kind', '')}")
                if order.get("objective"):
                    print(f"   objective={order.get('objective')}")
                elif order.get("summary"):
                    print(f"   summary={order.get('summary')}")
        print("\nReplies: `respond 1 text...`, `route groq llama-3.3-70b-versatile`, `approve 1`, `reject 2`, `note ...`, `enqueue 3`, `q` to quit")
        ans_raw = timed_input("reply> ", ATTENTION_TIMEOUT_SECONDS)
        if ans_raw is TIMEOUT_SENTINEL:
            run_autonomous_slow_lane_tick(st, book, page, parser)
            print("\nSession chat timed out; autonomous heartbeat written.")
            continue
        ans = str(ans_raw).strip()
        if ans.lower() in {"q", "quit", "exit"}:
            return 0
        lowered = ans.lower()
        decision = "comment"
        score = 50
        selected_order: dict[str, Any] | None = None
        enqueue_result: dict[str, Any] = {}
        notes = ans
        if lowered.startswith("respond "):
            selected_dialogue, reply_text, error = parse_conduit_response_command(ans, conduit_rows)
            if selected_dialogue is None:
                print(f"No queued dialogue response saved: {error}")
                continue
            response_result = record_conduit_dialogue_response(selected_dialogue, reply_text, st)
            print(f"Indy_READs terminal response saved: {response_result['decision']} {response_result['score']} | outbound_matrix_send=False")
            continue
        if lowered.startswith(("use ", "route ", "model ", "orchestrate ", "set model ", "switch to ")):
            intent, error = parse_orchestration_intent_command(ans)
            if intent is None:
                print(f"No orchestration intent saved: {error}")
                continue
            directive = compose_indy_orchestration_check_message(intent)
            directive_result = queue_indy_directive_message(directive, intent=intent)
            record_indy_judgment(
                book=book,
                page=page,
                parser=parser,
                decision="route",
                score=100,
                score_label_value=score_label(100),
                notes=f"{ans} | intent={current_orchestration_intent_summary(intent)} | directive={directive_result['delivery_status']}",
                repair_instruction="",
                term_correction="",
                favorite_line="",
                confusion="",
                socket_active=transport_socket_active(),
                terminal_active=True,
                batch_size=int(st.get("slow_lane", {}).get("ingestion_batch_size", 0) or 0) or None,
                extra={
                    "intent": intent,
                    "directive": directive,
                    "directive_result": directive_result,
                    "response_kind": "indy_orchestration_directive",
                },
            )
            print(f"Indy orchestration intent saved: {current_orchestration_intent_summary(intent)}")
            print(f"Directive queued: {directive_result['outbox']}")
            continue
        if lowered.startswith(("approve ", "enqueue ", "reject ")):
            parts = lowered.split()
            verb = parts[0]
            idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            if 1 <= idx <= len(orders):
                selected_order = orders[idx - 1]
            if verb == "reject":
                decision, score = "rejected", 10
            else:
                decision, score = "approved", 100
                if selected_order is not None:
                    enqueue_result = enqueue_goal_work_order(selected_order)
                    notes = f"{ans} | enqueue={enqueue_result.get('ok', False)}"
            if verb == "reject" and selected_order is not None:
                notes = f"{ans} | rejected"
        elif lowered.startswith("note "):
            notes = ans[5:].strip() or ans
        record_indy_judgment(
            book=book,
            page=page,
            parser=parser,
            decision=decision,
            score=score,
            score_label_value=score_label(score),
            notes=notes,
            repair_instruction="",
            term_correction="",
            favorite_line="",
            confusion="",
            socket_active=transport_socket_active(),
            terminal_active=True,
            batch_size=int(st.get("slow_lane", {}).get("ingestion_batch_size", 0) or 0) or None,
            extra={"selected_order": selected_order, "enqueue_result": enqueue_result, "reply": ans},
        )
        tune_and_record_heartbeat(st, book, page, parser, score=score, terminal_active=True)
        print(f"Judgment saved: {decision} {score} | enqueue={enqueue_result.get('ok', False)}")


def write_indy_boot_report(payload: dict[str, Any]) -> str:
    INDY_BOOT_RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    stamp_value = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest = sha_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str))[:16]
    path = INDY_BOOT_RECEIPT_DIR / f"indy_reads_bootstrap_{stamp_value}_{digest}.json"
    payload["receipt_path"] = str(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return str(path.relative_to(ROOT))


def persist_indy_bootstrap_rows(
    *,
    boot_packet: dict[str, Any],
    context: dict[str, Any],
    slot0: dict[str, Any],
    slot1: dict[str, Any],
) -> dict[str, Any]:
    defaults = default_indy_boot_content(
        slot0_text=str(slot0.get("text") or ""),
        slot1_text=str(slot1.get("text") or ""),
        boot_packet=boot_packet,
        context=context,
    )
    slot0_text = str(slot0.get("text") or "")
    slot1_text = str(slot1.get("text") or "")
    slot0_tokens = slot0.get("token_accounting") if isinstance(slot0.get("token_accounting"), dict) else {}
    slot1_tokens = slot1.get("token_accounting") if isinstance(slot1.get("token_accounting"), dict) else {}
    tokens_in = int(slot0_tokens.get("prompt_tokens") or 0) + int(slot1_tokens.get("prompt_tokens") or 0)
    tokens_out = int(slot0_tokens.get("completion_tokens") or 0) + int(slot1_tokens.get("completion_tokens") or 0)
    receipt_paths = [
        str(slot0.get("report_path") or ""),
        str(slot1.get("report_path") or ""),
    ]
    evidence_refs = [
        "04_RUNTIME/indy_reads_boot_packet.json",
        "04_RUNTIME/INDY_READS/indy_reads_service_manifest.json",
        *[p for p in receipt_paths if p],
    ]
    db_refs = [
        "lucidota_control.active_operation_mode",
        "lucidota_canon.manual_current",
        "lucidota_canon.root_orchestrator_current",
        "lucidota_canon.workload_audit_current",
        "lucidota_canon.workload_audit_telemetry_current",
        "lucidota_canon.indy_reads_self_model",
        "lucidota_canon.indy_reads_llmwiki_entry",
        "lucidota_canon.indy_reads_hunch_log",
        "lucidota_canon.indy_reads_learning_queue",
        "lucidota_canon.indy_reads_system_map",
        "lucidota_canon.indy_reads_mistake_ledger",
        "lucidota_canon.indy_reads_research_source",
        "lucidota_canon.indy_reads_metacognition_current",
    ]
    local_receipt_uuid = uuid.uuid5(uuid.NAMESPACE_URL, "|".join(receipt_paths + [slot0_text[:256], slot1_text[:256]]))
    boot_status = "PROVEN" if slot0.get("status") == "PASS" and slot1.get("status") == "PASS" else "PARTIAL"
    boot_report: dict[str, Any] = {
        "schema": "lucidota.indy_reads.bootstrap_report.v1",
        "boot_packet_ref": str(INDY_BOOT_PACKET_PATH),
        "slot_0": slot0,
        "slot_1": slot1,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "proof_status": boot_status,
        "receipt_uuid": str(local_receipt_uuid),
        "evidence_refs": evidence_refs,
        "db_refs": db_refs,
    }

    if not db_available():
        return {
            "ok": False,
            "boot_status": "DB_BLOCKED",
            "proof_status": "UNKNOWN",
            "receipt_uuid": str(local_receipt_uuid),
            "evidence_refs": evidence_refs,
            "db_refs": db_refs,
            "report_path": write_indy_boot_report({**boot_report, "db_blocked": True}),
        }

    self_model = defaults["self_model"]
    llmwiki_entry = defaults["llmwiki_entry"]
    hunch_log = defaults["hunch_log"]
    system_map = defaults["system_map"]
    mistake_ledger = defaults["mistake_ledger"]
    learning_queue = defaults["learning_queue"]
    metacognition_current = defaults["metacognition_current"]
    research_source = {
        "actor_id": "indy_reads_runtime",
        "source_name": "live DB/manual/workload/mode surfaces",
        "source_type": "db_surface",
        "source_locator": "lucidota_control.active_operation_mode + lucidota_canon.manual_current + lucidota_canon.root_orchestrator_current + lucidota_canon.workload_audit_current",
        "access_status": "readable",
        "summary": "Current boot used live DB/PostgREST truth surfaces and the Bonsai local model lane.",
        "evidence_refs": evidence_refs,
        "db_refs": db_refs,
        "proof_status": boot_status,
        "functionality_explanation": "Indy_READs research source inventory; keep secrets out, keep evidence refs in.",
        "ontology_index": {
            "primitive_refs": ["state", "duplex", "allocation"],
            "claim_type": "research_source_boot",
            "evidence_type": "boot_packet_and_slot_receipts",
            "actor_role": "indy_reads_runtime",
            "subsystem_refs": ["active_operation_mode", "manual_current", "root_orchestrator_current", "workload_audit_current"],
            "risk_tier": "T3",
            "proof_status": boot_status,
            "receipt_refs": ["indy_reads_exocortex_activation_gate"],
            "next_route": ["workload_audit_current", "manual_current", "root_orchestrator_current"],
        },
    }

    try:
        with psycopg.connect(DB_URL) as conn:  # type: ignore[union-attr]
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO lucidota_indy.indy_reads_self_model
                      (actor_id, author, role, boundaries, voice, relationship_to_operator, relationship_to_LUCIDOTA,
                       relationship_to_northern_strike, relationship_to_Krampus, relationship_to_Santa,
                       investigation_style, learning_style, preferred_tools, evidence_standard, receipt_standard,
                       mistake_handling, curiosity_targets, current_limitations, next_upgrade, summary,
                       goals_refs, confidence, evidence_refs, db_refs, proof_status, functionality_explanation,
                       ontology_index)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb)
                    RETURNING self_model_id::text
                    """,
                    (
                        self_model["actor_id"],
                        self_model["author"],
                        self_model["role"],
                        self_model["boundaries"],
                        self_model["voice"],
                        self_model["relationship_to_operator"],
                        self_model["relationship_to_LUCIDOTA"],
                        self_model["relationship_to_northern_strike"],
                        self_model["relationship_to_Krampus"],
                        self_model["relationship_to_Santa"],
                        self_model["investigation_style"],
                        self_model["learning_style"],
                        self_model["preferred_tools"],
                        self_model["evidence_standard"],
                        self_model["receipt_standard"],
                        self_model["mistake_handling"],
                        self_model["curiosity_targets"],
                        self_model["current_limitations"],
                        self_model["next_upgrade"],
                        self_model["summary"],
                        json.dumps(self_model["goals_refs"], default=str),
                        float(self_model["confidence"]),
                        json.dumps(evidence_refs, default=str),
                        json.dumps(db_refs, default=str),
                        self_model["proof_status"],
                        self_model["functionality_explanation"],
                        json.dumps(self_model["ontology_index"], default=str),
                    ),
                )
                self_model_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO lucidota_indy.indy_reads_llmwiki_entry
                      (actor_id, author, topic, summary, body, confidence, evidence_refs, db_refs, next_questions,
                       mistake_risk, promotion_candidate, proof_status, functionality_explanation, ontology_index)
                    VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s,%s,%s::jsonb)
                    RETURNING llmwiki_entry_id::text
                    """,
                    (
                        llmwiki_entry["actor_id"],
                        llmwiki_entry["author"],
                        llmwiki_entry["topic"],
                        llmwiki_entry["summary"],
                        llmwiki_entry["body"],
                        float(llmwiki_entry["confidence"]),
                        json.dumps(evidence_refs, default=str),
                        json.dumps(db_refs, default=str),
                        json.dumps(llmwiki_entry["next_questions"], default=str),
                        llmwiki_entry["mistake_risk"],
                        bool(llmwiki_entry["promotion_candidate"]),
                        llmwiki_entry["proof_status"],
                        llmwiki_entry["functionality_explanation"],
                        json.dumps(llmwiki_entry["ontology_index"], default=str),
                    ),
                )
                llmwiki_entry_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO lucidota_indy.indy_reads_hunch_log
                      (actor_id, topic, hunch, confidence, evidence_refs, db_refs, next_questions, proof_status,
                       functionality_explanation, ontology_index)
                    VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb)
                    RETURNING hunch_log_id::text
                    """,
                    (
                        hunch_log["actor_id"],
                        hunch_log["topic"],
                        hunch_log["hunch"],
                        float(hunch_log["confidence"]),
                        json.dumps(evidence_refs, default=str),
                        json.dumps(db_refs, default=str),
                        json.dumps(hunch_log["next_questions"], default=str),
                        hunch_log["proof_status"],
                        hunch_log["functionality_explanation"],
                        json.dumps(hunch_log["ontology_index"], default=str),
                    ),
                )
                hunch_log_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO lucidota_indy.indy_reads_learning_queue
                      (actor_id, topic, summary, status, priority, next_route, evidence_refs, db_refs, proof_status,
                       functionality_explanation, ontology_index)
                    VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb)
                    RETURNING learning_queue_id::text
                    """,
                    (
                        learning_queue["actor_id"],
                        learning_queue["topic"],
                        learning_queue["summary"],
                        learning_queue["status"],
                        int(learning_queue["priority"]),
                        learning_queue["next_route"],
                        json.dumps(evidence_refs, default=str),
                        json.dumps(db_refs, default=str),
                        learning_queue["proof_status"],
                        learning_queue["functionality_explanation"],
                        json.dumps(learning_queue["ontology_index"], default=str),
                    ),
                )
                learning_queue_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO lucidota_indy.indy_reads_system_map
                      (actor_id, topic, summary, subsystem_refs, evidence_refs, db_refs, proof_status,
                       functionality_explanation, ontology_index)
                    VALUES (%s,%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb)
                    RETURNING system_map_id::text
                    """,
                    (
                        system_map["actor_id"],
                        system_map["topic"],
                        system_map["summary"],
                        json.dumps(system_map["subsystem_refs"], default=str),
                        json.dumps(evidence_refs, default=str),
                        json.dumps(db_refs, default=str),
                        system_map["proof_status"],
                        system_map["functionality_explanation"],
                        json.dumps(system_map["ontology_index"], default=str),
                    ),
                )
                system_map_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO lucidota_indy.indy_reads_mistake_ledger
                      (actor_id, mistake_summary, mistake_risk, correction, evidence_refs, db_refs, proof_status,
                       functionality_explanation, ontology_index)
                    VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb)
                    RETURNING mistake_ledger_id::text
                    """,
                    (
                        mistake_ledger["actor_id"],
                        mistake_ledger["mistake_summary"],
                        mistake_ledger["mistake_risk"],
                        mistake_ledger["correction"],
                        json.dumps(evidence_refs, default=str),
                        json.dumps(db_refs, default=str),
                        mistake_ledger["proof_status"],
                        mistake_ledger["functionality_explanation"],
                        json.dumps(mistake_ledger["ontology_index"], default=str),
                    ),
                )
                mistake_ledger_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO lucidota_indy.indy_reads_research_source
                      (actor_id, source_name, source_type, source_locator, access_status, summary, evidence_refs, db_refs,
                       proof_status, functionality_explanation, ontology_index)
                    VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb)
                    RETURNING research_source_id::text
                    """,
                    (
                        research_source["actor_id"],
                        research_source["source_name"],
                        research_source["source_type"],
                        research_source["source_locator"],
                        research_source["access_status"],
                        research_source["summary"],
                        json.dumps(evidence_refs, default=str),
                        json.dumps(db_refs, default=str),
                        research_source["proof_status"],
                        research_source["functionality_explanation"],
                        json.dumps(research_source["ontology_index"], default=str),
                    ),
                )
                research_source_id = cur.fetchone()[0]

                cur.execute(
                    """
                    INSERT INTO lucidota_indy.indy_reads_metacognition_current_state
                      (state_key, actor_id, owner_role, what_i_am, what_i_am_for, operator_model, case_model, system_model,
                       learning_next, refusal_standard, self_model_ref, llmwiki_ref, hunch_log_ref, system_map_ref,
                       mistake_ledger_ref, learning_queue_ref, research_source_ref, boot_packet_ref, evidence_refs, db_refs,
                       proof_status, functionality_explanation, ontology_index)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s::jsonb)
                    ON CONFLICT (state_key) DO UPDATE SET
                        actor_id = EXCLUDED.actor_id,
                        owner_role = EXCLUDED.owner_role,
                        what_i_am = EXCLUDED.what_i_am,
                        what_i_am_for = EXCLUDED.what_i_am_for,
                        operator_model = EXCLUDED.operator_model,
                        case_model = EXCLUDED.case_model,
                        system_model = EXCLUDED.system_model,
                        learning_next = EXCLUDED.learning_next,
                        refusal_standard = EXCLUDED.refusal_standard,
                        self_model_ref = EXCLUDED.self_model_ref,
                        llmwiki_ref = EXCLUDED.llmwiki_ref,
                        hunch_log_ref = EXCLUDED.hunch_log_ref,
                        system_map_ref = EXCLUDED.system_map_ref,
                        mistake_ledger_ref = EXCLUDED.mistake_ledger_ref,
                        learning_queue_ref = EXCLUDED.learning_queue_ref,
                        research_source_ref = EXCLUDED.research_source_ref,
                        boot_packet_ref = EXCLUDED.boot_packet_ref,
                        evidence_refs = EXCLUDED.evidence_refs,
                        db_refs = EXCLUDED.db_refs,
                        proof_status = EXCLUDED.proof_status,
                        functionality_explanation = EXCLUDED.functionality_explanation,
                        ontology_index = EXCLUDED.ontology_index,
                        refreshed_at = now()
                    RETURNING state_key
                    """,
                    (
                        metacognition_current["state_key"],
                        metacognition_current["actor_id"],
                        metacognition_current["owner_role"],
                        metacognition_current["what_i_am"],
                        metacognition_current["what_i_am_for"],
                        metacognition_current["operator_model"],
                        metacognition_current["case_model"],
                        metacognition_current["system_model"],
                        metacognition_current["learning_next"],
                        metacognition_current["refusal_standard"],
                        self_model_id,
                        llmwiki_entry_id,
                        hunch_log_id,
                        system_map_id,
                        mistake_ledger_id,
                        learning_queue_id,
                        research_source_id,
                        metacognition_current["boot_packet_ref"],
                        json.dumps(evidence_refs, default=str),
                        json.dumps(db_refs, default=str),
                        metacognition_current["proof_status"],
                        metacognition_current["functionality_explanation"],
                        json.dumps(metacognition_current["ontology_index"], default=str),
                    ),
                )
                cur.fetchone()

                workload_summary = context.get("workload_audit_current", {})
                action_summary = "Indy_READs booted through IronClaw local model runtime with Bonsai slot_0 synthesis and slot_1 skeptic verification, then wrote self-model/wiki/hunch/system-map/mistake/learning/metacognition rows."
                debt_reason = ""
                proof_status = boot_status
                if boot_status != "PROVEN":
                    debt_reason = "first boot is still provisional until the workload/mode surfaces are fully reflected"
                token_source = slot0_tokens.get("source") or slot1_tokens.get("source") or "local_counter"
                if token_source not in {"provider_api", "local_counter", "receipt_file", "manual_operator_input", "unknown"}:
                    token_source = "local_counter"
                cur.execute(
                    """
                    INSERT INTO lucidota_audit.workload_audit_ledger
                      (actor_id, actor_class, caller, provider, model_id, action_summary, tokens_in, tokens_out,
                       token_source, receipt_uuid, evidence_refs, proof_status, debt_reason, functionality_explanation,
                       ontology_index)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb)
                    RETURNING workload_audit_uuid::text
                    """,
                    (
                        "indy_reads_runtime",
                        "indy_reads",
                        "indy_reads",
                        "local",
                        "bonsai_q1_0",
                        action_summary,
                        tokens_in,
                        tokens_out,
                        token_source,
                        str(local_receipt_uuid),
                        json.dumps(evidence_refs, default=str),
                        proof_status,
                        debt_reason,
                        "Tracks live token expenditures, execution workloads, and racing metrics between Codex Cloud and Indy Local models to prevent un-indexed handwaving.",
                        json.dumps(
                            {
                                "primitive_refs": ["telemetry", "duplex", "allocation"],
                                "claim_type": "indy_boot_workload",
                                "evidence_type": "slot_receipts_and_db_rows",
                                "actor_role": "indy_reads_runtime",
                                "subsystem_refs": [
                                    "active_operation_mode",
                                    "manual_current",
                                    "root_orchestrator_current",
                                    "workload_audit_current",
                                    "indy_reads_self_model",
                                    "indy_reads_llmwiki_entry",
                                    "indy_reads_metacognition_current",
                                ],
                                "risk_tier": "T3",
                                "proof_status": proof_status,
                                "receipt_refs": ["indy_reads_exocortex_activation_gate"],
                                "next_route": ["workload_audit_current", "active_operation_mode", "manual_current", "root_orchestrator_current"],
                            },
                            default=str,
                        ),
                    ),
                )
                workload_audit_uuid = cur.fetchone()[0]
            conn.commit()
    except Exception as exc:
        return {
            "ok": False,
            "boot_status": "DB_WRITE_FAILED",
            "proof_status": "UNKNOWN",
            "receipt_uuid": str(local_receipt_uuid),
            "error": type(exc).__name__,
            "message": str(exc),
            "report_path": write_indy_boot_report({**boot_report, "db_write_failed": True, "error": type(exc).__name__, "message": str(exc)}),
        }

    report = {
        **boot_report,
        "ok": True,
        "boot_status": "BOOTED",
        "proof_status": "PROVEN" if proof_status == "PROVEN" else "PARTIAL",
        "self_model_id": self_model_id,
        "llmwiki_entry_id": llmwiki_entry_id,
        "hunch_log_id": hunch_log_id,
        "learning_queue_id": learning_queue_id,
        "system_map_id": system_map_id,
        "mistake_ledger_id": mistake_ledger_id,
        "research_source_id": research_source_id,
        "workload_audit_uuid": workload_audit_uuid,
        "receipt_uuid": str(local_receipt_uuid),
        "evidence_refs": evidence_refs,
        "db_refs": db_refs,
        "report_path": "",
    }
    report["report_path"] = write_indy_boot_report(report)
    return report


def run_indy_bootstrap(*, force: bool = False, timeout_sec: float = 180.0) -> dict[str, Any]:
    boot_packet = load_boot_packet()
    context = indy_boot_context_snapshot()
    existing_self_model_count = int(fetch_scalar("SELECT count(*) FROM lucidota_indy.indy_reads_self_model") or 0)
    if existing_self_model_count > 0 and not force:
        report = {
            "ok": True,
            "boot_status": "ALREADY_BOOTED",
            "proof_status": "PROVEN",
            "receipt_uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, "indy_reads_boot_already_booted")),
            "existing_self_model_count": existing_self_model_count,
            "evidence_refs": ["lucidota_canon.indy_reads_self_model", "lucidota_canon.indy_reads_metacognition_current"],
            "db_refs": [
                "lucidota_control.active_operation_mode",
                "lucidota_canon.manual_current",
                "lucidota_canon.root_orchestrator_current",
                "lucidota_canon.workload_audit_current",
                "lucidota_canon.indy_reads_self_model",
                "lucidota_canon.indy_reads_metacognition_current",
            ],
        }
        report["report_path"] = write_indy_boot_report(report)
        return report

    slot0 = run_boot_slot(lane="bonsai_q1_0", slot_role="slot_0_synthesis", boot_packet=boot_packet, context=context, max_tokens=512, timeout_sec=timeout_sec)
    slot1 = run_boot_slot(lane="bonsai_q1_0", slot_role="slot_1_skeptic_verifier", boot_packet=boot_packet, context=context, max_tokens=384, timeout_sec=timeout_sec)
    return persist_indy_bootstrap_rows(boot_packet=boot_packet, context=context, slot0=slot0, slot1=slot1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["reader", "chat", "journal", "bootstrap"])
    ap.add_argument("--mode", dest="mode_flag", choices=["reader", "chat", "journal", "bootstrap"])
    ap.add_argument("--respond-once", action="store_true", help="Process one queued Indy/Conduit chat row and exit.")
    ap.add_argument("--response-text", default=None, help="Optional explicit response body for --respond-once.")
    ap.add_argument("--journal-title", default=None)
    ap.add_argument("--journal-body", default=None)
    ap.add_argument("--journal-kind", default="note")
    ap.add_argument("--force-bootstrap", action="store_true", help="Re-run Indy_READs bootstrap even if a self-model row already exists.")
    ap.add_argument("--receipt-dir", default=str(INDY_CONDUIT_RECEIPT_DIR))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    mode = args.mode_flag or args.mode or "reader"
    ensure_dirs()
    st = load_state()
    tune_and_record_heartbeat(st, None, None, None, score=None, terminal_active=sys.stdin.isatty())
    if mode == "journal":
        title = (args.journal_title or "Indy Journal Entry").strip()
        body = (args.journal_body or sys.stdin.read() or "").strip()
        if not body:
            body = title
        journal = write_journal_entry(title=title, body=body, kind=args.journal_kind)
        wiki = write_wiki_page(title=title, body=body)
        result = {"ok": True, "journal": journal, "wiki": wiki}
        if args.json:
            print(json.dumps(result, sort_keys=True, default=str))
        else:
            print("INDY_JOURNAL=PASS")
            print(f"JOURNAL_PATH={journal['path']}")
            print(f"WIKI_PATH={wiki['path']}")
        return 0
    if mode == "bootstrap":
        result = run_indy_bootstrap(force=args.force_bootstrap)
        if args.json:
            print(json.dumps(result, sort_keys=True, default=str))
        else:
            print(f"INDY_BOOTSTRAP={result.get('boot_status', 'UNKNOWN')}")
            print(f"RECEIPT_UUID={result.get('receipt_uuid', '')}")
            print(f"REPORT_PATH={result.get('report_path', '')}")
        return 0 if result.get("ok") else 1
    if mode == "chat":
        if args.respond_once:
            result = process_queued_conduit_once(st, response_text=args.response_text, receipt_dir=Path(args.receipt_dir))
            if args.json:
                print(json.dumps(result, sort_keys=True, default=str))
            else:
                print(f"INDY_READS_CHAT={result['status']}")
                print(f"RECEIPT_PATH={result['receipt_path']}")
                if result.get("response"):
                    print(f"RESPONSE_ID={result['response'].get('response_id', '')}")
            return 0 if result.get("ok") else 1
        return goals_chat_loop(st)
    while True:
        b = pick_book(st)
        if not b:
            banner("EXIT")
            print(f"CSV data: {CSV_PATH}")
            print(f"Persona config: {PERSONA_CONFIG_PATH}")
            print(f"Adapter registry: {ADAPTER_REGISTRY_PATH}")
            print("INDY_READs paused.")
            return 0
        standard_flow(b, st)


if __name__ == "__main__":
    raise SystemExit(main())
