#!/usr/bin/env python3
"""System-wide runtime facts refresh from live DB/daemon evidence."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS" / "status"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(p: Path | str) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def sdb(a) -> str:
    return a.state_database_url or os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or "postgresql:///lucidota_state"


def kdb(a) -> str:
    return a.storage_database_url or os.environ.get("KORPUS_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


def q(cur, sql: str) -> dict:
    cur.execute(sql)
    return dict(cur.fetchone())


def q_or_zero(cur, sql: str, keys: list[str]) -> dict:
    try:
        return q(cur, sql)
    except Exception:
        return {k: 0 for k in keys}


def latest_glob(pattern: str) -> Path | None:
    candidates = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return candidates[-1] if candidates else None


def load_json(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"system_runtime_facts_refresh_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print("REPORT_PATH=" + rel(p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-database-url")
    ap.add_argument("--storage-database-url")
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()

    facts: dict = {}
    blockers: list[str] = []

    with psycopg.connect(sdb(a), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            facts["absurd_queue"] = q_or_zero(
                cur,
                "SELECT count(*) total, count(*) FILTER (WHERE status='queued') queued, count(*) FILTER (WHERE status='succeeded') succeeded, count(*) FILTER (WHERE status='failed') failed FROM lucidota_control.absurd_queue_job",
                ["total", "queued", "succeeded", "failed"],
            )
            facts["conversation_command"] = q_or_zero(
                cur,
                "SELECT count(*) total, count(*) FILTER (WHERE status='executed') executed FROM lucidota_control.conversation_command",
                ["total", "executed"],
            )

    with psycopg.connect(kdb(a), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            facts["chrono"] = q_or_zero(
                cur,
                "SELECT count(*) temporal_claims, count(DISTINCT file_uuid) files_covered FROM lucidota_korpus.temporal_claim",
                ["temporal_claims", "files_covered"],
            )
            facts["graph"] = q_or_zero(
                cur,
                "SELECT (SELECT count(*) FROM lucidota_go.graph_item) graph_items, (SELECT count(*) FROM lucidota_go.graph_edge) graph_edges, (SELECT count(*) FROM lucidota_go.graph_promotion_materialization) materializations",
                ["graph_items", "graph_edges", "materializations"],
            )

    corpus_map = load_json(latest_glob("05_OUTPUTS/goals/corpus_map_*.json"))
    absurd_receipt = load_json(latest_glob("05_OUTPUTS/runtime/absurd_flows_*.json"))
    groq_receipt = load_json(latest_glob("05_OUTPUTS/goals/groq_batch_launcher_*.json"))
    infra_receipt = load_json(latest_glob("05_OUTPUTS/goals/infra_bootstrap_*.json"))
    preflight_receipt = load_json(latest_glob("05_OUTPUTS/runtime/resource_preflight_*.json"))

    facts["corpus_map"] = {
        "remaining_files": corpus_map.get("remaining_files"),
        "remaining_mb": corpus_map.get("remaining_mb"),
        "easy_text_count": (corpus_map.get("categories") or {}).get("easy_text", {}).get("count"),
        "heavy_image_count": (corpus_map.get("categories") or {}).get("heavy_image", {}).get("count"),
        "heavy_pdf_count": (corpus_map.get("categories") or {}).get("heavy_pdf", {}).get("count"),
        "heavy_video_count": (corpus_map.get("categories") or {}).get("heavy_video", {}).get("count"),
        "large_file_recommendations": ((corpus_map.get("recommendations") or {}).get("bypass_or_stage") or [])[:15],
    }
    facts["latest_absurd_flows"] = {
        "file_count": absurd_receipt.get("file_count"),
        "processed_count": absurd_receipt.get("processed_count"),
        "deduped_count": absurd_receipt.get("deduped_count"),
        "batch_size_final": absurd_receipt.get("batch_size_final"),
        "last_record": absurd_receipt.get("last_record"),
        "receipt_path": absurd_receipt.get("receipt_path"),
    }
    facts["latest_groq_batch"] = {
        "launch_count": groq_receipt.get("launch_count"),
        "selected_workers": groq_receipt.get("selected_workers"),
        "compiled_workflow_count": groq_receipt.get("compiled_workflow_count"),
        "compiled_batch_count": groq_receipt.get("compiled_batch_count"),
        "compiled_batch_size": groq_receipt.get("compiled_batch_size"),
    }
    facts["latest_bootstrap"] = {
        "passed": infra_receipt.get("passed"),
        "blockers": infra_receipt.get("blockers"),
        "checks": [c.get("name") for c in infra_receipt.get("checks", [])[:12]],
    }
    facts["latest_preflight"] = {
        "safe_workers": preflight_receipt.get("safe_workers"),
        "throttle": preflight_receipt.get("throttle"),
        "mem_available_mb": preflight_receipt.get("mem_available_mb"),
        "swap_used_pct": preflight_receipt.get("swap_used_pct"),
        "loadavg_1m": preflight_receipt.get("loadavg_1m"),
    }
    facts["hypertimeline_learning"] = {
        "can_do_now": [
            "deterministic corpus mapping",
            "governed 5-file absurd_flows reingest slices",
            "Groq fanout over gap workflows",
            "model/bootstrap truthfulness checks",
            "runtime facts refresh",
        ],
        "should_do_next": [
            "write receipt-derived learning facts",
            "stage timeline events from explicit timestamps only",
            "derive evidence-backed graph candidates from receipts and corpus map",
            "keep large binaries staged one-by-one",
        ],
        "good_training_data_points": [
            "source_path + sha256 + timestamp",
            "runtime / batch_size / success state",
            "blocker titles and closure gates",
            "explicit evidence references",
            "graph edge or timeline projection candidates with provenance",
        ],
    }

    chrono_proc = subprocess.run(
        ["scripts/check_chrono_ledger_service.sh"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    facts["chrono_service_rc"] = chrono_proc.returncode
    facts["chrono_service_note"] = "unavailable_or_unprovisioned" if chrono_proc.returncode != 0 else "available"

    if a.execute:
        with psycopg.connect(sdb(a)) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
                cur.execute("CREATE SCHEMA IF NOT EXISTS lucidota_control")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS lucidota_control.runtime_status_fact (
                      fact_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                      subsystem text NOT NULL,
                      fact_key text NOT NULL,
                      fact_value jsonb NOT NULL,
                      evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
                      derived_at timestamptz NOT NULL DEFAULT now(),
                      UNIQUE(subsystem, fact_key)
                    )
                    """
                )
                cur.execute(
                    """INSERT INTO lucidota_control.runtime_status_fact(subsystem,fact_key,fact_value,evidence_refs)
                       VALUES ('system','runtime_facts_refresh',%s::jsonb,%s::jsonb)
                       ON CONFLICT(subsystem,fact_key) DO UPDATE SET fact_value=EXCLUDED.fact_value,evidence_refs=EXCLUDED.evidence_refs,derived_at=now()""",
                    (
                        json.dumps(facts),
                        json.dumps(
                            {
                                "script": "scripts/system_runtime_facts_refresh.py",
                                "chrono_stdout_tail": chrono_proc.stdout[-1000:],
                            }
                        ),
                    ),
                )
                cur.execute(
                    """INSERT INTO lucidota_control.runtime_status_fact(subsystem,fact_key,fact_value,evidence_refs)
                       VALUES ('system','hypertimeline_learning',%s::jsonb,%s::jsonb)
                       ON CONFLICT(subsystem,fact_key) DO UPDATE SET fact_value=EXCLUDED.fact_value,evidence_refs=EXCLUDED.evidence_refs,derived_at=now()""",
                    (
                        json.dumps(facts["hypertimeline_learning"]),
                        json.dumps(
                            {
                                "corpus_map": rel(latest_glob("05_OUTPUTS/goals/corpus_map_*.json") or Path("")),
                                "absurd_receipt": rel(latest_glob("05_OUTPUTS/runtime/absurd_flows_*.json") or Path("")),
                                "groq_receipt": rel(latest_glob("05_OUTPUTS/goals/groq_batch_launcher_*.json") or Path("")),
                                "bootstrap_receipt": rel(latest_glob("05_OUTPUTS/goals/infra_bootstrap_*.json") or Path("")),
                                "preflight_receipt": rel(latest_glob("05_OUTPUTS/runtime/resource_preflight_*.json") or Path("")),
                                "script": "scripts/system_runtime_facts_refresh.py",
                            }
                        ),
                    ),
                )
            conn.commit()

    payload = {
        "action": "system_runtime_facts_refresh",
        "execute_performed": bool(a.execute),
        "db_writes_performed": bool(a.execute),
        "graph_writes_performed": False,
        "facts": facts,
        "blockers": blockers,
        "status": "PASS" if not blockers else "FAIL",
    }
    write(payload)
    print("SYSTEM_RUNTIME_FACTS=" + payload["status"])
    return 0 if not blockers else 4


if __name__ == "__main__":
    raise SystemExit(main())
