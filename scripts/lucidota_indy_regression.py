#!/usr/bin/env python3
"""Indy_Reads regression smoke: brief, corrections, auth, queue, corpus."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
DB = os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state")
PY = ROOT / ".venv" / "bin" / "python"
if not PY.exists():
    PY = Path(sys.executable)


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PY), str(ROOT / "scripts" / "lucidota_indy_brief.py"), *args],
        text=True,
        capture_output=True,
        check=False,
    )


def as_json(proc: subprocess.CompletedProcess[str]) -> dict:
    if proc.returncode != 0:
        return {"_error": proc.stderr.strip() or proc.stdout.strip(), "_returncode": proc.returncode}
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"_error": f"json decode failed: {exc}", "_stdout": proc.stdout[:500]}


def cleanup(correction_id: str | None, queue_id: str | None, auth_service: str, auth_account: str) -> None:
    with psycopg.connect(DB) as conn:
        if correction_id:
            conn.execute(
                "UPDATE lucidota_indy.task_memory SET status='archived', updated_at=now() WHERE memory_id=%s::uuid",
                (correction_id,),
            )
        if queue_id:
            conn.execute(
                "UPDATE lucidota_indy.side_queue SET status='dismissed', updated_at=now() WHERE queue_id=%s::uuid",
                (queue_id,),
            )
        conn.execute(
            "DELETE FROM lucidota_indy.auth_inventory WHERE service=%s AND account_hint=%s",
            (auth_service, auth_account),
        )
        conn.commit()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    brief = as_json(run(["--json"]))
    checks.update(
        {
            "brief_identity": brief.get("identity") == "Indy_Reads",
            "brief_citations": len(brief.get("citations", [])) >= 4,
            "brief_counters": "workflow_events" in brief.get("counters", {}),
            "brief_queue_present": "queue" in brief,
            "brief_auth_present": "auth" in brief,
            "brief_corpus_present": brief.get("corpus", {}).get("unit_count", 0) >= 12,
        }
    )

    corpus = as_json(run(["corpus", "--json"]))
    checks.update(
        {
            "corpus_local_scope": corpus.get("source_scope") == "allow-listed local project-brain markdown only; no Drive access",
            "corpus_distilled": bool(corpus.get("distilled", {}).get("runtime_jobs")),
            "corpus_units": corpus.get("unit_count", 0) >= 20,
            "corpus_sources": all(src.get("path", "").startswith("00_PROJECT_BRAIN/") for src in corpus.get("sources", [])),
        }
    )

    corr = as_json(
        run(
            [
                "correct",
                "Regression correction loop smoke",
                "--body",
                "operator correction captured without changing command flow",
                "--source",
                "regression",
                "--json",
            ]
        )
    )
    correction_id = corr.get("memory_id")
    checks["correction_insert"] = bool(correction_id)

    queue = as_json(
        run(
            [
                "queue",
                "Regression quiet queue smoke",
                "--item-type",
                "review",
                "--body",
                "prove no-interruption side queue path",
                "--urgency",
                "low",
                "--json",
            ]
        )
    )
    queue_id = queue.get("queue_id")
    listed_queue = as_json(run(["queue-list", "--limit", "20", "--json"]))
    checks["queue_insert"] = bool(queue_id)
    checks["queue_list"] = bool(queue_id) and any(item.get("title") == "Regression quiet queue smoke" for item in listed_queue.get("queue", []))

    auth_service = "regression-auth-smoke"
    auth_account = "indy-regression-local"
    auth = as_json(
        run(
            [
                "auth-add",
                auth_service,
                "--account-hint",
                auth_account,
                "--status",
                "not_exposed",
                "--scope-note",
                "regression proves redacted inventory only",
                "--secret-ref",
                "regression-pointer-not-a-secret",
                "--json",
            ]
        )
    )
    listed_auth = as_json(run(["auth-list", "--limit", "50", "--json"]))
    checks["auth_upsert"] = auth.get("service") == auth_service and auth.get("status") == "not_exposed"
    checks["auth_list"] = any(
        item.get("service") == auth_service
        and item.get("account_hint") == auth_account
        and item.get("access_status") == "not_exposed"
        for item in listed_auth.get("auth", [])
    )

    cleanup(str(correction_id) if correction_id else None, str(queue_id) if queue_id else None, auth_service, auth_account)
    checks["cleanup"] = True

    details["brief_corpus"] = brief.get("corpus", {})
    details["corpus_sha256"] = corpus.get("artifact_sha256")
    details["failures"] = [name for name, ok in checks.items() if not ok]
    report = {"ok": all(checks.values()), "checks": checks, "details": details}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
