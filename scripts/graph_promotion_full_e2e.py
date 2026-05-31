#!/usr/bin/env python3
"""Full graph promotion E2E: authorized command envelope -> promotion helper -> receipts.

Mutation class: materializer (writes to canonical graph_item + graph_journal via helper).
Requires --execute AND operator to invoke directly. No model calls.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "05_OUTPUTS/graph"

_STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", "postgresql:///lucidota_state")


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def rel(p) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def runc(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=180)
    out = {"cmd": " ".join(cmd), "rc": p.returncode,
           "stdout_tail": p.stdout[-2000:], "stderr_tail": p.stderr[-2000:]}
    for line in p.stdout.splitlines():
        if "=" in line and line.split("=", 1)[0] in {
            "REPORT_PATH", "COMMAND_UUID", "MATERIALIZATION_UUID",
            "GRAPH_ITEM_UUID", "JOURNAL_UUID",
        }:
            out[line.split("=", 1)[0]] = line.split("=", 1)[1]
    return out


def write(payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"graph_promotion_full_e2e_{stamp()}.json"
    payload.setdefault("generated_at", now())
    payload["report_path"] = rel(p)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("REPORT_PATH=" + rel(p))


def insert_materialization_command(
    *,
    label: str,
    evidence_ref: str,
    artifact_ref: str,
    authority_class: str,
    rationale: str,
) -> str:
    """Insert a materialization-authorized command envelope into conversation_command.

    Returns the new command_uuid (text).
    Raises RuntimeError on DB failure.
    """
    try:
        import psycopg2
    except ImportError:
        raise RuntimeError("psycopg2 not available")

    idem = sha256_str(f"graph_promotion_full_e2e:{label}:{rationale}")[:32]
    envelope = {
        "allowed_effect": "graph materialization via operator_confirmed authority",
        "graph_materialization_policy": "graph_promoter_transaction",
        "staging_only": False,
        "evidence_ref": evidence_ref,
        "artifact_ref": artifact_ref,
        "authority_class": authority_class,
        "rationale": rationale,
        "source": "graph_promotion_full_e2e",
        "target_refs": [f"graph_item:{label.replace(' ', '_').lower()[:32]}"],
    }
    instruction = (
        f"Operator authorized graph materialization: {label}. "
        f"Authority: {authority_class}. Rationale: {rationale}."
    )

    conn = psycopg2.connect(_STATE_DSN)
    try:
        with conn, conn.cursor() as cur:
            target_ref = f"graph_item:{label.replace(' ', '_').lower()[:32]}"
            cur.execute("""
                INSERT INTO lucidota_control.conversation_command
                  (plain_language_instruction, command_envelope, source_surface_id,
                   source_artifact_refs, target_refs, evidence_refs, allowed_effect, authority_class,
                   canonical_mutation_allowed, conversation_required, status, idempotency_key)
                VALUES (%s, %s, %s, %s, %s, %s,
                        'graph materialization via operator_confirmed authority',
                        %s, false, true, 'accepted', %s)
                ON CONFLICT (idempotency_key) DO UPDATE
                  SET status = 'accepted',
                      command_envelope = EXCLUDED.command_envelope,
                      target_refs = EXCLUDED.target_refs,
                      updated_at = now()
                RETURNING command_uuid::text
            """, (
                instruction,
                json.dumps(envelope),
                "graph_promotion_full_e2e",
                json.dumps([artifact_ref]),
                json.dumps([target_ref]),
                json.dumps([evidence_ref]),
                authority_class,
                idem,
            ))
            row = cur.fetchone()
        conn.close()
    except Exception as e:
        conn.close()
        raise RuntimeError(f"command insert failed: {e}") from e

    if not row:
        raise RuntimeError("command insert returned no row")
    return row[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Operator-confirmed graph promotion E2E. Writes to canonical graph."
    )
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--label", default="Full E2E graph promotion proof")
    ap.add_argument("--evidence-ref", default="05_OUTPUTS/work_loops/real_code_loop_ledger.jsonl")
    ap.add_argument("--artifact-ref", default="scripts/graph_promotion_full_e2e.py")
    ap.add_argument("--authority-class", default="operator_confirmed_finding")
    ap.add_argument("--rationale", default="Operator-confirmed E2E materialization proof")
    a = ap.parse_args()

    if not a.execute:
        write({"action": "graph_promotion_full_e2e", "execute_performed": False,
               "blockers": ["execute_required"], "status": "FAIL"})
        return 2

    for dep in ("scripts/graph_materialization_helper.py",
                "scripts/graph_materialization_receipt_query.py"):
        if not (ROOT / dep).exists():
            write({"action": "graph_promotion_full_e2e", "execute_performed": False,
                   "blockers": [f"missing_dep:{dep}"], "status": "FAIL"})
            return 2

    blockers: list[str] = []
    steps: list[dict] = []

    # Step 1: insert materialization-authorized command envelope directly
    cmd_uuid: str | None = None
    try:
        cmd_uuid = insert_materialization_command(
            label=a.label,
            evidence_ref=a.evidence_ref,
            artifact_ref=a.artifact_ref,
            authority_class=a.authority_class,
            rationale=a.rationale,
        )
        steps.append({"step": "insert_command_envelope", "rc": 0, "command_uuid": cmd_uuid})
        print(f"COMMAND_UUID={cmd_uuid}")
    except Exception as e:
        blockers.append(f"command_insert_failed:{e}")
        steps.append({"step": "insert_command_envelope", "rc": 1, "error": str(e)})

    # Step 2: materialize via helper
    mat: dict = {}
    if cmd_uuid:
        candidate = json.dumps({
            "term": "ENTITY", "label": a.label, "status": "staged",
            "detail": {"source": "scripts/graph_promotion_full_e2e.py", "rationale": a.rationale},
        })
        mat = runc([
            sys.executable, "scripts/graph_materialization_helper.py", "materialize",
            "--execute", "--operator-confirmed", "--promotion-only",
            "--command-envelope-uuid", cmd_uuid,
            "--candidate-payload-json", candidate,
            "--evidence-ref", a.evidence_ref,
            "--source-system", "graph_promotion_full_e2e",
            "--authority-class", a.authority_class,
            "--rationale", a.rationale,
        ])
        steps.append(mat)
        if mat["rc"] != 0:
            blockers.append("materialize_failed")

    # Step 3: query receipt
    if mat.get("MATERIALIZATION_UUID"):
        q = runc([sys.executable, "scripts/graph_materialization_receipt_query.py",
                  "--materialization-uuid", mat["MATERIALIZATION_UUID"]])
        steps.append(q)
        if q["rc"] != 0:
            blockers.append("receipt_query_failed")

    payload = {
        "action": "graph_promotion_full_e2e",
        "execute_performed": True,
        "db_writes_performed": bool(cmd_uuid),
        "canonical_graph_writes_performed": bool(mat.get("GRAPH_ITEM_UUID")),
        "steps": steps,
        "command_uuid": cmd_uuid,
        "materialization_uuid": mat.get("MATERIALIZATION_UUID"),
        "journal_uuid": mat.get("JOURNAL_UUID"),
        "blockers": blockers,
        "status": "PASS" if not blockers else "FAIL",
    }
    write(payload)
    print("GRAPH_PROMOTION_FULL_E2E=" + payload["status"])
    return 0 if not blockers else 4


if __name__ == "__main__":
    raise SystemExit(main())
