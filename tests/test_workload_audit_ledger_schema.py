from __future__ import annotations

import subprocess


ROOT_DSN = "postgresql:///lucidota_state"


def psql_scalar(sql: str) -> str:
    proc = subprocess.run(
        ["psql", ROOT_DSN, "-Atc", sql],
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def test_workload_audit_ledger_schema_exposes_required_columns() -> None:
    expected = {
        "actor_id",
        "actor_class",
        "caller",
        "provider",
        "model_id",
        "action_summary",
        "tokens_in",
        "tokens_out",
        "token_source",
        "receipt_uuid",
        "evidence_refs",
        "proof_status",
        "debt_reason",
        "created_at",
        "refreshed_at",
        "functionality_explanation",
        "ontology_index",
    }

    sql = """
        SELECT string_agg(column_name, ',' ORDER BY column_name)
        FROM information_schema.columns
        WHERE table_schema = 'lucidota_canon'
          AND table_name = 'workload_audit_ledger'
    """
    columns = set(filter(None, psql_scalar(sql).split(",")))
    assert expected.issubset(columns), columns


def test_workload_audit_current_schema_exists() -> None:
    sql = """
        SELECT count(*)
        FROM information_schema.columns
        WHERE table_schema = 'lucidota_canon'
          AND table_name = 'workload_audit_current'
    """
    assert int(psql_scalar(sql) or "0") > 0
