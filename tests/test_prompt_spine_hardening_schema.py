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


def test_prompt_record_enforces_link_or_ambient_null_reason() -> None:
    sql = """
        SELECT pg_get_constraintdef(c.oid)
        FROM pg_constraint c
        WHERE c.conrelid = 'lucidota_control.prompt_record'::regclass
          AND c.conname = 'enforce_attribution_invariant'
    """
    constraint = psql_scalar(sql)
    assert "cardinality(linked_work_order_uuid) > 0" in constraint
    assert "unlinked_reason = 'ambient/daemon/probe'" in constraint


def test_prompt_surface_exposes_work_order_uuid_and_null_reason() -> None:
    sql = """
        SELECT string_agg(column_name, ',' ORDER BY column_name)
        FROM information_schema.columns
        WHERE table_schema = 'lucidota_canon'
          AND table_name = 'prompts_filed'
    """
    columns = set(filter(None, psql_scalar(sql).split(",")))
    assert {"work_order_uuid", "null_reason"}.issubset(columns), columns


def test_model_invocation_receipt_exposes_null_reason() -> None:
    sql = """
        SELECT string_agg(column_name, ',' ORDER BY column_name)
        FROM information_schema.columns
        WHERE table_schema = 'lucidota_canon'
          AND table_name = 'model_invocation_receipt'
    """
    columns = set(filter(None, psql_scalar(sql).split(",")))
    assert "null_reason" in columns, columns


def test_prompt_record_unlinked_rows_use_the_ambient_escape_hatch() -> None:
    sql = """
        SELECT count(*)
        FROM lucidota_control.prompt_record
        WHERE cardinality(linked_work_order_uuid) = 0
          AND coalesce(unlinked_reason, '') <> 'ambient/daemon/probe'
    """
    assert int(psql_scalar(sql) or "0") == 0

