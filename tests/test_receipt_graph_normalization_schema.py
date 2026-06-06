from __future__ import annotations

import subprocess


def psql(sql: str) -> str:
    proc = subprocess.run(
        [
            "psql",
            "postgresql:///lucidota_state",
            "-v",
            "ON_ERROR_STOP=1",
            "-P",
            "pager=off",
            "-At",
            "-c",
            sql,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def test_receipt_graph_normalization_schema_defines_spine_and_visible_status_layer() -> None:
    assert psql(
        """
        select count(*)::text
        from information_schema.tables
        where table_schema = 'lucidota_canon'
          and table_name = 'model_identifier'
        """
    ) == "1"
    assert psql(
        """
        select count(*)::text
        from information_schema.columns
        where table_schema = 'lucidota_control'
          and table_name = 'prompt_record'
          and column_name = 'unlinked_reason'
        """
    ) == "1"
    assert psql(
        """
        select count(*)::text
        from information_schema.tables
        where table_schema = 'lucidota_control'
          and table_name = 'worker'
        """
    ) == "1"
    assert psql(
        """
        select count(*)::text
        from information_schema.tables
        where table_schema = 'lucidota_control'
          and table_name = 'work_order_attempt'
        """
    ) == "1"
    assert psql(
        """
        select count(*)::text
        from information_schema.columns
        where table_schema = 'lucidota_audit'
          and table_name = 'workload_audit_ledger'
          and column_name in ('model_identifier_uuid', 'work_order_uuid', 'work_order_attempt_uuid', 'worker_id')
        """
    ) == "4"
    assert psql(
        """
        select count(*)::text
        from information_schema.columns
        where table_schema = 'lucidota_canon'
        and table_name = 'model_invocation_receipt'
        and column_name in ('model_identifier_uuid', 'work_order_uuid', 'work_order_attempt_uuid', 'worker_id')
        """
    ) == "4"
    assert psql(
        """
        select count(*)::text
        from information_schema.columns
        where table_schema = 'lucidota_canon'
          and table_name = 'provider_call_receipt'
          and column_name in ('model_identifier_uuid', 'work_order_uuid', 'work_order_attempt_uuid', 'worker_id')
        """
    ) == "4"
    assert psql(
        """
        select count(*)::text
        from information_schema.views
        where table_schema = 'lucidota_audit'
          and table_name = 'visible_status_layer'
        """
    ) == "1"
    assert psql(
        """
        select count(*)::text
        from lucidota_canon.prompts_filed
        where prompt_id = '355bc98f-f65d-4dc0-9fb9-319cdcfb819a'
        """
    ) == "0"
    assert psql(
        """
        select count(*)::text
        from lucidota_control.prompt_record
        where cardinality(linked_work_order_uuid) = 0
          and btrim(coalesce(unlinked_reason, blockers, notes, '')) = ''
        """
    ) == "0"
