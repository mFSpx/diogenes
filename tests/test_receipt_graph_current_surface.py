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


def test_receipt_graph_current_surface_shows_indy_lane() -> None:
    assert psql(
        """
        select count(*)::text
        from lucidota_audit.visible_status_layer
        where worker = 'indy_reads_runtime'
        """
    ) == "2"
    latest = psql(
        """
        select worker || '|' || work_order || '|' || model_identifier || '|' || proof_status || '|' || receipt_uuid
        from lucidota_audit.visible_status_layer
        where worker = 'indy_reads_runtime'
        order by timestamp desc
        limit 1
        """
    )
    assert "indy_reads_runtime" in latest
    assert "bonsai_q1_0" in latest
    assert "46de582f-f35e-5562-a189-92652e562e73" in latest
    assert psql(
        """
        select count(*)::text
        from lucidota_audit.workload_audit_ledger
        where worker_id = 'indy_reads_runtime'
          and model_identifier_uuid = 'b0f0a0b0-0000-4000-8000-000000000001'
        """
    ) == "2"
