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


def test_indy_reads_self_model_schema_exposes_identity_fields() -> None:
    sql = """
        SELECT string_agg(column_name, ',' ORDER BY column_name)
        FROM information_schema.columns
        WHERE table_schema = 'lucidota_canon'
          AND table_name = 'indy_reads_self_model'
    """
    columns = set(filter(None, psql_scalar(sql).split(",")))
    expected = {
        "actor_id",
        "author",
        "boundaries",
        "confidence",
        "created_at",
        "db_refs",
        "evidence_refs",
        "functionality_explanation",
        "goals_refs",
        "next_upgrade",
        "ontology_index",
        "proof_status",
        "role",
        "self_model_id",
        "voice",
    }
    assert expected.issubset(columns), columns
