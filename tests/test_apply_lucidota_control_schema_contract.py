from pathlib import Path


def test_apply_schema_sources_canonical_pg_env():
    script = Path("scripts/apply_lucidota_control_schema.sh").read_text()
    assert 'source "$ROOT/scripts/lucidota_pg_user_env.sh"' in script
    assert 'psql "$DB_URL"' in script
