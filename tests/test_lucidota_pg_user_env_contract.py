from pathlib import Path


def test_lucidota_pg_user_env_targets_system_postgres():
    text = Path("scripts/lucidota_pg_user_env.sh").read_text(encoding="utf-8")
    assert 'export PGHOST="/var/run/postgresql"' in text
    assert 'export PGPORT="5432"' in text
    assert 'export PGDATABASE="lucidota_state"' in text
    assert 'export DATABASE_URL="postgresql:///lucidota_state"' in text
    assert 'export LUCIDOTA_GO_STORAGE_DSN="postgresql:///lucidota_storage"' in text
