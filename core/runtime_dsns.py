"""Shared DSN resolution helpers for LUCIDOTA runtime boundaries.

These helpers intentionally avoid ABSURD-specific environment names so the
current Absurd / Postgres / SQLite layering stays explicit and portable.
"""
from __future__ import annotations

import os


def resolve_state_dsn(default: str = "postgresql:///lucidota_state") -> str:
    return (
        os.environ.get("LUCIDOTA_ABSURD_STATE_DSN")
        or os.environ.get("LUCIDOTA_GO_STATE_DSN")
        or os.environ.get("DATABASE_URL")
        or default
    )


def resolve_storage_dsn(default: str = "postgresql:///lucidota_storage") -> str:
    return (
        os.environ.get("LUCIDOTA_GO_STORAGE_DSN")
        or os.environ.get("KORPUS_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
        or default
    )


def resolve_sqlite_path(default: str = "") -> str:
    return os.environ.get("LUCIDOTA_SQLITE_PATH") or default
