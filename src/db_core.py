from __future__ import annotations

import contextlib
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterator, TypeVar

import psycopg2
import psycopg2.errors
from psycopg2.extensions import connection as PGConnection
from psycopg2.extensions import cursor as PGCursor

T = TypeVar("T")
DEFAULT_DSN = os.environ.get("LUCIDOTA_GO_STORAGE_DSN") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_storage"


@dataclass
class RetryConfig:
    attempts: int = 5
    base_seconds: float = 0.25
    jitter_seconds: float = 1.0


class DurableConnection:
    def __init__(self, dsn: str = DEFAULT_DSN, retry: RetryConfig | None = None) -> None:
        self.dsn = dsn
        self.retry = retry or RetryConfig()
        self.conn: PGConnection | None = None

    def __enter__(self) -> "DurableConnection":
        self.conn = psycopg2.connect(self.dsn)
        self.conn.autocommit = False
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.conn is None:
            return
        if exc_type is not None:
            self.conn.rollback()
        self.conn.close()
        self.conn = None

    def require_conn(self) -> PGConnection:
        if self.conn is None:
            raise RuntimeError("connection is not open")
        return self.conn

    @contextlib.contextmanager
    def cursor(self) -> Iterator[PGCursor]:
        conn = self.require_conn()
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    @contextlib.contextmanager
    def file_savepoint(self, cur: PGCursor) -> Iterator[None]:
        cur.execute("SAVEPOINT file_insert_sp")
        try:
            yield
        except Exception:
            cur.execute("ROLLBACK TO SAVEPOINT file_insert_sp")
            cur.execute("RELEASE SAVEPOINT file_insert_sp")
            raise
        else:
            cur.execute("RELEASE SAVEPOINT file_insert_sp")

    def run_serialized(self, fn: Callable[[PGConnection], T]) -> T:
        conn = self.require_conn()
        last_error: BaseException | None = None
        for attempt in range(self.retry.attempts):
            try:
                result = fn(conn)
                conn.commit()
                return result
            except psycopg2.errors.SerializationFailure as exc:
                conn.rollback()
                last_error = exc
                time.sleep((2 ** attempt) + random.uniform(0, 1))
            except psycopg2.errors.DeadlockDetected as exc:
                conn.rollback()
                last_error = exc
                time.sleep((2 ** attempt) + random.uniform(0, 1))
        raise RuntimeError(f"transaction retry budget exhausted: {last_error}")
