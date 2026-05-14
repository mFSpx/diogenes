#!/usr/bin/env python3
"""Tiny River online-learning bridge for committed LUCIDOTA workflow events.

This is intentionally small: DBOS/Postgres commits events; this script learns
success/failure reflex hints with River and writes bounded scores back to Postgres.
It is not runtime authority. DBOS/governance remains the decision gate.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import psycopg
from river import compose, linear_model, preprocessing

DEFAULT_STATE_DB = "postgresql://mfspx@/lucidota_state"


@dataclass(frozen=True)
class Event:
    event_id: str
    created_at: str
    source: str
    phase: str
    status: str
    decision: str


def ensure_schema(conn: psycopg.Connection) -> None:
    schema = (os.environ.get("LUCIDOTA_HOME") or os.getcwd()) + "/06_SCHEMA/004_learning_reflex.sql"
    with open(schema, "r", encoding="utf-8") as fh:
        conn.execute(fh.read())
    conn.commit()


def load_events(conn: psycopg.Connection, limit: int) -> list[Event]:
    rows = conn.execute(
        """
        SELECT event_id::text, created_at::text, source, phase, status,
               COALESCE(detail->>'decision', detail->>'survey_decision', '') AS decision
        FROM lucidota_control.workflow_event
        WHERE status IN ('succeeded', 'failed')
        ORDER BY created_at ASC, event_id ASC
        LIMIT %s
        """,
        (limit,),
    ).fetchall()
    return [Event(*row) for row in rows]


def features(e: Event) -> dict[str, object]:
    return {
        "source": e.source,
        "phase": e.phase,
        "decision": e.decision or "none",
        "is_survey": int(e.source == "lucidota_survey"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(prog="lucidota-river-reflex")
    ap.add_argument("--db-url", default=os.environ.get("DBOS_SYSTEM_DATABASE_URL", DEFAULT_STATE_DB))
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    model = compose.Pipeline(
        preprocessing.OneHotEncoder(),
        linear_model.LogisticRegression(),
    )
    grouped: dict[tuple[str, str, str], dict[str, int | float | None]] = defaultdict(
        lambda: {"examples": 0, "successes": 0, "failures": 0, "prediction": None}
    )

    with psycopg.connect(args.db_url) as conn:
        ensure_schema(conn)
        events = load_events(conn, args.limit)
        for e in events:
            x = features(e)
            y = e.status == "succeeded"
            pred = model.predict_proba_one(x).get(True, 0.5)
            model.learn_one(x, y)
            key = (e.source, e.phase, e.decision or "")
            bucket = grouped[key]
            bucket["examples"] = int(bucket["examples"]) + 1
            bucket["successes"] = int(bucket["successes"]) + int(y)
            bucket["failures"] = int(bucket["failures"]) + int(not y)
            bucket["prediction"] = float(pred)

        for (source, phase, decision), bucket in grouped.items():
            examples = int(bucket["examples"])
            successes = int(bucket["successes"])
            failures = int(bucket["failures"])
            rate = successes / examples if examples else 0.0
            conn.execute(
                """
                INSERT INTO lucidota_learning.river_score
                  (source, phase, decision, examples, successes, failures, success_rate, river_prediction, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,now())
                ON CONFLICT (source, phase, decision) DO UPDATE SET
                  examples = EXCLUDED.examples,
                  successes = EXCLUDED.successes,
                  failures = EXCLUDED.failures,
                  success_rate = EXCLUDED.success_rate,
                  river_prediction = EXCLUDED.river_prediction,
                  updated_at = now()
                """,
                (source, phase, decision, examples, successes, failures, rate, bucket["prediction"]),
            )
        conn.execute(
            """
            INSERT INTO lucidota_learning.river_run (status, events_seen, examples_trained, detail)
            VALUES ('succeeded', %s, %s, %s::jsonb)
            """,
            (len(events), len(events), json.dumps({"groups": len(grouped)})),
        )
        conn.commit()

    report = {"ok": True, "events_seen": len(events), "groups": len(grouped)}
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
