#!/usr/bin/env python3
"""Create the Bytewax logical replication slot in lucidota_state if it does not exist."""
import os
import sys
import argparse
import psycopg2

SLOT_NAME = "lucidota_bytewax_abductive_slot"
PLUGIN = "pgoutput"


def main() -> None:
    parser = argparse.ArgumentParser(description="Init Bytewax logical replication slot")
    parser.add_argument("--dry-run", action="store_true", help="Check existence only, do not create")
    args = parser.parse_args()

    dsn = os.environ.get("LUCIDOTA_GO_STATE_DSN", "postgresql:///lucidota_state")
    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute(
            "SELECT slot_name, plugin, active FROM pg_replication_slots WHERE slot_name = %s",
            (SLOT_NAME,),
        )
        row = cur.fetchone()

        if row:
            print(f"Slot already exists: name={row[0]} plugin={row[1]} active={row[2]}")
            sys.exit(0)

        if args.dry_run:
            print(f"DRY RUN: slot '{SLOT_NAME}' does not exist — would create with plugin '{PLUGIN}'")
            sys.exit(0)

        cur.execute(
            "SELECT pg_create_logical_replication_slot(%s, %s)",
            (SLOT_NAME, PLUGIN),
        )
        result = cur.fetchone()
        print(f"Created: {result}")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
