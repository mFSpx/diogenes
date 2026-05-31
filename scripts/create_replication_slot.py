import os
import sys
import argparse
import psycopg2

SLOT_NAME = "lucidota_bytewax_abductive_slot"
PLUGIN = "test_decoding"


def main():
    parser = argparse.ArgumentParser(description="Create Bytewax logical replication slot")
    parser.add_argument("--dry-run", action="store_true", help="Check existence without creating")
    args = parser.parse_args()

    dsn = os.environ.get("LUCIDOTA_GO_STATE_DSN", "postgresql:///lucidota_state")
    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = True
        cur = conn.cursor()

        cur.execute("SELECT slot_name FROM pg_replication_slots WHERE slot_name = %s", (SLOT_NAME,))
        exists = cur.fetchone()

        if exists:
            print(f"Slot '{SLOT_NAME}' already exists.")
            sys.exit(0)

        if args.dry_run:
            print(f"DRY RUN: slot '{SLOT_NAME}' does not exist — would create with plugin '{PLUGIN}'")
            sys.exit(0)

        cur.execute("SELECT pg_create_logical_replication_slot(%s, %s)", (SLOT_NAME, PLUGIN))
        row = cur.fetchone()
        print(f"Created slot: {row}")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
