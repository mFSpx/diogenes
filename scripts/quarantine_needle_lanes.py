#!/usr/bin/env python3
import os
import psycopg2

def main():
    dsn = os.environ.get('LUCIDOTA_GO_STATE_DSN', 'postgresql:///lucidota_state')
    conn = psycopg2.connect(dsn)
    cur = conn.cursor()
    cur.execute("""
        UPDATE lucidota_control.runtime_status_fact 
        SET fact_value = fact_value || jsonb_build_object('status','offline_quarantine','reason','JAX checkpoints not compiled to GGUF')
        WHERE fact_key LIKE '%needle%'
    """)
    rowcount = cur.rowcount
    if rowcount == 0:
        print("No needle facts matched.")
    else:
        print(rowcount)
    conn.commit()
    cur.close()
    conn.close()

if __name__ == '__main__':
    main()
