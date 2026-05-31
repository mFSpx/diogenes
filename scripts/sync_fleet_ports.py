#!/usr/bin/env python3
import os
import socket
import psycopg2

PORT_MAP = {
    8080: "llama_primary",
    8081: "mamba_ram",
    8082: "bonsai_ternary",
    8083: "mamba_gpu",
    8101: "bge_fleet",
}

DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", "postgresql:///lucidota_state")

conn = psycopg2.connect(DSN)
cursor = conn.cursor()

for port, lane in PORT_MAP.items():
    status = "ready" if socket.connect_ex(("127.0.0.1", port)) == 0 else "offline"
    cursor.execute(
        "UPDATE lucidota_control.runtime_status_fact "
        "SET fact_value = fact_value || jsonb_build_object(%s, %s) "
        "WHERE fact_key = %s",
        ("port_status", status, lane),
    )
    print(f"{lane}:{port} -> {status}")

conn.commit()
cursor.close()
conn.close()
