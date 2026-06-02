import argparse
import json
import os
import subprocess
import sys
import time
import shlex
import requests
import psycopg
from pathlib import Path
from datetime import datetime, timezone

def create_schema_and_table():
    conn = psycopg.connect(os.environ.get('LUCIDOTA_GO_STATE_DSN', 'postgresql:///lucidota_state'))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE SCHEMA IF NOT EXISTS lucidota_model;
        CREATE TABLE IF NOT EXISTS lucidota_model.model_startup_receipt (
            id SERIAL PRIMARY KEY,
            model_name TEXT,
            instance_id TEXT,
            slots INT,
            startup_started_at TIMESTAMPTZ,
            health_passed_at TIMESTAMPTZ,
            peak_ram_mb FLOAT,
            peak_vram_mb FLOAT,
            steady_ram_mb FLOAT,
            steady_vram_mb FLOAT,
            oom_killed BOOLEAN DEFAULT FALSE,
            accepted BOOLEAN,
            rejection_reason TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def get_available_ram():
    with open('/proc/meminfo', 'r') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1024  # Convert KB to MB

def get_available_vram():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return int(result.stdout.strip())

def admit_next(cmd: list, port: int, model_name: str, slots: int = 1) -> dict:
    available_ram = get_available_ram()
    available_vram = get_available_vram()

    if available_ram < 1500 or available_vram < 400:
        return {'admitted': False, 'reason': 'insufficient headroom'}

    process = subprocess.Popen(cmd)
    instance_id = str(process.pid)
    startup_started_at = datetime.now(timezone.utc)

    peak_ram = 0
    peak_vram = 0
    health_passed = False

    for _ in range(60):  # 120 seconds total
        try:
            response = requests.get(f'http://localhost:{port}/health', timeout=2)
            if response.status_code == 200:
                health_passed = True
                break
        except requests.RequestException:
            pass
        time.sleep(2)

    if not health_passed:
        process.terminate()
        return {'admitted': False, 'reason': 'health_timeout'}

    # Measure steady state
    steady_ram = []
    steady_vram = []
    for _ in range(3):
        steady_ram.append(get_available_ram())
        steady_vram.append(get_available_vram())
        time.sleep(3)

    steady_ram_mb = sum(steady_ram) / len(steady_ram)
    steady_vram_mb = sum(steady_vram) / len(steady_vram)

    conn = psycopg.connect(os.environ.get('LUCIDOTA_GO_STATE_DSN', 'postgresql:///lucidota_state'))
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO lucidota_model.model_startup_receipt (
            model_name, instance_id, slots, startup_started_at, health_passed_at,
            peak_ram_mb, peak_vram_mb, steady_ram_mb, steady_vram_mb, accepted
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        model_name, instance_id, slots, startup_started_at, datetime.now(timezone.utc),
        peak_ram, peak_vram, steady_ram_mb, steady_vram_mb, True
    ))
    conn.commit()
    cursor.close()
    conn.close()

    return {'admitted': True, 'peak_ram_mb': peak_ram, 'peak_vram_mb': peak_vram}

def main():
    parser = argparse.ArgumentParser(description='Model admission controller')
    parser.add_argument('--cmd', required=True, help='Command to start the model')
    parser.add_argument('--port', type=int, required=True, help='Port to check health on')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--slots', type=int, default=1, help='Number of slots')
    args = parser.parse_args()

    create_schema_and_table()
    result = admit_next(shlex.split(args.cmd), args.port, args.model, args.slots)
    print(json.dumps(result))

if __name__ == '__main__':
    main()
