#!/usr/bin/env python3
"""
lucidota_guvna.py — Governor Phase-1 daemon
Reads live kernel telemetry and decides pressure-rung action.

Rungs (NEVER-CULL ladder):
  rung0  — nominal, no action
  rung1  — high pressure: lower new-work concurrency (emit advisory + DB fact)
  rung2  — critical pressure: raise MemoryHigh threshold (emit advisory)
  rung3  — severe pressure: shrink BGE fleet via lucidota_bge_fleet.sh COUNT

--dry-run mode: decides + logs, NEVER actuates.
--once: single-shot (original phase-0 compat).
Daemon mode (default): loops every --interval seconds, logs to governor_action table.

Receipt written to: 05_OUTPUTS/runtime/guvna_decision_<ts>.json
DB log: lucidota_control.governor_action (lucidota_state)
Rung fact: lucidota_control.runtime_status_fact governor_rung (lucidota_state)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import psycopg2
    _HAS_PG = True
except ImportError:
    _HAS_PG = False

_STATE_DSN = os.environ.get("LUCIDOTA_GO_STATE_DSN", "postgresql:///lucidota_state")

ROOT = Path(os.environ.get("LUCIDOTA_HOME", "/home/mfspx/LUCIDOTA"))
OUTPUT_DIR = ROOT / "05_OUTPUTS" / "runtime"
FLEET_SCRIPT = ROOT / "scripts" / "lucidota_bge_fleet.sh"
FLEET_PID_GLOB = "04_RUNTIME/inference_os/bge_fleet_*.pid"

# ── Thresholds ─────────────────────────────────────────────────────────────────
# PSI "some avg10" values (percent, 0-100)
CPU_PSI_RUNG1  = 20.0   # rung1 when CPU pressure exceeds this
CPU_PSI_RUNG2  = 50.0
CPU_PSI_RUNG3  = 75.0

MEM_PSI_RUNG1  = 5.0    # rung1 when MEM pressure exceeds this
MEM_PSI_RUNG2  = 15.0
MEM_PSI_RUNG3  = 30.0

IO_PSI_RUNG1   = 10.0   # rung1 when IO pressure exceeds this
IO_PSI_RUNG2   = 25.0
IO_PSI_RUNG3   = 60.0

MEM_AVAIL_RUNG1_MB = 2000  # rung1 when MemAvailable < this
MEM_AVAIL_RUNG2_MB = 1200
MEM_AVAIL_RUNG3_MB = 700

# VRAM thresholds (MiB)
VRAM_RUNG1_PCT  = 0.80   # rung1 when used/total > 80%
VRAM_RUNG2_PCT  = 0.90
VRAM_RUNG3_PCT  = 0.95

# Fleet shrink targets per rung3
FLEET_RUNG3_TARGET = 1


def parse_psi(path: str) -> dict:
    """Parse a /proc/pressure/{cpu,memory,io} file. Returns {some_avg10, some_avg60, ...}."""
    result = {}
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                kind = parts[0]  # "some" or "full"
                for token in parts[1:]:
                    if "=" in token:
                        k, v = token.split("=", 1)
                        try:
                            result[f"{kind}_{k.replace('.', '_')}"] = float(v)
                        except ValueError:
                            pass
    except FileNotFoundError:
        pass
    return result


def parse_meminfo() -> int:
    """Return MemAvailable in MiB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb // 1024
    except Exception:
        pass
    return -1


def parse_vram() -> tuple[int, int]:
    """Return (used_mib, total_mib) or (-1, -1) if nvidia-smi unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            timeout=5, stderr=subprocess.DEVNULL
        ).decode().strip()
        # "2 MiB, 4096 MiB"
        parts = [p.strip().split()[0] for p in out.split(",")]
        return int(parts[0]), int(parts[1])
    except Exception:
        return -1, -1


def count_bge_fleet() -> int:
    """Count live BGE fleet PID files."""
    fleet_dir = ROOT / "04_RUNTIME" / "inference_os"
    if not fleet_dir.exists():
        return 0
    return len(list(fleet_dir.glob("bge_fleet_*.pid")))


def compute_rung(cpu_psi: dict, mem_psi: dict, io_psi: dict,
                 mem_avail_mb: int, vram_used: int, vram_total: int) -> tuple[int, list[str]]:
    """Return (rung, reasons[]) — highest rung wins."""
    rung = 0
    reasons = []

    cpu_some10 = cpu_psi.get("some_avg10", 0.0)
    mem_some10 = mem_psi.get("some_avg10", 0.0)
    io_some10  = io_psi.get("some_avg10", 0.0)

    # CPU pressure
    if cpu_some10 >= CPU_PSI_RUNG3:
        rung = max(rung, 3); reasons.append(f"cpu_psi_some_avg10={cpu_some10:.2f}>={CPU_PSI_RUNG3}")
    elif cpu_some10 >= CPU_PSI_RUNG2:
        rung = max(rung, 2); reasons.append(f"cpu_psi_some_avg10={cpu_some10:.2f}>={CPU_PSI_RUNG2}")
    elif cpu_some10 >= CPU_PSI_RUNG1:
        rung = max(rung, 1); reasons.append(f"cpu_psi_some_avg10={cpu_some10:.2f}>={CPU_PSI_RUNG1}")

    # Memory pressure
    if mem_some10 >= MEM_PSI_RUNG3:
        rung = max(rung, 3); reasons.append(f"mem_psi_some_avg10={mem_some10:.2f}>={MEM_PSI_RUNG3}")
    elif mem_some10 >= MEM_PSI_RUNG2:
        rung = max(rung, 2); reasons.append(f"mem_psi_some_avg10={mem_some10:.2f}>={MEM_PSI_RUNG2}")
    elif mem_some10 >= MEM_PSI_RUNG1:
        rung = max(rung, 1); reasons.append(f"mem_psi_some_avg10={mem_some10:.2f}>={MEM_PSI_RUNG1}")

    # IO pressure
    if io_some10 >= IO_PSI_RUNG3:
        rung = max(rung, 3); reasons.append(f"io_psi_some_avg10={io_some10:.2f}>={IO_PSI_RUNG3}")
    elif io_some10 >= IO_PSI_RUNG2:
        rung = max(rung, 2); reasons.append(f"io_psi_some_avg10={io_some10:.2f}>={IO_PSI_RUNG2}")
    elif io_some10 >= IO_PSI_RUNG1:
        rung = max(rung, 1); reasons.append(f"io_psi_some_avg10={io_some10:.2f}>={IO_PSI_RUNG1}")

    # MemAvailable
    if mem_avail_mb >= 0:
        if mem_avail_mb < MEM_AVAIL_RUNG3_MB:
            rung = max(rung, 3); reasons.append(f"mem_avail={mem_avail_mb}MB<{MEM_AVAIL_RUNG3_MB}MB")
        elif mem_avail_mb < MEM_AVAIL_RUNG2_MB:
            rung = max(rung, 2); reasons.append(f"mem_avail={mem_avail_mb}MB<{MEM_AVAIL_RUNG2_MB}MB")
        elif mem_avail_mb < MEM_AVAIL_RUNG1_MB:
            rung = max(rung, 1); reasons.append(f"mem_avail={mem_avail_mb}MB<{MEM_AVAIL_RUNG1_MB}MB")

    # VRAM
    if vram_total > 0:
        vram_pct = vram_used / vram_total
        if vram_pct >= VRAM_RUNG3_PCT:
            rung = max(rung, 3); reasons.append(f"vram_pct={vram_pct:.2f}>={VRAM_RUNG3_PCT}")
        elif vram_pct >= VRAM_RUNG2_PCT:
            rung = max(rung, 2); reasons.append(f"vram_pct={vram_pct:.2f}>={VRAM_RUNG2_PCT}")
        elif vram_pct >= VRAM_RUNG1_PCT:
            rung = max(rung, 1); reasons.append(f"vram_pct={vram_pct:.2f}>={VRAM_RUNG1_PCT}")

    if not reasons:
        reasons.append("nominal")

    return rung, reasons


RUNG_DESCRIPTIONS = {
    0: "nominal — no action",
    1: "lower new-work concurrency (advisory)",
    2: "raise MemoryHigh threshold (advisory)",
    3: "shrink BGE fleet to reduce memory pressure",
}


def actuate_rung3(fleet_width: int, dry_run: bool) -> dict:
    """Shrink fleet if rung3 and not dry-run. Returns action_taken dict."""
    target = FLEET_RUNG3_TARGET
    if fleet_width <= target:
        return {"action": "fleet_already_minimal", "fleet_width": fleet_width, "target": target}
    if dry_run:
        return {"action": "DRY_RUN_would_shrink_fleet", "fleet_width": fleet_width, "target": target,
                "command": f"{FLEET_SCRIPT} {target}"}
    # Actuate: call fleet script with target count
    try:
        result = subprocess.run(
            ["bash", str(FLEET_SCRIPT), str(target)],
            capture_output=True, text=True, timeout=30
        )
        return {
            "action": "fleet_shrink_issued",
            "fleet_width_before": fleet_width,
            "target": target,
            "returncode": result.returncode,
            "stdout": result.stdout.strip()[:500],
            "stderr": result.stderr.strip()[:200],
        }
    except Exception as e:
        return {"action": "fleet_shrink_error", "error": str(e)}


def _db_log(decision: dict, receipt_path: str, dry_run: bool) -> None:
    """Write decision to governor_action table and update runtime_status_fact governor_rung."""
    if not _HAS_PG:
        return
    d = decision["decision"]
    t = decision["telemetry"]
    action_key = decision["action"].get("action", "none")
    try:
        conn = psycopg2.connect(_STATE_DSN)
        with conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lucidota_control.governor_action
                  (rung, rung_description, action_key, action_detail, dry_run, telemetry_snapshot, receipt_path)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                d["rung"], d["rung_description"], action_key,
                json.dumps(decision["action"]),
                dry_run, json.dumps(t), receipt_path,
            ))
            # Update runtime_status_fact so workers can see current rung cheaply
            cur.execute("""
                INSERT INTO lucidota_control.runtime_status_fact
                  (subsystem, fact_key, fact_value, derived_at)
                VALUES ('governor', 'governor_rung', %s::jsonb, now())
                ON CONFLICT (subsystem, fact_key) DO UPDATE SET
                  fact_value = EXCLUDED.fact_value, derived_at = now()
            """, (json.dumps(d["rung"]),))
            # Rung-1+: write ABSURD concurrency advisory
            if d["rung"] >= 1 and not dry_run:
                recommended = max(1, 4 - d["rung"])
                cur.execute("""
                    INSERT INTO lucidota_control.runtime_status_fact
                      (subsystem, fact_key, fact_value, derived_at)
                    VALUES ('governor', 'absurd_recommended_max_workers', %s::jsonb, now())
                    ON CONFLICT (subsystem, fact_key) DO UPDATE SET
                      fact_value = EXCLUDED.fact_value, derived_at = now()
                """, (json.dumps(recommended),))
        conn.close()
    except Exception as e:
        print(f"[guvna] DB log failed: {e}", file=sys.stderr)


def run_once(dry_run: bool) -> dict:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Collect telemetry
    cpu_psi = parse_psi("/proc/pressure/cpu")
    mem_psi = parse_psi("/proc/pressure/memory")
    io_psi  = parse_psi("/proc/pressure/io")
    mem_avail_mb = parse_meminfo()
    vram_used, vram_total = parse_vram()
    fleet_width = count_bge_fleet()

    rung, reasons = compute_rung(cpu_psi, mem_psi, io_psi, mem_avail_mb, vram_used, vram_total)

    action_result = {}
    if rung == 3:
        action_result = actuate_rung3(fleet_width, dry_run)
    elif rung in (1, 2):
        action_result = {
            "action": "DRY_RUN_advisory" if dry_run else "advisory",
            "description": RUNG_DESCRIPTIONS[rung],
        }
    else:
        action_result = {"action": "none"}

    decision = {
        "schema": "guvna_decision_v1",
        "ts": ts,
        "dry_run": dry_run,
        "telemetry": {
            "cpu_psi": cpu_psi,
            "mem_psi": mem_psi,
            "io_psi": io_psi,
            "mem_avail_mb": mem_avail_mb,
            "vram_used_mib": vram_used,
            "vram_total_mib": vram_total,
            "bge_fleet_width": fleet_width,
        },
        "decision": {
            "rung": rung,
            "rung_description": RUNG_DESCRIPTIONS[rung],
            "reasons": reasons,
        },
        "action": action_result,
    }

    return decision, ts


def _print_decision(decision: dict, out_path: Path, mode: str) -> None:
    d = decision["decision"]
    t = decision["telemetry"]
    print(f"[guvna {mode}] rung={d['rung']} — {d['rung_description']}")
    print(f"  cpu_psi_some_avg10 : {t['cpu_psi'].get('some_avg10', 'N/A'):.2f}%")
    print(f"  mem_psi_some_avg10 : {t['mem_psi'].get('some_avg10', 'N/A'):.2f}%")
    print(f"  io_psi_some_avg10  : {t['io_psi'].get('some_avg10', 'N/A'):.2f}%")
    print(f"  mem_avail          : {t['mem_avail_mb']} MB")
    print(f"  vram               : {t['vram_used_mib']} / {t['vram_total_mib']} MiB")
    print(f"  bge_fleet_width    : {t['bge_fleet_width']}")
    print(f"  reasons            : {', '.join(d['reasons'])}")
    print(f"  action             : {decision['action']}")
    print(f"  receipt            : {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="LUCIDOTA Governor Phase-1: daemon that reads telemetry, decides pressure rung, logs to DB."
    )
    parser.add_argument("--once", action="store_true", default=False,
                        help="Single-shot run (no daemon loop).")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Decide and log only; never actuate.")
    parser.add_argument("--interval", type=int, default=30,
                        help="Daemon loop interval in seconds (default 30).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = "DRY-RUN" if args.dry_run else "LIVE"

    if args.once:
        decision, ts = run_once(dry_run=args.dry_run)
        out_path = OUTPUT_DIR / f"guvna_decision_{ts}.json"
        out_path.write_text(json.dumps(decision, indent=2))
        _db_log(decision, str(out_path), args.dry_run)
        _print_decision(decision, out_path, mode)
        return 0

    # Daemon loop
    print(f"[guvna] daemon started, interval={args.interval}s, mode={mode}", flush=True)
    tick = 0
    while True:
        try:
            decision, ts = run_once(dry_run=args.dry_run)
            out_path = OUTPUT_DIR / f"guvna_decision_{ts}.json"
            out_path.write_text(json.dumps(decision, indent=2))
            _db_log(decision, str(out_path), args.dry_run)
            d = decision["decision"]
            if d["rung"] > 0 or tick % 10 == 0:
                _print_decision(decision, out_path, mode)
            else:
                print(f"[guvna {mode}] tick={tick} rung=0 avail={decision['telemetry']['mem_avail_mb']}MB", flush=True)
        except Exception as e:
            print(f"[guvna] tick error: {e}", file=sys.stderr, flush=True)
        tick += 1
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    sys.exit(main())
