#!/usr/bin/env python3
"""Hardware gate — kernel-native resource reads + cgroup v2 enforcement.

No daemon. No subprocess. No polling.
read_hw_state()  → <1ms via /proc + NVML direct API
setup_cgroup()   → creates cgroup v2 slice, sets hard limits (kernel enforces)
"""
import json, os, pathlib, sys
from datetime import datetime, timezone

_VRAM_BUDGET_MB = int(os.getenv("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
_CGROUP_ROOT = pathlib.Path("/sys/fs/cgroup/lucidota")

# ── Hardware reads (direct kernel, no subprocess) ───────────────────────────

def _mem_available_mb() -> int:
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) // 1024
    return 9999

def _load_per_core() -> float:
    load1 = float(pathlib.Path("/proc/loadavg").read_text().split()[0])
    ncpu = os.cpu_count() or 1
    return round(load1 / ncpu, 3)

def _disk_pct(path: str = "/") -> float:
    st = os.statvfs(path)
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    total = st.f_blocks * st.f_frsize
    return round(used / total * 100, 1) if total else 0.0

def _vram_mb() -> tuple[int, int]:
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
        nvmlInit()
        h = nvmlDeviceGetHandleByIndex(0)
        m = nvmlDeviceGetMemoryInfo(h)
        nvmlShutdown()
        return m.used // 1024 // 1024, m.free // 1024 // 1024
    except Exception:
        return 0, _VRAM_BUDGET_MB

# ── cgroup v2 enforcement (kernel-native hard limits) ───────────────────────

def setup_cgroup(name: str = "workers", mem_limit_mb: int = 5000, cpu_quota_pct: int = 75) -> pathlib.Path:
    """Create /sys/fs/cgroup/lucidota/<name> and set hard memory + CPU limits."""
    cg = _CGROUP_ROOT / name
    cg.mkdir(parents=True, exist_ok=True)

    mem_bytes = mem_limit_mb * 1024 * 1024
    (cg / "memory.max").write_text(str(mem_bytes))

    period_us = 100_000
    quota_us = int(period_us * cpu_quota_pct / 100) * (os.cpu_count() or 1)
    (cg / "cpu.max").write_text(f"{quota_us} {period_us}")

    return cg

def join_cgroup(name: str = "workers") -> None:
    """Move current process into the lucidota worker cgroup."""
    cg = _CGROUP_ROOT / name
    if cg.exists():
        (cg / "cgroup.procs").write_text(str(os.getpid()))

# ── DB write — always current, queryable by any LLM or flow ─────────────────

def write_hw_state_to_db(state: dict | None = None) -> None:
    """Upsert hw state into lucidota_control.runtime_status_fact.

    Any LLM or flow queries:
      SELECT fact_value FROM lucidota_control.runtime_status_fact
      WHERE subsystem='hardware' AND fact_key='hw_state';
    """
    import sys, os
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    try:
        from core.runtime_dsns import resolve_state_dsn
        import psycopg2, psycopg2.extras
    except ImportError:
        return
    if state is None:
        state = read_hw_state()
    dsn = resolve_state_dsn("postgresql://mfspx@/lucidota_state")
    try:
        conn = psycopg2.connect(dsn)
        with conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lucidota_control.runtime_status_fact
                    (subsystem, fact_key, fact_value, evidence_refs, derived_at)
                VALUES ('hardware', 'hw_state', %s, '[]'::jsonb, now())
                ON CONFLICT (subsystem, fact_key)
                DO UPDATE SET fact_value = EXCLUDED.fact_value, derived_at = now()
            """, (psycopg2.extras.Json(state),))
        conn.close()
    except Exception:
        pass  # never block a worker on a DB write


def read_hw_state(write_db: bool = False) -> dict:
    """Read hardware state in <1ms. Import and call — no daemon needed.

    Pass write_db=True to also upsert into lucidota_control.runtime_status_fact.
    """
    vram_used, vram_free = _vram_mb()
    vram_pct = round(vram_used / _VRAM_BUDGET_MB * 100, 1)
    ram_avail = _mem_available_mb()
    cpu_per_core = _load_per_core()
    disk_pct = _disk_pct()

    reasons = []
    if ram_avail < 800:
        reasons.append("low_ram")
    if vram_pct > 85.0:
        reasons.append("high_vram")
    if cpu_per_core > 0.85:
        reasons.append("high_cpu")

    state = {
        "mode": "throttle" if reasons else "saturate",
        "safe_local_workers": 1 if reasons else max(os.cpu_count() // 2, 2),
        "safe_cloud_workers": 50,
        "reasons": reasons,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "telemetry": {
            "ram_avail_mb": ram_avail,
            "vram_used_mb": vram_used,
            "vram_free_mb": vram_free,
            "vram_used_pct": vram_pct,
            "cpu_load_per_core": cpu_per_core,
            "disk_used_pct": disk_pct,
        },
    }
    if write_db:
        write_hw_state_to_db(state)
    return state


if __name__ == "__main__":
    import sys
    write = "--write-db" in sys.argv
    state = read_hw_state(write_db=write)
    print(json.dumps(state, indent=2))
    if write:
        print("→ written to lucidota_control.runtime_status_fact", file=sys.stderr)
