import argparse, json, signal, subprocess, time
import psutil
from datetime import datetime, timezone

_STATE_FILE = "/tmp/lucidota_governor_state.json"
_MAX_VRAM_MB = 4096
_stop = False


def read_governor_state() -> dict:
    try:
        with open(_STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _vram_used_pct() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(out.strip()) / _MAX_VRAM_MB * 100.0
    except Exception:
        return 0.0


def _sample() -> dict:
    vm = psutil.virtual_memory()
    cpu = psutil.getloadavg()[0] / psutil.cpu_count()
    disk = psutil.disk_usage("/").percent
    return {
        "ram_avail_mb": int(vm.available / 2**20),
        "vram_used_pct": round(_vram_used_pct(), 1),
        "cpu_load_per_core": round(cpu, 2),
        "disk_used_pct": round(disk, 1),
    }


def _handle_sig(*_):
    global _stop
    _stop = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    cpu_cnt = psutil.cpu_count()
    max_local = max(cpu_cnt // 2, 2)
    safe_local = max_local
    spike = clean = 0

    while not _stop:
        t = _sample()
        reasons = []
        if t["ram_avail_mb"] < 800:
            reasons.append("low_ram")
        if t["vram_used_pct"] > 85.0:
            reasons.append("high_vram")
        if t["cpu_load_per_core"] > 0.85:
            reasons.append("high_cpu")

        if reasons:
            clean = 0
            spike += 1
            safe_local = max(1, safe_local - 2) if spike == 1 else 1
            mode = "throttle"
        else:
            spike = 0
            clean += 1
            if clean >= 2 and safe_local < max_local:
                safe_local += 1
            mode = "saturate"

        state = {
            "mode": mode,
            "safe_local_workers": safe_local,
            "safe_cloud_workers": 50,
            "reasons": reasons,
            "sampled_at": datetime.now(timezone.utc).isoformat(),
            "telemetry": t,
        }

        if args.dry_run:
            print(json.dumps(state, indent=2))
        else:
            with open(_STATE_FILE, "w") as f:
                json.dump(state, f)

        time.sleep(5)


if __name__ == "__main__":
    main()
