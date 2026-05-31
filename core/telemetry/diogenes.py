#!/usr/bin/env python3
"""Diogenes: passive human I/O and hardware telemetry compression for LUCIDOTA."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Iterable

try:
    import psutil
except Exception:  # pragma: no cover - psutil is expected but optional
    psutil = None


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CompressedActivity:
    mouse_delta_sum: float
    keystroke_burst: int
    click_count: int
    scroll_count: int
    flow_friction_score: float
    window_seconds: int
    source: str = "diogenes"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HardwareTelemetry:
    timestamp_utc: str
    cpu: dict[str, Any]
    memory: dict[str, Any]
    gpu: dict[str, Any]
    integrated_gpu: dict[str, Any]
    power: dict[str, Any]
    thermal: dict[str, Any]
    pcie: dict[str, Any]
    window_title: str | None = None
    source: str = "diogenes"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _read_text(path: str | Path, default: str = "") -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return default


def _parse_pressure_file(path: str | Path) -> dict[str, float]:
    text = _read_text(path)
    if not text:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        for part in line.split():
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                out[k] = float(v)
            except Exception:
                continue
    return out


def _active_window_title() -> str | None:
    if not shutil.which("xdotool"):
        return None
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowname"],
            text=True, capture_output=True, timeout=1, check=False,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _gpu_query() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi unavailable"}
    query = [
        "memory.used,memory.free,memory.total,utilization.gpu,temperature.gpu,fan.speed,power.draw,power.limit,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + query[0], query[1]],
        text=True,
        capture_output=True,
        check=False,
        timeout=2,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"available": False, "reason": "nvidia-smi query failed", "stderr": result.stderr[-500:]}
    devices: list[dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 12:
            continue
        used, free, total, util, temp, fan, power_draw, power_limit, gen_cur, gen_max, width_cur, width_max = parts[:12]
        try:
            devices.append(
                {
                    "memory_used_mb": int(float(used)),
                    "memory_free_mb": int(float(free)),
                    "memory_total_mb": int(float(total)),
                    "utilization_gpu_percent": float(util),
                    "temperature_gpu_c": float(temp),
                    "fan_speed_percent": float(fan) if fan not in {"[N/A]", "N/A"} else None,
                    "power_draw_w": float(power_draw) if power_draw not in {"[N/A]", "N/A"} else None,
                    "power_limit_w": float(power_limit) if power_limit not in {"[N/A]", "N/A"} else None,
                    "pcie_link_gen_current": int(float(gen_cur)) if gen_cur not in {"[N/A]", "N/A"} else None,
                    "pcie_link_gen_max": int(float(gen_max)) if gen_max not in {"[N/A]", "N/A"} else None,
                    "pcie_link_width_current": int(float(width_cur)) if width_cur not in {"[N/A]", "N/A"} else None,
                    "pcie_link_width_max": int(float(width_max)) if width_max not in {"[N/A]", "N/A"} else None,
                }
            )
        except Exception:
            continue
    return {"available": bool(devices), "devices": devices, "primary": devices[0] if devices else None}


def _integrated_gpu_state() -> dict[str, Any]:
    candidates: list[str] = []
    if shutil.which("lspci"):
        result = subprocess.run(["lspci", "-nn"], text=True, capture_output=True, check=False, timeout=2)
        if result.stdout:
            for line in result.stdout.splitlines():
                if any(token in line.lower() for token in ("vga compatible controller", "3d controller", "display controller")):
                    candidates.append(line.strip())
    return {
        "present": bool(candidates),
        "models": candidates,
    }


def _power_state() -> dict[str, Any]:
    readings: dict[str, Any] = {"system_power_w": None}
    if shutil.which("upower"):
        try:
            out = subprocess.run(["upower", "-e"], text=True, capture_output=True, check=False, timeout=2).stdout
            devices = [line.strip() for line in out.splitlines() if line.strip()]
            for device in devices:
                info = subprocess.run(["upower", "-i", device], text=True, capture_output=True, check=False, timeout=2)
                if "energy-rate" in info.stdout or "power" in info.stdout:
                    for line in info.stdout.splitlines():
                        if "energy-rate" in line or "power" in line.lower():
                            m = re.search(r"([0-9.]+)\s*W", line)
                            if m:
                                readings["system_power_w"] = float(m.group(1))
                                return readings
        except Exception:
            pass
    for candidate in Path("/sys/class/power_supply").glob("*/power_now"):
        try:
            raw = candidate.read_text(encoding="utf-8", errors="ignore").strip()
            if raw.isdigit():
                readings["system_power_w"] = round(int(raw) / 1_000_000.0, 3)
                return readings
        except Exception:
            continue
    return readings


def _thermal_state() -> dict[str, Any]:
    cpu_temps: list[float] = []
    for zone in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            raw = zone.read_text(encoding="utf-8", errors="ignore").strip()
            if raw:
                cpu_temps.append(round(int(raw) / 1000.0, 3))
        except Exception:
            continue
    gpu = _gpu_query()
    primary = gpu.get("primary") or {}
    return {
        "cpu_temp_c": max(cpu_temps) if cpu_temps else None,
        "gpu_temp_c": primary.get("temperature_gpu_c"),
        "gpu_fan_speed_percent": primary.get("fan_speed_percent"),
    }


def sample_hardware_telemetry() -> dict[str, Any]:
    """Capture host hardware stress without touching model residency."""
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    cpu_percent = psutil.cpu_percent(interval=None, percpu=True) if psutil else []
    load_avg = os.getloadavg() if hasattr(os, "getloadavg") else (None, None, None)
    vm = psutil.virtual_memory() if psutil else None
    sm = psutil.swap_memory() if psutil else None
    pressure_cpu = _parse_pressure_file("/proc/pressure/cpu")
    pressure_mem = _parse_pressure_file("/proc/pressure/memory")
    pressure_io = _parse_pressure_file("/proc/pressure/io")
    gpu = _gpu_query()
    mem = {
        "used_mb": round((vm.used / 1024 / 1024), 3) if vm else None,
        "available_mb": round((vm.available / 1024 / 1024), 3) if vm else None,
        "percent": round(vm.percent, 3) if vm else None,
        "swap_used_mb": round((sm.used / 1024 / 1024), 3) if sm else None,
        "swap_percent": round(sm.percent, 3) if sm else None,
        "pressure": {"cpu": pressure_cpu, "memory": pressure_mem, "io": pressure_io},
    }
    cpu = {
        "load_avg_1m": load_avg[0] if load_avg else None,
        "load_avg_5m": load_avg[1] if load_avg else None,
        "load_avg_15m": load_avg[2] if load_avg else None,
        "overall_percent": round(sum(cpu_percent) / len(cpu_percent), 3) if cpu_percent else None,
        "per_core_percent": [round(float(v), 3) for v in cpu_percent] if cpu_percent else [],
        "cores": psutil.cpu_count(logical=True) if psutil else None,
    }
    telemetry = HardwareTelemetry(
        timestamp_utc=timestamp,
        cpu=cpu,
        memory=mem,
        gpu=gpu,
        integrated_gpu=_integrated_gpu_state(),
        power=_power_state(),
        thermal=_thermal_state(),
        window_title=_active_window_title(),
        pcie={
            "bandwidth_utilization_proxy": (
                round(
                    float(gpu.get("primary", {}).get("pcie_link_width_current") or 0)
                    / max(float(gpu.get("primary", {}).get("pcie_link_width_max") or 1), 1.0),
                    3,
                )
                if gpu.get("primary")
                else None
            ),
            "link_gen_current": (gpu.get("primary") or {}).get("pcie_link_gen_current"),
            "link_gen_max": (gpu.get("primary") or {}).get("pcie_link_gen_max"),
            "link_width_current": (gpu.get("primary") or {}).get("pcie_link_width_current"),
            "link_width_max": (gpu.get("primary") or {}).get("pcie_link_width_max"),
        },
    )
    return telemetry.as_dict()


def compress_activity(payload: dict[str, Any] | None = None, state: dict[str, Any] | None = None, *, window_seconds: int = 2) -> dict[str, Any]:
    payload = payload or {}
    state = state or {}
    mouse_sources: Iterable[Any] = (
        payload.get("mouse_deltas")
        or state.get("mouse_deltas")
        or payload.get("mouse_moves")
        or state.get("mouse_moves")
        or []
    )
    mouse_delta_sum = _float(payload.get("mouse_delta_sum") or state.get("mouse_delta_sum"), 0.0)
    if mouse_sources:
        for item in mouse_sources:
            if isinstance(item, dict):
                mouse_delta_sum += abs(_float(item.get("dx"))) + abs(_float(item.get("dy")))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                mouse_delta_sum += abs(_float(item[0])) + abs(_float(item[1]))
            else:
                mouse_delta_sum += abs(_float(item))

    keystroke_burst = _int(payload.get("keystroke_burst") or state.get("keystroke_burst"), 0)
    if not keystroke_burst:
        key_events = payload.get("key_events") or state.get("key_events") or []
        if isinstance(key_events, list):
            keystroke_burst = sum(1 for item in key_events if item)
        else:
            keystroke_burst = _int(key_events, 0)

    click_count = _int(payload.get("click_count") or state.get("click_count"), 0)
    scroll_count = _int(payload.get("scroll_count") or state.get("scroll_count"), 0)

    friction = (_float(keystroke_burst) + _float(click_count) + _float(scroll_count)) / max(1.0, float(window_seconds))
    return CompressedActivity(
        mouse_delta_sum=round(mouse_delta_sum, 3),
        keystroke_burst=keystroke_burst,
        click_count=click_count,
        scroll_count=scroll_count,
        flow_friction_score=round(friction, 3),
        window_seconds=int(window_seconds),
    ).as_dict()


def staple_activity(packet: dict[str, Any], payload: dict[str, Any] | None = None, state: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(packet or {})
    activity = compress_activity(payload, state)
    out["compressed_activity"] = activity
    out["mouse_delta_sum"] = activity["mouse_delta_sum"]
    out["keystroke_burst"] = activity["keystroke_burst"]
    return out


def staple_hardware(packet: dict[str, Any], telemetry: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(packet or {})
    out["hardware_telemetry"] = telemetry or sample_hardware_telemetry()
    return out
