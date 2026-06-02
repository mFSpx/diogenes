from __future__ import annotations

from pathlib import Path


def test_bge_fleet_defaults_cpu_safe_and_gpu_requires_explicit_ngl():
    script = Path("scripts/lucidota_bge_fleet.sh").read_text(encoding="utf-8")
    safe_ops = Path("scripts/lucidota_safe_ops_env.sh").read_text(encoding="utf-8")

    assert 'export LUCIDOTA_BGE_NGL="${LUCIDOTA_BGE_NGL:-0}"' in safe_ops
    assert 'NGL="${LUCIDOTA_BGE_NGL:-0}"' in script
    assert ':-99' not in safe_ops
    assert ':-99' not in script
    assert "elif (( NGL > 0 )); then" in script
    assert "COUNT=16" in script
    assert "COUNT=1" in script
    assert "GPU_INSTANCES=1" in script
    assert "SLOTS_PER=$(( COUNT / GPU_INSTANCES ))" in script
    assert "CTX_PER=$(( SLOTS_PER * 2048 ))" in script


def test_bge_fleet_restarts_existing_port_when_ngl_mode_mismatches():
    script = Path("scripts/lucidota_bge_fleet.sh").read_text(encoding="utf-8")

    assert "restart_if_mode_mismatch" in script
    assert 'expected_ngl="$1"' in script
    assert 'cmdline="$(tr \'\\0\' \' \' < "/proc/$pid/cmdline")"' in script
    assert 'healthy but not ${label} (-ngl ${expected_ngl})' in script
