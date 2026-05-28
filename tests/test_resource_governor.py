import json
import subprocess
import sys
from pathlib import Path

from scripts import resource_governor as rg


def test_governance_decision_throttles_and_learns_from_pressure():
    snapshot = {
        "cpu": {"count": 8, "loadavg_1m": 16.0},
        "memory": {"available_mb": 512, "swap_used_pct": 42.0},
        "disk": {"used_pct": 94.0},
        "vram": {"gpus": [{"memory_total_mb": 4096, "memory_used_mb": 3900}]},
    }

    decision = rg.governance_decision(
        snapshot,
        requested_workers=6,
        policy=rg.ResourcePolicy(
            default_workers=4,
            max_workers=8,
            min_mem_available_mb=1024,
            max_swap_used_pct=25,
            max_disk_used_pct=90,
            max_load_per_cpu=1.2,
            max_vram_used_pct=90,
        ),
    )

    assert decision["throttle"] is True
    assert decision["safe_workers"] == 1
    assert "mem_available_below_floor" in decision["reasons"]
    assert "swap_pressure" in decision["reasons"]
    assert "disk_pressure" in decision["reasons"]
    assert "vram_pressure" in decision["reasons"]
    assert any(delta["heuristic"] == "reduce_worker_count" for delta in decision["learning_deltas"])


def test_pid_registry_receipt_contains_non_negotiable_fields(tmp_path):
    entry = rg.PidRegistration(
        pid=12345,
        owner="pytest",
        purpose="collar a fake worker",
        cwd="/tmp",
        command=["python3", "worker.py"],
        max_memory_mb=256,
        max_cpu_percent=50.0,
        kill_policy="terminate_then_kill_after_5s",
        status="running",
    )

    path = rg.write_pid_registry(tmp_path, entry, telemetry={"safe_workers": 1}, db_result={"attempted": False})
    data = json.loads(path.read_text())

    assert path.parent == tmp_path / "05_OUTPUTS" / "runtime"
    assert data["schema"] == "lucidota.resource_governor.pid_registry.v1"
    for key in ["pid", "owner", "purpose", "cwd", "command", "max_memory_mb", "max_cpu_percent", "kill_policy"]:
        assert key in data["registration"]
    assert data["registration"]["pid"] == 12345
    assert data["registration"]["kill_policy"] == "terminate_then_kill_after_5s"
    assert data["telemetry"]["safe_workers"] == 1
    assert data["db_result"]["attempted"] is False


def test_pg_supervision_plan_targets_only_stale_idle_transactions():
    rows = [
        {"pid": 10, "state": "idle in transaction", "xact_age_seconds": 901, "query": "RELEASE SAVEPOINT korpus_file"},
        {"pid": 11, "state": "active", "xact_age_seconds": 1200, "query": "select pg_sleep(10)"},
        {"pid": 12, "state": "idle in transaction", "xact_age_seconds": 10, "query": "select 1"},
        {"pid": 13, "state": "idle in transaction", "xact_age_seconds": 999, "query": "protected"},
    ]

    plan = rg.pg_supervision_plan(rows, max_idle_xact_seconds=300, protected_pids={13})

    assert [item["pid"] for item in plan["terminate_candidates"]] == [10]
    assert plan["terminate_candidates"][0]["reason"] == "stale_idle_in_transaction"
    assert plan["protected_pids"] == [13]


def test_resource_governor_schema_is_wired_into_control_schema_apply():
    schema = Path("06_SCHEMA/122_resource_governor.sql").read_text()
    apply_script = Path("scripts/apply_lucidota_control_schema.sh").read_text()

    assert "CREATE TABLE IF NOT EXISTS lucidota_control.pid_registry" in schema
    assert "CREATE TABLE IF NOT EXISTS lucidota_control.resource_throttle_receipt" in schema
    assert "CREATE TABLE IF NOT EXISTS lucidota_control.pg_supervision_receipt" in schema
    assert "122_resource_governor.sql" in apply_script


def test_resource_governor_spawn_dry_run_is_collared_without_spawning(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/resource_governor.py",
            "--root",
            str(tmp_path),
            "--database-url",
            "",
            "spawn",
            "--owner",
            "pytest",
            "--purpose",
            "prove dry-run collar",
            "--",
            sys.executable,
            "-c",
            "print('would spawn')",
        ],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    report_line = next(line for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH="))
    receipt = json.loads((tmp_path / report_line.split("=", 1)[1]).read_text())
    assert receipt["schema"] == "lucidota.resource_governor.pid_registry.v1"
    assert receipt["registration"]["status"] == "planned"
    assert receipt["registration"]["pid"] == 0
    assert any("would spawn" in part for part in receipt["registration"]["command"])


def test_resource_governor_allows_json_flag_after_subcommand(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/resource_governor.py",
            "--root",
            str(tmp_path),
            "--database-url",
            "",
            "preflight",
            "--requested-workers",
            "1",
            "--json",
        ],
        text=True,
        capture_output=True,
        timeout=20,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "RESOURCE_GOVERNOR=PASS" in proc.stdout
    assert "\"schema\": \"lucidota.resource_governor.preflight.v1\"" in proc.stdout
