#!/usr/bin/env python3
"""LUCIDOTA model VRAM/loadout governor.

Advisory only: observes GPU memory, estimates resident slot VRAM, writes the
load/defer/reject decision to Postgres, and never imports or loads model weights.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB = os.environ.get("DBOS_SYSTEM_DATABASE_URL", "postgresql://mfspx@/lucidota_state")
SCHEMA = ROOT / "06_SCHEMA" / "002_model_runtime.sql"

GOVERNOR_SQL = """
CREATE TABLE IF NOT EXISTS lucidota_runtime.load_governor_decision (
  decision_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  loadout_id text NOT NULL,
  target_gpu text NOT NULL DEFAULT '',
  budget_vram_mb integer NOT NULL,
  observed_used_mb integer,
  observed_free_mb integer,
  estimated_required_mb integer NOT NULL,
  headroom_mb integer NOT NULL,
  decision text NOT NULL CHECK (decision IN ('allow','defer','reject')),
  rationale text NOT NULL DEFAULT '',
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE lucidota_runtime.load_governor_decision
  ADD COLUMN IF NOT EXISTS gpu_observation jsonb NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE lucidota_runtime.load_governor_decision
  ADD COLUMN IF NOT EXISTS action_plan jsonb NOT NULL DEFAULT '{}'::jsonb;
CREATE INDEX IF NOT EXISTS load_governor_decision_loadout_idx
  ON lucidota_runtime.load_governor_decision (loadout_id, created_at DESC);
"""


def gpu_memory() -> dict[str, Any]:
    """Return current GPU memory without initializing CUDA or loading models."""
    if not shutil.which("nvidia-smi"):
        return {"status": "missing", "message": "nvidia-smi not found"}
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version,pstate",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return {"status": "error", "stderr": result.stderr[-500:]}
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 7:
            continue
        idx, name, total, used, free, driver, pstate = parts[:7]
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "total_mb": int(total),
                "used_mb": int(used),
                "free_mb": int(free),
                "driver_version": driver,
                "pstate": pstate,
            }
        )
    if not gpus:
        return {"status": "error", "stderr": result.stderr[-500:], "stdout": result.stdout[-500:]}
    primary = gpus[0]
    return {"status": "ok", "selected_index": primary["index"], **primary, "gpus": gpus}


def estimate_mb(parameter_count: int | None, quantization: str | None, explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    if not parameter_count:
        return 128
    quant = (quantization or "").lower()
    bytes_per_param = 0.5 if "4" in quant else 1.0 if "8" in quant else 2.0
    # 25% runtime overhead + small KV/runtime floor. Conservative enough for planning,
    # still explicitly marked as an estimate until benchmarks replace it.
    return max(64, int(parameter_count * bytes_per_param / 1_000_000 * 1.25) + 96)


def build_action_plan(
    slots: list[dict[str, Any]],
    gpu: dict[str, Any],
    budget_vram_mb: int,
    reserve_mb: int,
) -> dict[str, Any]:
    """Turn estimated slots + current GPU state into an actionable load/defer plan."""
    observed_free = gpu.get("free_mb") if gpu.get("status") == "ok" else None
    observed_used = gpu.get("used_mb") if gpu.get("status") == "ok" else None
    observed_total = gpu.get("total_mb") if gpu.get("status") == "ok" else None
    effective_budget = min(budget_vram_mb, observed_total) if observed_total else budget_vram_mb
    allocatable_now = max(0, (min(effective_budget, observed_free) if observed_free is not None else effective_budget) - reserve_mb)
    max_slot_capacity = max(0, effective_budget - reserve_mb)

    remaining_now = allocatable_now
    required = 0
    load_mb = 0
    defer_mb = 0
    reject_mb = 0
    plan_slots: list[dict[str, Any]] = []

    for slot in slots:
        need = int(slot["estimated_total_mb"])
        required += need
        item = dict(slot)
        item["remaining_before_mb"] = remaining_now
        if need > max_slot_capacity:
            item["action"] = "reject"
            item["reason"] = f"slot requires {need}MB; max slot capacity after reserve is {max_slot_capacity}MB"
            reject_mb += need
        elif need <= remaining_now:
            item["action"] = "load"
            item["reason"] = f"fits current allocatable VRAM; load before lower-priority deferred slots"
            remaining_now -= need
            load_mb += need
        else:
            item["action"] = "defer"
            item["reason"] = f"needs {need}MB but only {remaining_now}MB is allocatable now"
            defer_mb += need
        item["remaining_after_mb"] = remaining_now
        plan_slots.append(item)

    headroom_after_full_load = effective_budget - required - reserve_mb
    decision = "reject" if reject_mb else "defer" if defer_mb else "allow"
    return {
        "decision": decision,
        "effective_budget_mb": effective_budget,
        "reserve_mb": reserve_mb,
        "observed_used_mb": observed_used,
        "observed_free_mb": observed_free,
        "allocatable_now_mb": allocatable_now,
        "estimated_required_mb": required,
        "estimated_load_now_mb": load_mb,
        "estimated_defer_mb": defer_mb,
        "estimated_reject_mb": reject_mb,
        "headroom_mb": headroom_after_full_load,
        "slots": plan_slots,
    }


def rationale_for(plan: dict[str, Any]) -> str:
    return (
        f"decision={plan['decision']} required={plan['estimated_required_mb']}MB "
        f"load_now={plan['estimated_load_now_mb']}MB defer={plan['estimated_defer_mb']}MB "
        f"reject={plan['estimated_reject_mb']}MB reserve={plan['reserve_mb']}MB "
        f"allocatable_now={plan['allocatable_now_mb']}MB budget={plan['effective_budget_mb']}MB "
        f"full_load_headroom={plan['headroom_mb']}MB"
    )


def loadout_rows(conn: Any, loadout_id: str | None) -> tuple[tuple[Any, ...], list[tuple[Any, ...]]]:
    if loadout_id:
        loadout = conn.execute(
            """
            SELECT loadout_id, target_gpu, budget_vram_mb
            FROM lucidota_runtime.resident_loadout
            WHERE loadout_id = %s
            ORDER BY active DESC, created_at DESC
            LIMIT 1
            """,
            (loadout_id,),
        ).fetchone()
    else:
        loadout = conn.execute(
            """
            SELECT loadout_id, target_gpu, budget_vram_mb
            FROM lucidota_runtime.resident_loadout
            WHERE active = true
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()
    if not loadout:
        raise SystemExit(f"no resident loadout found{f' for {loadout_id}' if loadout_id else ''}")
    rows = conn.execute(
        """
        SELECT
          s.slot_name,
          s.instance_count,
          s.expected_vram_mb,
          m.model_id,
          m.role,
          m.parameter_count,
          m.quantization,
          m.expected_vram_mb
        FROM lucidota_runtime.resident_loadout_slot s
        JOIN lucidota_runtime.model_candidate m USING(model_id)
        WHERE s.loadout_id = %s
        ORDER BY s.priority ASC, s.slot_name
        """,
        (loadout[0],),
    ).fetchall()
    return loadout, list(rows)


def evaluate(conn: Any, loadout_id: str | None, reserve_mb: int) -> dict[str, Any]:
    conn.execute(SCHEMA.read_text())
    conn.execute(GOVERNOR_SQL)
    (lid, target_gpu, budget), rows = loadout_rows(conn, loadout_id)

    slots: list[dict[str, Any]] = []
    for slot, count, slot_mb, model_id, role, params, quant, model_mb in rows:
        explicit = slot_mb if slot_mb is not None else model_mb
        each = estimate_mb(params, quant, explicit)
        slots.append(
            {
                "slot": slot,
                "model_id": model_id,
                "role": role,
                "instances": count,
                "estimated_each_mb": each,
                "estimated_total_mb": each * count,
            }
        )

    gpu = gpu_memory()
    plan = build_action_plan(slots, gpu, int(budget), reserve_mb)
    rationale = rationale_for(plan)
    detail = {"slots": plan["slots"], "gpu": gpu, "reserve_mb": reserve_mb, "action_plan": plan}

    row = conn.execute(
        """
        INSERT INTO lucidota_runtime.load_governor_decision
          (loadout_id, target_gpu, budget_vram_mb, observed_used_mb, observed_free_mb,
           estimated_required_mb, headroom_mb, decision, rationale, detail, gpu_observation, action_plan)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
        RETURNING decision_id::text
        """,
        (
            lid,
            target_gpu,
            plan["effective_budget_mb"],
            plan["observed_used_mb"],
            plan["observed_free_mb"],
            plan["estimated_required_mb"],
            plan["headroom_mb"],
            plan["decision"],
            rationale,
            json.dumps(detail),
            json.dumps(gpu),
            json.dumps(plan),
        ),
    ).fetchone()
    conn.commit()
    return {
        "ok": True,
        "decision_id": row[0],
        "loadout_id": lid,
        "decision": plan["decision"],
        "rationale": rationale,
        "gpu": gpu,
        "action_plan": plan,
        "slots": plan["slots"],
    }


def main() -> int:
    import psycopg

    parser = argparse.ArgumentParser(prog="lucidota-model-governor")
    parser.add_argument("--loadout-id")
    parser.add_argument("--reserve-mb", type=int, default=512)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    with psycopg.connect(DB) as conn:
        report = evaluate(conn, args.loadout_id, args.reserve_mb)
    print(json.dumps(report, sort_keys=True) if args.json else report)
    return 0 if report["decision"] in ("allow", "defer") else 1


if __name__ == "__main__":
    raise SystemExit(main())
