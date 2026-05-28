#!/usr/bin/env python3
"""Regression smoke for the advisory LUCIDOTA model governor.

Uses synthetic slots/GPU observations only; it must not load model weights.
"""
from __future__ import annotations

import json

from lucidota_model_governor import build_action_plan, estimate_mb


def assert_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def main() -> int:
    slots = [
        {
            "slot": "listener",
            "model_id": "mamba2-1.3b-listener",
            "role": "listener",
            "instances": 1,
            "estimated_each_mb": 909,
            "estimated_total_mb": 909,
        },
        {
            "slot": "router_swarm",
            "model_id": "needle-26m",
            "role": "router",
            "instances": 6,
            "estimated_each_mb": 128,
            "estimated_total_mb": 768,
        },
        {
            "slot": "indy_reads_heavy_hitter",
            "model_id": "deepseek-1.5b-indy_reads-reads",
            "role": "heavy_hitter",
            "instances": 1,
            "estimated_each_mb": 1033,
            "estimated_total_mb": 1033,
        },
    ]

    allow = build_action_plan(
        slots,
        {"status": "ok", "total_mb": 4096, "used_mb": 256, "free_mb": 3840},
        budget_vram_mb=4096,
        reserve_mb=512,
    )
    assert_equal(allow["decision"], "allow", "allow decision")
    assert_equal([s["action"] for s in allow["slots"]], ["load", "load", "load"], "allow actions")

    defer = build_action_plan(
        slots,
        {"status": "ok", "total_mb": 4096, "used_mb": 2300, "free_mb": 1796},
        budget_vram_mb=4096,
        reserve_mb=512,
    )
    assert_equal(defer["decision"], "defer", "defer decision")
    assert_equal([s["action"] for s in defer["slots"]], ["load", "defer", "defer"], "defer actions")

    reject = build_action_plan(
        [{**slots[0], "slot": "oversize", "estimated_total_mb": 3800}],
        {"status": "ok", "total_mb": 4096, "used_mb": 0, "free_mb": 4096},
        budget_vram_mb=4096,
        reserve_mb=512,
    )
    assert_equal(reject["decision"], "reject", "reject decision")
    assert_equal(reject["slots"][0]["action"], "reject", "reject action")

    assert_equal(estimate_mb(26_000_000, "4-bit", None), 112, "4-bit estimate")
    assert_equal(estimate_mb(26_000_000, "8-bit", 333), 333, "explicit estimate")

    print(json.dumps({"ok": True, "allow": allow, "defer": defer, "reject": reject}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
