from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

LIVE_BASE_URL = "http://127.0.0.1:3000"
TARGET_ROUTE = "indy_reads_target_model_loadout_current"
FABRIC_ROUTE = "indy_reads_vram_coprocessor_fabric_current"
REQUIRED_LOADOUT_STATE_FIELDS = {
    "intended_target",
    "current_substitute",
    "admitted_runtime",
    "resident_now",
    "swapout_candidate",
    "missing_artifact",
    "receipt_fields",
}
EMERGENCY_CLOSURE_KEYS = {
    "deepseek_1p5b_auxiliary_swapout",
    "ram_mamba_bonsai_tiny_overflow",
}


def _get(route: str) -> Any:
    try:
        with urllib.request.urlopen(f"{LIVE_BASE_URL}/{route}", timeout=10) as resp:
            assert resp.status == 200
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AssertionError(f"GET /{route} returned HTTP {exc.code}: {body}") from exc


def _by_key(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["model_key"]: row for row in rows}


def _status(row: dict[str, Any], field: str) -> bool:
    payload = row[field]
    assert isinstance(payload, dict), f"{row['model_key']}.{field} must be a JSON object"
    assert "status" in payload, f"{row['model_key']}.{field} must expose status"
    return payload["status"] is True


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True).lower()


def test_indy_reads_target_loadout_routes_are_exposed_by_postgrest_openapi() -> None:
    openapi = _get("")
    paths = openapi.get("paths", {})

    assert f"/{TARGET_ROUTE}" in paths
    assert f"/{FABRIC_ROUTE}" in paths
    assert "get" in paths[f"/{TARGET_ROUTE}"]
    assert "get" in paths[f"/{FABRIC_ROUTE}"]


def test_indy_reads_target_model_loadout_canonicalizes_target_body_not_emergency_stack() -> None:
    rows = _get(TARGET_ROUTE)
    assert rows
    by_key = _by_key(rows)

    for row in rows:
        missing = REQUIRED_LOADOUT_STATE_FIELDS - set(row)
        assert not missing, f"{row.get('model_key', '<unknown>')} missing fields: {sorted(missing)}"
        for field in REQUIRED_LOADOUT_STATE_FIELDS - {"receipt_fields"}:
            _status(row, field)
        assert isinstance(row["receipt_fields"], dict)
        assert row["receipt_fields"], f"{row['model_key']} must expose receipt_fields"

    needle = by_key["needle_26m_router_swarm"]
    assert _status(needle, "intended_target")
    assert _status(needle, "admitted_runtime")
    assert needle["logical_lane_count"] == 20
    assert needle["receipt_fields"]["shared_weight_proven"] is True
    assert needle["receipt_fields"]["kv_policy"] == "shared_weight_cpu_preload"

    bonsai = by_key["bonsai_8b_1bit_dual_lane"]
    assert _status(bonsai, "intended_target")
    assert _status(bonsai, "admitted_runtime")
    assert bonsai["logical_lane_count"] == 2
    assert bonsai["receipt_fields"]["shared_weight_proven"] is False
    assert bonsai["receipt_fields"]["max_context_admitted"] == 10000

    bimamba = by_key["bimamba_mamba2_1p3b_ternary"]
    assert _status(bimamba, "intended_target")
    assert _status(bimamba, "missing_artifact")
    assert not _status(bimamba, "resident_now")

    deepseek = by_key["deepseek_1p5b_auxiliary_swapout"]
    assert not _status(deepseek, "intended_target")
    assert _status(deepseek, "current_substitute")
    assert _status(deepseek, "admitted_runtime")
    assert _status(deepseek, "swapout_candidate")
    assert "emergency closure" in _json_text(deepseek["current_substitute"])

    mamba_overflow = by_key["ram_mamba_bonsai_tiny_overflow"]
    assert not _status(mamba_overflow, "intended_target")
    assert _status(mamba_overflow, "current_substitute")
    assert _status(mamba_overflow, "admitted_runtime")
    assert mamba_overflow["preemption_group"] == "ram_overflow"
    assert "emergency closure" in _json_text(mamba_overflow["current_substitute"])

    intended_keys = {row["model_key"] for row in rows if row["intended_target"]["status"]}
    assert intended_keys.isdisjoint(EMERGENCY_CLOSURE_KEYS)


def test_indy_reads_vram_coprocessor_fabric_calculates_bonsai_kv_pressure() -> None:
    rows = _get(f"{FABRIC_ROUTE}?limit=1")
    assert len(rows) == 1
    row = rows[0]
    assert row.get("fabric_id", FABRIC_ROUTE) == FABRIC_ROUTE

    pressure = {
        (item["context_tokens"], item["k_quant"], item["v_quant"]): item
        for item in row["bonsai_kv_pressure"]
    }
    expected_per_lane_mb = {
        (16000, "q4", "q8"): 750.0,
        (16000, "q4", "q4"): 500.0,
        (12000, "q4", "q8"): 562.5,
        (12000, "q4", "q4"): 375.0,
        (10000, "q4", "q4"): 312.5,
    }

    assert set(pressure) == set(expected_per_lane_mb)
    for key, per_lane_mb in expected_per_lane_mb.items():
        assert pressure[key]["per_lane_kv_mb"] == per_lane_mb
        assert pressure[key]["dual_lane_kv_mb"] == per_lane_mb * 2
        assert "admission_decision" in pressure[key]

    assert pressure[(16000, "q4", "q8")]["admission_decision"] == "reject"
    assert pressure[(16000, "q4", "q4")]["admission_decision"] == "reject"
    assert pressure[(12000, "q4", "q8")]["admission_decision"] == "reject"
    assert pressure[(12000, "q4", "q4")]["admission_decision"] == "reject"
    assert pressure[(10000, "q4", "q4")]["admission_decision"] == "admit"
    assert row["kv_governor"]["formula"] == (
        "context_tokens * layers * kv_heads * head_dim * (k_bytes + v_bytes) / 1048576"
    )
    components = {item["fabric_key"]: item for item in row["fabric_components"]}
    assert "treelite_stack_deterministic_gate_asset" in components
    assert "fft_gpu_batch_kernels_where_measured" in components
    assert "bernoulli_venturi_adversarial_harness_reserve" in components
    for item in components.values():
        assert item["receipt_fields"]["kv_policy"]
        assert "logical_lane_count" in item["receipt_fields"]
