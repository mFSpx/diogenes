#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_luci_doctor_json_is_live() -> None:
    subprocess.run(
        [
            str(ROOT / "luci"),
            "elastic",
            "shape",
            "emit",
            "--artifact-uuid",
            "b3dad894-4b5c-4e89-9831-ad619f74cafe",
            "--synthetic",
            "--signal",
            "OBJECT=0.95",
            "--signal",
            "INDY_READS=1.22",
            "--signal",
            "ABSURD=0.76",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    proc = subprocess.run([str(ROOT / "luci"), "doctor", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "lucidota.luci.doctor.v1"
    assert "checks" in payload
    assert "live" in payload["checks"]
    assert payload["checks"]["live"]["root_orchestrator_current"]["ok"] is True
    assert payload["checks"]["live"]["daemon_status"]["ok"] is True
    assert payload["checks"]["live"]["controller_grant"]["ok"] is True
    assert payload["checks"]["live"]["agent_thread_runtime"]["ok"] is True
    assert payload["controller_grant"]["grant_key"] == "default_local_operator"
    assert payload["controller_grant"]["effective_status"] == "active"
    assert payload["agent_thread_runtime"]["thread_key"] == "root_operator_thread"
    assert payload["checks"]["live"]["percyphon_current"]["ok"] is True
    assert payload["checks"]["live"]["percyphon_village_matrix"]["ok"] is True
    assert payload["checks"]["live"]["elastic_shape_current"]["ok"] is True
    assert payload["checks"]["live"]["indy_attention_pressure_current"]["ok"] is True
    assert payload["orchestration"]
    assert payload["orchestration"]["mode"] == "sub_orchestrator"
    assert isinstance(payload.get("next_command_refs"), list) and payload["next_command_refs"]
    assert "root_orchestrator_current" in payload["next_command_refs"]
    assert "daemon_status" in payload["next_command_refs"]
    assert isinstance(payload.get("shape_refs"), list)
    assert "elastic_shape_current" in payload["shape_refs"]
    assert "indy_attention_pressure_current" in payload["shape_refs"]
    assert isinstance(payload.get("percyphon_refs"), list) and payload["percyphon_refs"]
    assert "percyphon_current" in payload["percyphon_refs"]
    assert isinstance(payload.get("route_refs"), list) and payload["route_refs"]
    assert isinstance(payload.get("surface_refs"), list) and payload["surface_refs"]
    assert isinstance(payload.get("renderer_refs"), list) and payload["renderer_refs"]
    assert payload["checks"]["live"]["root_orchestrator_current"]["rows"][0]["route_list"][0]["route_id"] in payload["route_refs"]
    assert isinstance(payload.get("capability_refs"), list) and payload["capability_refs"]
    assert payload["surfaces"]["capability_current"]["rows"][0]["active_capabilities"][0]["capability_key"] in payload["capability_refs"]


def test_luci_status_json_is_live() -> None:
    subprocess.run(
        [
            str(ROOT / "luci"),
            "elastic",
            "shape",
            "emit",
            "--artifact-uuid",
            "b3dad894-4e89-9831-2749-ad619f75beef",
            "--synthetic",
            "--signal",
            "OBJECT=0.95",
            "--signal",
            "INDY_READS=1.22",
            "--signal",
            "ABSURD=0.76",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    proc = subprocess.run([str(ROOT / "luci"), "status", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "lucidota.luci.status.v1"
    assert payload["ok"] is True
    assert payload["active_goal"]
    assert payload["manual"]
    assert payload["root_orchestrator"]
    assert payload["orchestration"]
    assert payload["orchestration"]["mode"] == "sub_orchestrator"
    assert payload["controller_grant"]
    assert payload["agent_thread_runtime"]
    assert payload["percyphon_current"]
    assert payload["percyphon_village_matrix"]
    assert payload["elastic_shape_current"]
    assert payload["indy_attention_pressure_current"]
    assert payload["shape_residuals_current"]
    assert isinstance(payload.get("next_command_refs"), list) and payload["next_command_refs"]
    assert "root_orchestrator_current" in payload["next_command_refs"]
    assert "daemon_status" in payload["next_command_refs"]
    assert isinstance(payload.get("shape_refs"), list) and payload["shape_refs"]
    assert "elastic_shape_current" in payload["shape_refs"]
    assert isinstance(payload.get("percyphon_refs"), list) and payload["percyphon_refs"]
    assert isinstance(payload.get("route_refs"), list) and payload["route_refs"]
    assert payload["root_orchestrator"]["route_list"][0]["route_id"] in payload["route_refs"]
    assert payload["orchestration"]["sub_orchestrator_priority"][0] == "live_truth_surfaces"
    assert isinstance(payload.get("capability_refs"), list) and payload["capability_refs"]
    assert isinstance(payload.get("surface_refs"), list) and payload["surface_refs"]
    assert isinstance(payload.get("renderer_refs"), list) and payload["renderer_refs"]
    assert payload["capability_current"]["active_capabilities"][0]["capability_key"] in payload["capability_refs"]


def test_luci_capability_list_json_is_live() -> None:
    proc = subprocess.run([str(ROOT / "luci"), "capability", "list", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["schema"] == "lucidota.luci.capability_list.v1"
    assert payload["ok"] is True
    assert isinstance(payload["capabilities"], list) and payload["capabilities"]
    assert payload["current"]
    assert payload["orchestration"]
    assert payload["orchestration"]["mode"] == "sub_orchestrator"
    assert isinstance(payload.get("next_command_refs"), list) and payload["next_command_refs"]
    assert "manual_current" in payload["next_command_refs"]
    assert "capability_registry" in payload["next_command_refs"]
    assert "renderer_registry" in payload["next_command_refs"]
    assert "schema_owner_manifest" in payload["next_command_refs"]
    assert "controller_grant" in payload["next_command_refs"]
    assert "agent_thread_runtime" in payload["next_command_refs"]
    assert "surface_registry" in payload["next_command_refs"]
    assert isinstance(payload.get("capability_refs"), list) and payload["capability_refs"]
    assert isinstance(payload.get("surface_refs"), list) and payload["surface_refs"]
    assert isinstance(payload.get("renderer_refs"), list) and payload["renderer_refs"]
    assert payload["capabilities"][0]["capability_key"] in payload["capability_refs"]
