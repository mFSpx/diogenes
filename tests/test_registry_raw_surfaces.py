from __future__ import annotations

import json
import uuid
import urllib.request
import subprocess
from pathlib import Path

import psycopg


ROOT = Path(__file__).resolve().parents[1]


def test_provider_registry_raw_shell_alias_is_live() -> None:
    proc = subprocess.run([str(ROOT / "luci"), "provider", "registry", "raw", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source_url"].endswith("/provider_registry?order=provider_key.asc&limit=50")
    assert payload["payload"]


def test_capability_registry_raw_shell_alias_is_live() -> None:
    proc = subprocess.run([str(ROOT / "luci"), "capability", "registry", "raw", "--json"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["source_url"].endswith("/capability_registry?order=updated_at.desc&limit=50")
    assert payload["payload"]


def test_command_registry_view_is_live_and_typed() -> None:
    with urllib.request.urlopen("http://127.0.0.1:3000/command_registry?limit=1", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["command_id"]
    assert row["route_id"]
    assert row["capability_id"]
    assert row["surface_id"] == "worker_command_registry"
    assert row["intent"]
    assert row["renderer_id"]
    assert "packet" in row


def test_schema_owner_manifest_route_is_live_and_typed() -> None:
    with urllib.request.urlopen("http://127.0.0.1:3000/schema_owner_manifest?surface_id=eq.schema_owner_manifest", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    surface_ids = {row["surface_id"] for row in payload}
    assert surface_ids == {"schema_owner_manifest"}
    row = payload[0]
    assert row["canonical_owner"] == "lucidota_control"
    assert row["packet_class"] == "typed_packet"
    assert row["surface_kind"] == "table"
    assert row["approval_required"] is True


def test_schema_owner_manifest_covers_all_active_surfaces() -> None:
    with urllib.request.urlopen("http://127.0.0.1:3000/surface_registry?active=eq.true&limit=500", timeout=15) as resp:
        assert resp.status == 200
        active_surfaces = json.loads(resp.read().decode("utf-8"))
    with urllib.request.urlopen("http://127.0.0.1:3000/schema_owner_manifest?active=eq.true&limit=500", timeout=15) as resp:
        assert resp.status == 200
        manifest_rows = json.loads(resp.read().decode("utf-8"))

    active_ids = {row["surface_id"] for row in active_surfaces if row.get("active")}
    manifest_ids = {row["surface_id"] for row in manifest_rows if row.get("active")}
    assert active_ids == manifest_ids


def test_schema_owner_manifest_redefinition_requires_approval() -> None:
    with psycopg.connect("postgresql:///lucidota_state") as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT canonical_owner, packet_class, surface_kind, approval_required FROM lucidota_control.schema_owner_manifest WHERE surface_id = %s",
                ("schema_owner_manifest",),
            )
            original_owner, original_packet_class, original_surface_kind, original_approval_required = cur.fetchone()

    # First prove the guard rejects a redefinition without approval metadata.
    with psycopg.connect("postgresql:///lucidota_state") as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE lucidota_control.schema_owner_manifest SET canonical_owner = %s WHERE surface_id = %s",
                    ("blocked_owner", "schema_owner_manifest"),
                )
            conn.commit()
            raise AssertionError("expected schema_owner_manifest redefinition to be blocked")
        except psycopg.errors.InsufficientPrivilege:
            conn.rollback()

    # Then prove an approved redefinition can succeed and can be rolled back to the original state.
    approved_receipt = uuid.uuid4()
    with psycopg.connect("postgresql:///lucidota_state") as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE lucidota_control.schema_owner_manifest SET canonical_owner = %s, approved_by = %s, approved_at = now(), approval_receipt_uuid = %s, approval_note = %s WHERE surface_id = %s",
                ("lucidota_control", "pytest", approved_receipt, "redefinition guard test", "schema_owner_manifest"),
            )
            cur.execute(
                "UPDATE lucidota_control.schema_owner_manifest SET canonical_owner = %s, packet_class = %s, surface_kind = %s, approval_required = %s, approved_by = %s, approved_at = %s, approval_receipt_uuid = %s, approval_note = %s WHERE surface_id = %s",
                (original_owner, original_packet_class, original_surface_kind, original_approval_required, "", None, None, "", "schema_owner_manifest"),
            )


def test_surface_registry_route_is_live_and_typed() -> None:
    with urllib.request.urlopen("http://127.0.0.1:3000/surface_registry?surface_id=eq.manual_current", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["surface_id"] == "manual_current"
    assert row["canonical_owner"] == "lucidota_canon"
    assert row["packet_class"] == "typed_packet"
    assert row["surface_kind"] == "view"
    assert row["approval_required"] is True
    assert "approved_by" in row
    assert "approved_at" in row
    assert "approval_receipt_uuid" in row
    assert row["target"]
    assert isinstance(row.get("orchestration"), dict)
    assert row["orchestration"]["mode"] == "sub_orchestrator"
    assert row["orchestration"]["sub_orchestrator_priority"][0] == "live_truth_surfaces"
    assert isinstance(row.get("next_command_refs"), list) and row["next_command_refs"]
    assert "schema_owner_manifest" in row["next_command_refs"]
    assert "surface_registry" in row["next_command_refs"]
    assert "renderer_registry" in row["next_command_refs"]
    assert "packet" in row


def test_renderer_registry_route_is_live_and_typed() -> None:
    with urllib.request.urlopen("http://127.0.0.1:3000/renderer_registry?order=renderer_id.asc&limit=10", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["renderer_id"]
    assert row["renderer_kind"] == "script_path"
    assert row["command_count"] > 0
    assert row["source_surface"] == "worker_command_registry"
    assert isinstance(row.get("orchestration"), dict)
    assert row["orchestration"]["mode"] == "sub_orchestrator"
    assert row["orchestration"]["sub_orchestrator_priority"][0] == "live_truth_surfaces"
    assert isinstance(row.get("next_command_refs"), list) and row["next_command_refs"]
    assert "schema_owner_manifest" in row["next_command_refs"]
    assert "surface_registry" in row["next_command_refs"]
    assert "renderer_registry" in row["next_command_refs"]
    assert "packet" in row


def test_controller_grant_route_is_live_and_typed() -> None:
    with urllib.request.urlopen("http://127.0.0.1:3000/controller_grant?grant_key=eq.default_local_operator", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["grant_key"] == "default_local_operator"
    assert row["controller_name"] == "luci operator shell"
    assert row["controller_kind"] == "local"
    assert row["effective_status"] == "active"
    assert isinstance(row["allowed_envs"], list) and row["allowed_envs"]
    assert isinstance(row["allowed_routes"], list) and row["allowed_routes"]
    assert isinstance(row["allowed_commands"], list) and row["allowed_commands"]
    assert all(not str(cmd).startswith("./luci") for cmd in row["allowed_commands"])
    assert all(" " not in str(cmd) for cmd in row["allowed_commands"])
    assert row["max_parallel_threads"] >= 1
    assert row["max_spend"] >= 0
    assert row["receipt_uuid"] is None
    assert row["detail"]["bootstrap_local_only"] is True
    assert row["detail"]["budget_enforced"] is False
    assert row["expires_at"] is not None
    assert "packet" in row


def test_agent_thread_runtime_route_is_live_and_typed() -> None:
    with urllib.request.urlopen("http://127.0.0.1:3000/agent_thread_runtime?thread_key=eq.sub_orchestrator_thread", timeout=15) as resp:
        assert resp.status == 200
        payload = json.loads(resp.read().decode("utf-8"))

    assert isinstance(payload, list) and payload, payload
    row = payload[0]
    assert row["thread_key"] == "sub_orchestrator_thread"
    assert row["parent_thread_key"] == "root_operator_thread"
    assert row["controller_grant_key"] == "default_local_operator"
    assert row["thread_owner"] == "INDY_READs"
    assert row["runtime_kind"] == "codex"
    assert isinstance(row["model_policy"], dict)
    assert isinstance(row["env_identity"], dict)
    assert isinstance(row["budget_scope"], dict)
    assert isinstance(row["receipt_gate"], dict)
    assert row["receipt_gate"]["required_receipt"] == "control_grant_runtime_spine"
    assert row["receipt_uuid"] is None
    assert row["status"] == "active"
    assert "packet" in row


def test_controller_grant_effective_statuses_are_computed() -> None:
    suffix = uuid.uuid4().hex[:8]
    rows = [
        {
            "grant_key": f"test_active_{suffix}",
            "expires_at_sql": "now() + interval '1 day'",
            "revoked_at_sql": "NULL",
            "status": "active",
            "expected": "active",
        },
        {
            "grant_key": f"test_revoked_{suffix}",
            "expires_at_sql": "now() + interval '1 day'",
            "revoked_at_sql": "now()",
            "status": "active",
            "expected": "revoked",
        },
        {
            "grant_key": f"test_expired_{suffix}",
            "expires_at_sql": "now() - interval '1 day'",
            "revoked_at_sql": "NULL",
            "status": "active",
            "expected": "expired",
        },
    ]

    with psycopg.connect("postgresql:///lucidota_state", autocommit=False) as conn:
        with conn.cursor() as cur:
            try:
                for row in rows:
                    cur.execute(
                        f"""
                        INSERT INTO lucidota_control.controller_grant (
                            grant_key, controller_name, controller_kind, issued_by,
                            created_at, expires_at, revoked_at, revocation_reason,
                            status, allowed_envs, allowed_routes, allowed_commands,
                            allowed_models, max_parallel_threads, max_spend, receipt_uuid, detail
                        ) VALUES (
                            %s, %s, 'local', 'pytest',
                            now(), {row['expires_at_sql']}, {row['revoked_at_sql']}, '',
                            %s, ARRAY['pytest'], ARRAY['manual_current'], ARRAY['luci doctor --json'],
                            ARRAY['local'], 1, 0.00, NULL, jsonb_build_object('bootstrap_local_only', true)
                        )
                        """,
                        (
                            row["grant_key"],
                            f"pytest controller {row['grant_key']}",
                            row["status"],
                        ),
                    )
                conn.commit()

                for row in rows:
                    cur.execute(
                        """
                        SELECT effective_status
                        FROM lucidota_canon.controller_grant
                        WHERE grant_key = %s
                        """,
                        (row["grant_key"],),
                    )
                    assert cur.fetchone()[0] == row["expected"]
            finally:
                cur.execute(
                    "DELETE FROM lucidota_control.controller_grant WHERE grant_key LIKE %s",
                    (f"test_%_{suffix}",),
                )
                conn.commit()
