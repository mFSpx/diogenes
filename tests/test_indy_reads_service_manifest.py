from __future__ import annotations

import json
from pathlib import Path


def test_indy_reads_service_manifest_points_to_boot_packet_and_live_truth_surfaces() -> None:
    manifest = json.loads(Path("04_RUNTIME/INDY_READS/indy_reads_service_manifest.json").read_text(encoding="utf-8"))

    assert manifest["owner_persona"] == "INDY_READS"
    assert manifest["boot"]["startup_manifest"] == "04_RUNTIME/indy_reads_startup_comms_manifest.json"
    assert manifest["boot"]["boot_packet"] == "04_RUNTIME/indy_reads_boot_packet.json"
    assert manifest["assistance_route"]["adapter_runtime"] == "bonsai_q1_0"
    assert manifest["assistance_route"]["resident_sprawl_policy"] == "minimal_context_only"
    assert manifest["truth_refs"] == [
        "lucidota_control.active_operation_mode",
        "lucidota_canon.manual_current",
        "lucidota_canon.root_orchestrator_current",
        "lucidota_canon.workload_audit_current",
        "lucidota_canon.workload_audit_telemetry_current",
    ]
