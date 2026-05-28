import json
from pathlib import Path

from scripts.villager_status import build_status


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_villager_status_distinguishes_heartbeat_cycles_from_villagers(tmp_path):
    write_json(
        tmp_path / "05_OUTPUTS/diogenes/percyphon_kernel_bridge_a.json",
        {
            "status": "ROUTED",
            "generated_at": "2026-05-26T19:25:09Z",
            "source": "operator_cli",
            "authority_class": "operator_authored_assertion",
            "canonical_graph_writes_performed": False,
            "model_calls_performed": False,
            "blockers": [],
            "percyphon": {
                "authority": "procedural_scaffold_candidate_not_truth",
                "zero_vram": True,
                "slot_count": 12,
                "fluid_slot_count": 8,
                "slot": {
                    "name": "Villager-2921",
                    "alias": "Alias-a9ce",
                    "persona": "archivist",
                    "slot_index": 0,
                    "ternary_offset": 0,
                    "uuid": "5a4ce1a9-ce7b-0f89-625f-438feb085768",
                },
            },
        },
    )
    hb = tmp_path / "04_RUNTIME/fairyfuse/ternary_router_heartbeat.jsonl"
    hb.parent.mkdir(parents=True, exist_ok=True)
    hb.write_text(
        json.dumps(
            {
                "created_at": "2026-05-20T02:54:21Z",
                "cycle": 3403,
                "engine_channel": "cpu_fairyfuse_ternary",
                "status": {
                    "backend": "FAIRYFUSE_V1_RESIDENT_TERNARY",
                    "mode": "symbolic_no_weights",
                    "weights": {"mapped": False, "status": "missing_or_deferred"},
                    "load_errors": ["weights_missing:x"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    status = build_status(tmp_path, ["3403"])

    target = status["targets"]["3403"]
    assert target["status"] == "NO_VERIFIED_VILLAGER_RECORD"
    assert target["fairyfuse_cycle_matches"][0]["cycle"] == 3403
    assert target["fairyfuse_cycle_matches"][0]["mode"] == "symbolic_no_weights"
    assert status["village"]["routed_count"] == 1
    assert status["village"]["slots"][0]["name"] == "Villager-2921"
    assert status["canonical_graph_writes_performed"] is False
    assert status["model_calls_performed"] is False


def test_villager_status_reports_partial_villager_identifier_match(tmp_path):
    write_json(
        tmp_path / "05_OUTPUTS/diogenes/percyphon_kernel_bridge_b.json",
        {
            "status": "ROUTED",
            "generated_at": "2026-05-26T19:00:00Z",
            "source": "operator_cli",
            "authority_class": "operator_authored_assertion",
            "canonical_graph_writes_performed": False,
            "model_calls_performed": False,
            "blockers": [],
            "percyphon": {
                "authority": "procedural_scaffold_candidate_not_truth",
                "zero_vram": True,
                "slot": {
                    "name": "Villager-1234",
                    "alias": "Alias-test",
                    "persona": "witness",
                    "slot_index": 3,
                    "ternary_offset": -1,
                    "uuid": "abcd1234-0000-0000-0000-000000000000",
                },
            },
        },
    )

    status = build_status(tmp_path, ["1234"])

    target = status["targets"]["1234"]
    assert target["status"] == "VILLAGER_RECORD_FOUND"
    assert {m["match_reason"] for m in target["villager_matches"]} == {"slot_name_suffix", "slot_uuid_contains"}
