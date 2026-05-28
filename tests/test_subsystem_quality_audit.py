from scripts import subsystem_quality_audit as audit


def test_external_repo_reusable_prior_is_backlog_not_production_blocker() -> None:
    row = {
        "path": "01_REPOS/llama.cpp/scripts/server-test-model.py",
        "proof_hoard_role": "reusable_prior",
        "verdict": "KRAMPUSCHEW",
    }
    assert audit.is_proof_hoard_backlog(row) is True
    assert audit.production_blocking(row) is False


def test_active_runtime_non_promote_stays_production_blocking() -> None:
    row = {
        "path": "scripts/lucidota_start_strict_model_stack.sh",
        "proof_hoard_role": "active_runtime",
        "verdict": "REPAIR",
    }
    assert audit.is_proof_hoard_backlog(row) is False
    assert audit.production_blocking(row) is True


def test_generated_manifest_and_directory_entries_are_backlog_not_runtime_blockers() -> None:
    for path in ["scripts", "scripts/CORPSE_MANIFEST.jsonl", "scripts/KRAMPUSCHEWING_SCRIPT_CORPSES.jsonl"]:
        row = {"path": path, "verdict": "REPAIR"}
        assert audit.is_proof_hoard_backlog(row) is True
        assert audit.production_blocking(row) is False
