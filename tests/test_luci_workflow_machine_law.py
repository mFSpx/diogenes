from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "06_SCHEMA" / "145_luci_workflow_machine_law.sql"
SEED = ROOT / "scripts" / "seed_luci_workflow_machine_law.py"
SOURCE = ROOT / "GOALS" / "ROOT_ROTOR_WORKFLOW_MACHINE_LLM_DOCTRINE_VERBATIM.md"


def test_workflow_machine_schema_extends_existing_registry_not_parallel_table() -> None:
    schema = SCHEMA.read_text(encoding="utf-8")
    assert "ALTER TABLE lucidota_control.workflow_registry" in schema
    assert "CREATE TABLE IF NOT EXISTS lucidota_control.workflow_registry" not in schema
    for column in [
        "workflow_id text",
        "verb text",
        "input_object_types text[]",
        "output_object_types text[]",
        "deterministic_first boolean",
        "llm_allowed boolean",
        "llm_required boolean",
        "allowed_models text[]",
        "validator_workflow_id text",
        "receipt_type text",
        "promotion_policy text",
        "llm_allowed_reasons text[]",
        "ontology_tags text[]",
    ]:
        assert column in schema


def test_workflow_machine_schema_enforces_deterministic_first_llm_boxing() -> None:
    schema = SCHEMA.read_text(encoding="utf-8")
    assert "workflow_registry_llm_required_requires_allowed" in schema
    assert "CHECK (NOT llm_required OR llm_allowed)" in schema
    assert "workflow_registry_no_llm_means_deterministic_first" in schema
    assert "CHECK (llm_allowed OR deterministic_first)" in schema
    assert "workflow_registry_llm_reason_check" in schema
    for reason in [
        "ambiguous_human_language",
        "messy_summarization",
        "entity_claim_extraction_judgment",
        "conflict_explanation",
        "hypothesis_generation",
        "prompt_dialogue_response",
        "code_design_review",
        "natural_language_transformation",
        "low_confidence_router_fallback",
        "human_facing_synthesis",
    ]:
        assert reason in schema
    assert "CREATE OR REPLACE VIEW lucidota_canon.api_workflow_registry" in schema


def test_workflow_machine_seed_maps_operator_doctrine_to_registry_and_bible_node() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "deterministic workflow OS with optional LLM judgment adapters" in source
    seed = SEED.read_text(encoding="utf-8")
    assert "ROOT_ROTOR_WORKFLOW_MACHINE_LLM_DOCTRINE_VERBATIM.md" in seed
    assert "5.900.0" in seed
    assert "root-rotor-canon-forge" in seed
    assert "root-rotor-apply-node-payloads" in seed
    assert "root-rotor-red-team-audit" in seed
    assert "proposal_until_validator_receipt" in seed
    assert "Jsonb" in seed
