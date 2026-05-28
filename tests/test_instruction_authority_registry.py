from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]

def test_instruction_authority_registry_required_layers_present():
    env={**os.environ, 'INSTRUCTION_HYGIENE_SKIP_TESTS':'1'}
    subprocess.run([sys.executable,'scripts/run_instruction_hygiene.py','--dry-run'],cwd=ROOT,text=True,capture_output=True,check=True,env=env)
    d=json.loads((ROOT/'00_PROJECT_BRAIN/instruction_authority_registry.json').read_text())
    assert d['schema']=='lucidota.instruction_authority_registry.v1'
    for layer in ['operator_current_instruction','00_PROJECT_BRAIN/STATUS_LEDGER.md','00_PROJECT_BRAIN/CANONICAL_FINISHED_PRODUCT_MAP.md','00_PROJECT_BRAIN/spine_authority_registry.json','00_PROJECT_BRAIN/canonical_graph_write_allowlist.json','06_SCHEMA','scripts_with_tests','archived_reference','legacy_storage']:
        assert layer in d['authority_order']
    assert '00_PROJECT_BRAIN/STATUS_LEDGER.md' in d['canonical_files']
    assert '00_PROJECT_BRAIN/spine_authority_registry.json' in d['canonical_files']
    assert '00_PROJECT_BRAIN/canonical_graph_write_allowlist.json' in d['canonical_files']
    assert '00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md' in d['canonical_files']
    assert any(law.get('law_key') == 'blueprint_first_model_second_pocketflow_hygiene' for law in d['active_laws'])

def test_active_instruction_index_lists_only_active_sources():
    text=(ROOT/'00_PROJECT_BRAIN/ACTIVE_INSTRUCTION_INDEX.md').read_text()
    assert '# Active Canonical Instruction Sources' in text
    assert 'STATUS_LEDGER' in text
    assert 'spine_authority_registry.json' in text
    assert 'canonical_graph_write_allowlist.json' in text
    assert 'Archived / Historical Instruction Sources' in text
    assert 'BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md' in text

def test_storage_manifest_preserves_restore_path():
    receipts=sorted((ROOT/'05_OUTPUTS/instruction_hygiene').glob('instruction_hygiene_*.json'), key=lambda p:p.stat().st_mtime)
    assert receipts
    receipt=json.loads(receipts[-1].read_text())
    manifest=ROOT/receipt['restore_manifest']
    d=json.loads(manifest.read_text())
    assert d['schema']=='lucidota.instruction_archive_manifest.v1'
    assert d.get('restore_manifest_has_restore_commands') is True
    for rec in d.get('files_moved',[])+d.get('files_quarantined',[])+d.get('proposed_moves',[]):
        assert rec.get('restore_command')
        assert rec.get('from') and rec.get('to')
