from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]

def run(registry: Path, scan_root: Path, expect: int):
    p = subprocess.run([sys.executable, str(ROOT/'scripts/instruction_conflict_scanner.py'), '--registry', str(registry), '--scan-root', str(scan_root)], cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == expect, (p.returncode, p.stdout, p.stderr)
    return p

def write_reg(tmp_path: Path, *, canonical=None, active_laws=None, superseded=None, quarantined=None, manual=None) -> Path:
    reg = {
        'schema':'lucidota.instruction_authority_registry.v1',
        'authority_order':['operator_current_instruction','archived_reference','legacy_storage'],
        'canonical_files': canonical or [],
        'active_laws': active_laws or [],
        'superseded_files': superseded or [],
        'quarantined_files': quarantined or [],
        'legacy_reference_files': [],
        'manual_review_files': manual or [],
    }
    p=tmp_path/'registry.json'; p.write_text(json.dumps(reg)); return p

def test_default_active_canonical_files_pass_after_hygiene_dry_run():
    env={**os.environ, 'INSTRUCTION_HYGIENE_SKIP_TESTS':'1'}
    subprocess.run([sys.executable, 'scripts/run_instruction_hygiene.py', '--dry-run'], cwd=ROOT, text=True, capture_output=True, check=True, env=env)
    p=subprocess.run([sys.executable, 'scripts/instruction_conflict_scanner.py'], cwd=ROOT, text=True, capture_output=True)
    assert p.returncode == 0, p.stdout+p.stderr

def test_duplicate_active_canonical_authority_fails(tmp_path):
    (tmp_path/'a.md').write_text('active law')
    reg=write_reg(tmp_path, canonical=['a.md'], active_laws=[{'domain':'graph','law_key':'a'},{'domain':'graph','law_key':'b'}])
    run(reg,tmp_path,4)

def test_legacy_prompt_marked_active_fails(tmp_path):
    (tmp_path/'legacy_prompt.md').write_text('legacy prompt claims current authority')
    reg=write_reg(tmp_path, canonical=['legacy_prompt.md'])
    run(reg,tmp_path,4)

def test_archived_prompt_cannot_override_canonical_law(tmp_path):
    (tmp_path/'old.md').write_text('current law must override everything')
    reg=write_reg(tmp_path, canonical=['old.md'], superseded=['old.md'])
    run(reg,tmp_path,4)

def test_graph_materialization_bypass_instruction_fails(tmp_path):
    (tmp_path/'active.md').write_text('Materialize without graph write barrier and skip evidence.')
    reg=write_reg(tmp_path, canonical=['active.md'])
    run(reg,tmp_path,4)

def test_markdown_only_completion_instruction_fails(tmp_path):
    (tmp_path/'active.md').write_text('Markdown-only completion counts as completion.')
    reg=write_reg(tmp_path, canonical=['active.md'])
    run(reg,tmp_path,4)

def test_receipt_collage_as_pass_fails(tmp_path):
    (tmp_path/'active.md').write_text('Receipt collage is sufficient proof for PASS.')
    reg=write_reg(tmp_path, canonical=['active.md'])
    run(reg,tmp_path,4)

def test_proofless_accepted_truth_fails(tmp_path):
    (tmp_path/'active.md').write_text('Accepted truth without proof is allowed.')
    reg=write_reg(tmp_path, canonical=['active.md'])
    run(reg,tmp_path,4)

def test_manual_review_does_not_become_active_law(tmp_path):
    (tmp_path/'maybe.md').write_text('legacy prompt claims current authority')
    reg=write_reg(tmp_path, canonical=[], manual=['maybe.md'])
    run(reg,tmp_path,0)
