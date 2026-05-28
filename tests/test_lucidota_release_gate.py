#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from lucidota_release_gate import run_gate, REQUIRED

def test_release_gate_runs_required_product_proofs():
    res=run_gate()
    assert res['status']=='PASSED', res['stdout']+res['stderr']
    assert 'tests/test_lucidota_acceptance.py' in res['required_tests']
    assert 'tests/test_demo_product_snapshot.py' in res['required_tests']
    assert 'tests/test_mutation_safety_oracle.py' in res['required_tests']
    assert 'tests/test_strict_model_stack_admission.py' in REQUIRED
    assert 'tests/test_graph_materialization_command_policy.py' in REQUIRED
    assert 'tests/test_graph_promotion_gate_safety.py' in REQUIRED
