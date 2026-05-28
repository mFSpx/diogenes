#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from pipeline_contracts import get_contract, ContractError

def test_stage_contract_rejects_bad_input_before_execution():
    c=get_contract('parse')
    try:
        c.validate_input({'package_path':'x'})
    except ContractError as e:
        assert 'source_root' in str(e)
    else:
        raise AssertionError('bad parse input accepted')

def test_stage_contract_accepts_valid_input_and_output():
    c=get_contract('staging')
    c.validate_input({'chunks_path':'chunks.jsonl'})
    c.validate_output({'staging_path':'staging.jsonl','claim_count':1,'receipt_path':'receipt.json'})
