#!/usr/bin/env python3
from __future__ import annotations
from typing import Any
from spine_common import now, sha256_json

def build_cep(*, raw_command: str, normalized_intent: str, authority_class: str='operator_authored_assertion', source: str='operator_cli', target_refs: list[str]|None=None, evidence_refs: list[str]|None=None, allowed_effect: str='create_case_pipeline') -> dict[str,Any]:
    if not raw_command.strip(): raise ValueError('raw_command required')
    cep={'raw_command':raw_command,'normalized_intent':normalized_intent,'authority_class':authority_class,'source':source,'target_refs':target_refs or [],'evidence_refs':evidence_refs or [],'allowed_effect':allowed_effect,'canonical_mutation_allowed':False,'created_at':now()}
    cep['command_id']='cep:'+sha256_json({k:v for k,v in cep.items() if k!='created_at'})[:24]
    return cep
