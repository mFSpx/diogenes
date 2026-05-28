#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

class ContractError(ValueError):
    pass

@dataclass(frozen=True)
class StageContract:
    stage_name: str
    component_name: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    receipt_type: str

    def validate_input(self, payload: dict[str, Any]) -> None:
        validate_object(payload, self.input_schema, f'{self.stage_name}.input')

    def validate_output(self, payload: dict[str, Any]) -> None:
        validate_object(payload, self.output_schema, f'{self.stage_name}.output')


def _type_ok(value: Any, expected: str) -> bool:
    if expected == 'string': return isinstance(value, str)
    if expected == 'integer': return isinstance(value, int) and not isinstance(value, bool)
    if expected == 'number': return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == 'boolean': return isinstance(value, bool)
    if expected == 'object': return isinstance(value, dict)
    if expected == 'array': return isinstance(value, list)
    if expected == 'null': return value is None
    return True


def validate_object(payload: dict[str, Any], schema: dict[str, Any], label: str='payload') -> None:
    if not isinstance(payload, dict):
        raise ContractError(f'{label} must be object')
    for key in schema.get('required', []):
        if key not in payload:
            raise ContractError(f'{label} missing required key: {key}')
    props = schema.get('properties', {})
    for key, rule in props.items():
        if key not in payload: continue
        allowed = rule.get('enum')
        if allowed is not None and payload[key] not in allowed:
            raise ContractError(f'{label}.{key} not in enum {allowed}')
        typ = rule.get('type')
        if isinstance(typ, list):
            if not any(_type_ok(payload[key], t) for t in typ):
                raise ContractError(f'{label}.{key} wrong type')
        elif isinstance(typ, str) and not _type_ok(payload[key], typ):
            raise ContractError(f'{label}.{key} wrong type: expected {typ}')
        if rule.get('minLength') is not None and isinstance(payload[key], str) and len(payload[key]) < int(rule['minLength']):
            raise ContractError(f'{label}.{key} shorter than minLength')


def stage_contracts() -> dict[str, StageContract]:
    path_str = {'type':'string','minLength':1}
    obj = {'type':'object'}
    return {
        'intake': StageContract('intake','KRAMPUS custody',{'required':['source_folder'], 'properties':{'source_folder':path_str,'max_files':{'type':'integer'}}},{'required':['package','receipt_path'], 'properties':{'package':obj,'receipt_path':path_str}},'product_intake'),
        'parse': StageContract('parse','document parse',{'required':['package_path','source_root'], 'properties':{'package_path':path_str,'source_root':path_str}},{'required':['package','receipt_path'], 'properties':{'package':obj,'receipt_path':path_str}},'product_parse_pipeline'),
        'timeline': StageContract('timeline','Chrono timeline',{'required':['package_path','source_root'], 'properties':{'package_path':path_str,'source_root':path_str}},{'required':['timeline','receipt_path'], 'properties':{'timeline':{'type':'array'},'receipt_path':path_str}},'product_timeline'),
        'staging': StageContract('staging','ontology staging',{'required':['chunks_path'], 'properties':{'chunks_path':path_str}},{'required':['staging_path','claim_count','receipt_path'], 'properties':{'staging_path':path_str,'claim_count':{'type':'integer'},'receipt_path':path_str}},'chunk_to_staging'),
        'graph_candidate': StageContract('graph_candidate','graph candidate',{'required':['staging_path'], 'properties':{'staging_path':path_str}},{'required':['graph_candidates_path','candidate_count','receipt_path'], 'properties':{'graph_candidates_path':path_str,'candidate_count':{'type':'integer'},'receipt_path':path_str}},'graph_candidate'),
        'case_packet': StageContract('case_packet','case packet',{'required':['case_id','custody','parse','staging','graph'], 'properties':{'case_id':path_str,'custody':obj,'parse':obj,'timeline':obj,'staging':obj,'graph':obj}},{'required':['case_packet_path','receipt_path'], 'properties':{'case_packet_path':path_str,'receipt_path':path_str}},'case_packet'),
    }


def get_contract(stage_name: str) -> StageContract:
    contracts = stage_contracts()
    if stage_name not in contracts:
        raise ContractError(f'unknown pipeline stage: {stage_name}')
    return contracts[stage_name]
