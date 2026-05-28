#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from case_dashboard_data import compile_dashboard_data

def test_case_dashboard_data_compiles_counts_blockers_and_refs(tmp_path):
    data=compile_dashboard_data(case_id='case', custody={'package':{'normal_count':2,'quarantine_count':1},'quarantine':[{'relative_path':'a.zip'}]}, timeline={'claim_count':2,'timeline':[{'t':1}]}, claims=[{'claim_id':'c1','cluster_id':'cl1'}], contradictions=[{'contradiction_id':'x','resolution_status':'OPEN'}], next_actions=[{'action_id':'a1'}], receipts=['r1'], output_path=tmp_path/'dashboard.json')
    assert data['counts']['quarantined_files']==1
    assert data['counts']['contradiction_count']==1
    assert data['blocked_items']['quarantine'][0]['relative_path']=='a.zip'
    assert data['claim_clusters']==['cl1']
    assert (tmp_path/'dashboard.json').exists()
