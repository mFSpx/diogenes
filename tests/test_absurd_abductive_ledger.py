from __future__ import annotations
import json
from pathlib import Path

def test_absurd_ledger_normalizes_model_audit_and_graph_candidates(tmp_path: Path) -> None:
    from scripts.absurd_abductive_ledger import normalize_receipt, summarize_rows
    audit = tmp_path / 'model_invocation_audit.json'
    audit.write_text(json.dumps({'schema':'lucidota.model_invocation_audit.v1','verdict':'FAIL','status':'FAIL','missing_dedicated_model_audit_blocks':1,'five_task_audit_blocks':[{'block_id':'task_block_0001','audit_status':'MISSING_VALID_AUDIT_OUTPUT'}]}), encoding='utf-8')
    hunch = tmp_path / 'hunch_postgres_ingest.json'
    hunch.write_text(json.dumps({'schema':'lucidota.hunch_postgres_ingest.receipt.v1','records_upserted':93,'graph_candidates_written':93,'canonical_graph_writes_performed':False}), encoding='utf-8')
    board = summarize_rows([normalize_receipt(audit), normalize_receipt(hunch)])
    assert board['schema'] == 'lucidota.absurd_abductive.board.v1'
    assert board['object_counts']['Receipt'] == 2
    assert board['object_counts']['ModelAuditBlock'] == 1
    assert board['object_counts']['GraphCandidate'] == 93
    assert any(b['class'] == 'model_audit' for b in board['open_blockers'])
    assert board['canonical_graph_writes'] is False
