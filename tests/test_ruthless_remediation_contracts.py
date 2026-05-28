#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    assert p.returncode==0, (cmd,p.returncode,p.stdout,p.stderr)
    return p.stdout
def receipt(stdout):
    for line in stdout.splitlines():
        if line.startswith('RECEIPT_PATH='):
            return ROOT/line.split('=',1)[1]
    raise AssertionError(stdout)
def test_krampus_inventory_fixture_idempotent():
    with tempfile.TemporaryDirectory() as td:
        d=Path(td); (d/'a.md').write_text('ENTITY has ATTRIBUTE.'); (d/'b.bin').write_bytes(b'123')
        out1=run([sys.executable,'scripts/krampus_bounded_inventory.py','--target',str(d),'--dry-run','--max-files','10','--max-bytes','8'])
        out2=run([sys.executable,'scripts/krampus_bounded_inventory.py','--target',str(d),'--dry-run','--max-files','10','--max-bytes','8'])
        r1=json.load(open(receipt(out1))); r2=json.load(open(receipt(out2)))
        assert r1['inventory_key']==r2['inventory_key']
        assert r1['counts']['emitted']==2
        assert Path(ROOT/r1['jsonl_output']).exists()
def test_ocr_router_classifies_text_and_pdf():
    with tempfile.TemporaryDirectory() as td:
        d=Path(td); txt=d/'note.md'; pdf=d/'paper.pdf'; txt.write_text('x'); pdf.write_bytes(b'%PDF-1.4')
        rt=json.load(open(receipt(run([sys.executable,'scripts/ocr_document_router.py','--path',str(txt)]))))
        rp=json.load(open(receipt(run([sys.executable,'scripts/ocr_document_router.py','--path',str(pdf)]))))
        assert rt['classification']['status']=='OCR_NOT_REQUIRED'
        assert rp['classification']['status'] in {'OCR_READY','OCR_BLOCKED'}
        assert rp['classification']['ocr_executed'] is False
def test_ontology_staging_never_promotes():
    with tempfile.TemporaryDirectory() as td:
        f=Path(td)/'claim.md'; f.write_text('ENTITY has ATTRIBUTE. Evidence supports CLAIM. Event happens over TIME.')
        r=json.load(open(receipt(run([sys.executable,'scripts/ontology_staging_contract.py','--source-file',str(f)]))))
        assert r['status']=='STAGED'
        assert r['direct_truth_promotion_performed'] is False
        assert r['promotion_blocker']
def test_absurd_corpus_bridge_dry_run_contract():
    out=run([sys.executable,'scripts/absurd_corpus_job_bridge.py','--source-path','README.md','--lane','manifest_inventory','--dry-run'])
    r=json.load(open(receipt(out)))
    assert r['execute_performed'] is False
    assert r['idempotency_key']
    assert r['queue']=='korpus'
if __name__=='__main__':
    for name,fn in list(globals().items()):
        if name.startswith('test_'):
            fn(); print('PASS',name)
