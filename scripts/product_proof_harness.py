#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from lucidota_pipeline import LucidotaPipeline
from spine_common import rel

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('source_folder'); ap.add_argument('--case-id'); ap.add_argument('--output-root'); ap.add_argument('--max-files',type=int,default=100); a=ap.parse_args()
    run=LucidotaPipeline(case_id=a.case_id, output_root=a.output_root).run_fixture_pipeline(a.source_folder, max_files=a.max_files)
    print(json.dumps({'case_id':run.case_id,'output_dir':rel(run.output_dir),'failed_stage':run.failed_stage,'stages':[s.stage_name for s in run.stages]},sort_keys=True)); return 0 if not run.failed_stage else 2
if __name__=='__main__': raise SystemExit(main())
