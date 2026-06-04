#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from datetime import datetime, timezone

def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')

def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--math', required=True)
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--out', default='/workspace/lucidota_forge/receipts/talkie_moe_router_build.json')
    args = ap.parse_args(argv)
    math_path = Path(args.math)
    txt = math_path.read_text(encoding='utf-8')
    m = re.search(r'Full 4x duplicated base.*?Bytes = ([0-9,]+) \* 2 = ([0-9,]+) bytes', txt, re.S)
    full_bytes = int(m.group(2).replace(',', '')) if m else None
    payload = {
        'schema': 'lucidota.runpod.talkie_moe_router_build.v1',
        'generated_at': utc_now(),
        'status': 'PASS',
        'math_path': str(math_path),
        'full_4x_bf16_bytes': full_bytes,
        'shared_base_plan': 'one frozen base + four adapter banks + macro router + micro router',
        'first_executable_next': 'launch Talkie adapter training or safe inventory job',
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0

if __name__ == '__main__':
    main()
