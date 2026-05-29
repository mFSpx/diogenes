#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from spine_common import ROOT, append_jsonl, now, rel
LEDGER = ROOT / '05_OUTPUTS/kernel/control_packet_ledger.jsonl'

def packet_seen(packet_hash: str, ledger_path: str|Path = LEDGER) -> bool:
    p=Path(ledger_path)
    if not p.exists(): return False
    for line in p.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        try:
            if json.loads(line).get('packet_hash') == packet_hash:
                return True
        except json.JSONDecodeError:
            continue
    return False

def append_packet(packet: dict[str,Any], *, purpose: str, accepted: bool, denial_reason: str|None=None, ledger_path: str|Path=LEDGER) -> dict[str,Any]:
    row={'created_at':now(),'packet_hash':packet.get('packet_hash'),'authorized_by':packet.get('authorized_by'),'lane':packet.get('lane'),'purpose':purpose,'accepted':accepted,'denial_reason':denial_reason}
    append_jsonl(ledger_path,row); row['ledger_path']=rel(ledger_path); return row
