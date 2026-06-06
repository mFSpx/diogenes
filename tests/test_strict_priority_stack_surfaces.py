from __future__ import annotations

import json
import urllib.request

BASE = 'http://127.0.0.1:3000'

SURFACES = [
    ('skill_policy_current', 'policy_id'),
    ('chrono_current', None),
    ('payload_archive_status', None),
    ('todo_current', 'batch_uuid'),
    ('model_routing_blockers', 'routing_packet_id'),
]


def _fetch(surface: str):
    with urllib.request.urlopen(f'{BASE}/{surface}?limit=1', timeout=15) as resp:
        assert resp.status == 200
        return json.loads(resp.read().decode('utf-8'))


def test_current_surfaces_expose_strict_priority_stack():
    for surface, id_field in SURFACES:
        payload = _fetch(surface)
        assert isinstance(payload, list) and payload, surface
        row = payload[0]
        assert isinstance(row.get('orchestration'), dict), surface
        orch = row['orchestration']
        assert isinstance(orch.get('sub_orchestrator_priority'), list), surface
        assert orch['strict_priority_stack'][0] == 'live_truth_surfaces', surface
        if id_field:
            assert id_field in row, (surface, row.keys())
