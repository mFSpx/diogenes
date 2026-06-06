from __future__ import annotations

from pathlib import Path
from uuid import UUID

import scripts.prompt_ledger_capture as plc


def test_build_payload_links_work_order_when_explicit_uuid_is_provided(tmp_path):
    path = tmp_path / 'prompt.md'
    path.write_text('hello world\n', encoding='utf-8')

    payload = plc.build_payload(
        path=path,
        work_order_uuid=UUID('58465be6-9ecb-4f71-b86d-e3641c52d2d8'),
    )

    assert payload['linked_work_order_uuid'] == ['58465be6-9ecb-4f71-b86d-e3641c52d2d8']
    assert payload['blockers'] == ''
    assert payload['notes']


def test_build_payload_supports_raw_text_capture():
    payload = plc.build_payload(
        text='Save This Prompt, Pass on this Handoff:\nraw prompt body',
        source='operator',
        source_model='manual',
        conversation_session_id='cli-prompt',
        source_path='stdin://prompt',
        work_order_uuid=UUID('58465be6-9ecb-4f71-b86d-e3641c52d2d8'),
    )

    assert payload['source'] == 'operator'
    assert payload['source_model'] == 'manual'
    assert payload['conversation_session_id'] == 'cli-prompt'
    assert payload['source_path'] == 'stdin://prompt'
    assert payload['linked_work_order_uuid'] == ['58465be6-9ecb-4f71-b86d-e3641c52d2d8']
    assert 'raw prompt body' in payload['raw_prompt_text']


def test_build_payload_accepts_explicit_ontology_tags():
    payload = plc.build_payload(
        text='ontology tagged prompt',
        source='operator',
        source_model='manual',
        conversation_session_id='explicit-tags',
        source_path='stdin://prompt',
        ontology_tags=['SYSTEMIC_SWARM_HARDEN_V050', 'CORES', 'SPINE', 'NETWORKING'],
        subsystem_tags=['prompt-ledger', 'control-plane'],
        work_order_uuid=UUID('58465be6-9ecb-4f71-b86d-e3641c52d2d8'),
    )

    assert payload['ontology_tags'] == ['SYSTEMIC_SWARM_HARDEN_V050', 'CORES', 'SPINE', 'NETWORKING']
    assert payload['subsystem_tags'] == ['prompt-ledger', 'control-plane']


def test_build_payload_uses_message_uuid_for_raw_stdin_identity():
    payload = plc.build_payload(
        text='unique message body',
        source='operator',
        source_model='manual',
        message_uuid='11111111-1111-4111-8111-111111111111',
        source_path='stdin://prompt',
        work_order_uuid=UUID('58465be6-9ecb-4f71-b86d-e3641c52d2d8'),
    )

    assert payload['conversation_session_id'] == '11111111-1111-4111-8111-111111111111'


def test_build_payload_idempotency_depends_on_message_uuid():
    base = dict(
        text='same raw body',
        source='operator',
        source_model='manual',
        source_path='stdin://prompt',
        work_order_uuid=UUID('58465be6-9ecb-4f71-b86d-e3641c52d2d8'),
        ontology_tags=['SYSTEMIC_SWARM_HARDEN_V050'],
        subsystem_tags=['prompt-ledger'],
    )
    a = plc.build_payload(message_uuid='11111111-1111-4111-8111-111111111111', **base)
    b = plc.build_payload(message_uuid='22222222-2222-4222-8222-222222222222', **base)

    assert a['idempotency_key'] != b['idempotency_key']


def test_build_payload_refuses_internal_state_sources():
    payload_text = "<codex_internal_context source=\"goal\">\nself read"
    try:
        plc.build_payload(
            text=payload_text,
            source='operator',
            source_model='manual',
            source_path='GOALS/CURRENT_HANDOFF.md',
            work_order_uuid=UUID('58465be6-9ecb-4f71-b86d-e3641c52d2d8'),
        )
    except ValueError as exc:
        assert 'internal state source' in str(exc)
    else:
        raise AssertionError('expected ValueError for internal state source')


def test_discover_sources_skips_goal_loop_files(tmp_path):
    goals = tmp_path / 'GOALS'
    goals.mkdir()
    (goals / 'CURRENT_HANDOFF.md').write_text('Save This Prompt, Pass on this Handoff:\nloop', encoding='utf-8')
    (goals / 'GOAL_LOG.md').write_text('Save This Prompt, Pass on this Handoff:\nloop', encoding='utf-8')
    (goals / 'GOAL_PROMPTS.md').write_text('operator prompt', encoding='utf-8')

    sources = plc.discover_sources([goals / 'CURRENT_HANDOFF.md', goals / 'GOAL_LOG.md', goals / 'GOAL_PROMPTS.md'])

    assert sources == [goals / 'GOAL_PROMPTS.md']
