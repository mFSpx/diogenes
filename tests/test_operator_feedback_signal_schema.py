from __future__ import annotations
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def test_operator_feedback_schema_closes_learning_loop() -> None:
    schema = (ROOT / '06_SCHEMA' / '116_operator_feedback_signal.sql').read_text(encoding='utf-8')
    assert 'operator_feedback_signal' in schema
    assert 'loss_delta integer GENERATED ALWAYS AS' in schema
    assert 'trg_absurd_dead_letter_feedback_signal' in schema
    assert 'trg_conversation_command_feedback_signal' in schema
    assert 'fn_train_operator_feedback_batch' in schema
    assert 'river_run' in schema


def test_operator_feedback_protocol_is_durable_runtime_blueprint() -> None:
    blueprint = (ROOT / 'BOOKS' / '5_TIER_MACHINE_CONSOLIDATION_BLUEPRINT.json').read_text(encoding='utf-8')
    extensions = (ROOT / 'BOOKS' / 'GO_EXTENSIONS.json').read_text(encoding='utf-8')
    assert 'closed_loop_operator_feedback_loss' in blueprint
    assert 'operator_feedback_signal_protocol' in extensions
    assert 'lucidota_learning.operator_feedback_signal' in extensions
