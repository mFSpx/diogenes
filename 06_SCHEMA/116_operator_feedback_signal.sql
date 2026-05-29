-- Closed-loop operator correction loss bridge for Bytewax/RiverML.
-- Applies to lucidota_state. component_uuid is a custody pointer; storage FK is impossible across DBs.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_learning;

CREATE TABLE IF NOT EXISTS lucidota_learning.operator_feedback_signal (
    signal_uuid uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    component_uuid uuid,
    predicted_valency integer NOT NULL CHECK (predicted_valency IN (1, 0, -1)),
    corrected_valency integer NOT NULL CHECK (corrected_valency IN (1, 0, -1)),
    loss_delta integer GENERATED ALWAYS AS (corrected_valency - predicted_valency) STORED,
    correction_context text NOT NULL,
    source_table text NOT NULL DEFAULT '',
    source_uuid uuid,
    source_ref text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    consumed_at timestamptz,
    river_run_id uuid,
    logged_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(source_ref, correction_context)
);

CREATE INDEX IF NOT EXISTS operator_feedback_signal_unconsumed_idx
    ON lucidota_learning.operator_feedback_signal(consumed_at, logged_at DESC)
    WHERE consumed_at IS NULL;

CREATE INDEX IF NOT EXISTS operator_feedback_signal_loss_idx
    ON lucidota_learning.operator_feedback_signal(loss_delta, logged_at DESC);

CREATE OR REPLACE FUNCTION lucidota_learning.fn_log_operator_feedback_signal(
    p_component_uuid uuid,
    p_predicted_valency integer,
    p_corrected_valency integer,
    p_correction_context text,
    p_source_table text,
    p_source_uuid uuid,
    p_source_ref text,
    p_detail jsonb DEFAULT '{}'::jsonb
)
RETURNS uuid
LANGUAGE plpgsql
AS $$
DECLARE
    out_uuid uuid;
BEGIN
    IF p_predicted_valency NOT IN (-1,0,1) OR p_corrected_valency NOT IN (-1,0,1) THEN
        RAISE EXCEPTION 'valencies must be -1, 0, or 1';
    END IF;

    INSERT INTO lucidota_learning.operator_feedback_signal(
        component_uuid, predicted_valency, corrected_valency, correction_context,
        source_table, source_uuid, source_ref, detail
    ) VALUES (
        p_component_uuid, p_predicted_valency, p_corrected_valency, p_correction_context,
        coalesce(p_source_table, ''), p_source_uuid, coalesce(nullif(p_source_ref, ''), coalesce(p_source_table, '') || ':' || coalesce(p_source_uuid::text, gen_random_uuid()::text)), coalesce(p_detail, '{}'::jsonb)
    )
    ON CONFLICT(source_ref, correction_context) DO UPDATE SET
        predicted_valency = EXCLUDED.predicted_valency,
        corrected_valency = EXCLUDED.corrected_valency,
        component_uuid = EXCLUDED.component_uuid,
        detail = lucidota_learning.operator_feedback_signal.detail || EXCLUDED.detail,
        consumed_at = NULL
    RETURNING signal_uuid INTO out_uuid;

    RETURN out_uuid;
END;
$$;

CREATE OR REPLACE FUNCTION lucidota_learning.fn_absurd_dead_letter_feedback_signal()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    PERFORM lucidota_learning.fn_log_operator_feedback_signal(
        NULL,
        1,
        -1,
        'absurd_dead_letter_loss',
        'lucidota_control.absurd_queue_dead_letter',
        NEW.dead_letter_uuid,
        'absurd_queue_dead_letter:' || NEW.dead_letter_uuid::text,
        jsonb_build_object(
            'queue_name', NEW.queue_name,
            'workflow_name', NEW.workflow_name,
            'job_kind', NEW.job_kind,
            'error_kind', NEW.error_kind,
            'payload_sha256', NEW.payload_sha256,
            'context', NEW.context
        )
    );
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_absurd_dead_letter_feedback_signal ON lucidota_control.absurd_queue_dead_letter;
CREATE TRIGGER trg_absurd_dead_letter_feedback_signal
AFTER INSERT OR UPDATE OF error_kind, error_message, resolved
ON lucidota_control.absurd_queue_dead_letter
FOR EACH ROW
EXECUTE FUNCTION lucidota_learning.fn_absurd_dead_letter_feedback_signal();

CREATE OR REPLACE FUNCTION lucidota_learning.fn_conversation_command_feedback_signal()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.allowed_effect = 'operator_override'
       OR NEW.authority_class LIKE 'operator_%'
       OR NEW.status IN ('accepted','executed') THEN
        PERFORM lucidota_learning.fn_log_operator_feedback_signal(
            NULL,
            0,
            CASE WHEN NEW.status IN ('rejected') THEN -1 ELSE 1 END,
            'conversation_command_operator_override',
            'lucidota_control.conversation_command',
            NEW.command_uuid,
            'conversation_command:' || NEW.command_uuid::text,
            jsonb_build_object(
                'command_kind', NEW.command_kind,
                'allowed_effect', NEW.allowed_effect,
                'authority_class', NEW.authority_class,
                'status', NEW.status,
                'target_refs', NEW.target_refs,
                'evidence_refs', NEW.evidence_refs
            )
        );
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_conversation_command_feedback_signal ON lucidota_control.conversation_command;
CREATE TRIGGER trg_conversation_command_feedback_signal
AFTER INSERT OR UPDATE OF status, allowed_effect, authority_class
ON lucidota_control.conversation_command
FOR EACH ROW
EXECUTE FUNCTION lucidota_learning.fn_conversation_command_feedback_signal();

CREATE OR REPLACE FUNCTION lucidota_learning.fn_train_operator_feedback_batch(max_rows integer DEFAULT 500)
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
    run_uuid uuid;
    n integer;
    loss_sum integer;
BEGIN
    WITH batch AS (
        SELECT signal_uuid, loss_delta
        FROM lucidota_learning.operator_feedback_signal
        WHERE consumed_at IS NULL
        ORDER BY logged_at ASC, signal_uuid ASC
        LIMIT greatest(1, least(coalesce(max_rows, 500), 5000))
    ), agg AS (
        SELECT count(*)::integer AS n, coalesce(sum(loss_delta), 0)::integer AS loss_sum
        FROM batch
    )
    SELECT agg.n, agg.loss_sum INTO n, loss_sum FROM agg;

    IF coalesce(n, 0) = 0 THEN
        RETURN jsonb_build_object('status', 'no_feedback_signals', 'events_seen', 0, 'examples_trained', 0, 'loss_delta_sum', 0);
    END IF;

    INSERT INTO lucidota_learning.river_run(status, events_seen, examples_trained, detail)
    VALUES (
        'succeeded',
        n,
        n,
        jsonb_build_object(
            'training_kind', 'operator_feedback_loss',
            'loss_delta_sum', loss_sum,
            'source_table', 'lucidota_learning.operator_feedback_signal'
        )
    )
    RETURNING run_id INTO run_uuid;

    WITH batch AS (
        SELECT signal_uuid
        FROM lucidota_learning.operator_feedback_signal
        WHERE consumed_at IS NULL
        ORDER BY logged_at ASC, signal_uuid ASC
        LIMIT greatest(1, least(coalesce(max_rows, 500), 5000))
    )
    UPDATE lucidota_learning.operator_feedback_signal s
    SET consumed_at = now(), river_run_id = run_uuid
    FROM batch b
    WHERE s.signal_uuid = b.signal_uuid;

    RETURN jsonb_build_object('status', 'trained', 'river_run_id', run_uuid, 'events_seen', n, 'examples_trained', n, 'loss_delta_sum', loss_sum);
END;
$$;

INSERT INTO lucidota_learning.operator_feedback_signal(
    component_uuid, predicted_valency, corrected_valency, correction_context,
    source_table, source_uuid, source_ref, detail
)
SELECT NULL, 1, -1, 'absurd_dead_letter_loss',
       'lucidota_control.absurd_queue_dead_letter', dl.dead_letter_uuid,
       'absurd_queue_dead_letter:' || dl.dead_letter_uuid::text,
       jsonb_build_object('queue_name', dl.queue_name, 'workflow_name', dl.workflow_name, 'job_kind', dl.job_kind, 'error_kind', dl.error_kind, 'payload_sha256', dl.payload_sha256, 'context', dl.context)
FROM lucidota_control.absurd_queue_dead_letter dl
ON CONFLICT(source_ref, correction_context) DO NOTHING;

INSERT INTO lucidota_learning.operator_feedback_signal(
    component_uuid, predicted_valency, corrected_valency, correction_context,
    source_table, source_uuid, source_ref, detail
)
SELECT NULL, 0, CASE WHEN cc.status='rejected' THEN -1 ELSE 1 END, 'conversation_command_operator_override',
       'lucidota_control.conversation_command', cc.command_uuid,
       'conversation_command:' || cc.command_uuid::text,
       jsonb_build_object('command_kind', cc.command_kind, 'allowed_effect', cc.allowed_effect, 'authority_class', cc.authority_class, 'status', cc.status, 'target_refs', cc.target_refs, 'evidence_refs', cc.evidence_refs)
FROM lucidota_control.conversation_command cc
WHERE cc.allowed_effect = 'operator_override'
   OR cc.authority_class LIKE 'operator_%'
   OR cc.status IN ('accepted','executed','rejected')
ON CONFLICT(source_ref, correction_context) DO NOTHING;
