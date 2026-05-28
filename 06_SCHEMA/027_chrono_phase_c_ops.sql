-- FILE: 06_SCHEMA/027_chrono_phase_c_ops.sql
-- COMPONENT: LUCIDOTA Chrono-Ledger Phase C operational hardening
-- COMPLIANCE: Idempotent migration script; no destructive actions; no claim reset.
-- DEPENDENCIES: 06_SCHEMA/019_korpus_krampii.sql, 025_chrono_ledger_core.sql, 026_chrono_absurd_triggers.sql
-- PURPOSE: Durable replay cursor plus extractor dead-letter ledger for supervised daemon operation.

START TRANSACTION;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS lucidota_korpus.chrono_replay_cursor (
    cursor_name TEXT PRIMARY KEY,
    last_file_uuid UUID,
    last_file_first_seen_at TIMESTAMPTZ,
    processed_count BIGINT NOT NULL DEFAULT 0 CHECK (processed_count >= 0),
    last_replay_started_at TIMESTAMPTZ,
    last_replay_finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    detail JSONB NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE lucidota_korpus.chrono_replay_cursor IS
    'Durable high-water cursor for Chrono-Ledger missed-notification replay. LISTEN/NOTIFY is wakeup only; this cursor makes startup replay idempotent.';

INSERT INTO lucidota_korpus.chrono_replay_cursor(cursor_name)
VALUES ('chrono-ledger-daemon')
ON CONFLICT (cursor_name) DO NOTHING;

CREATE TABLE IF NOT EXISTS lucidota_korpus.chrono_dead_letter (
    dead_letter_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_uuid UUID REFERENCES lucidota_korpus.file_object(file_uuid) ON DELETE CASCADE,
    extractor_name TEXT NOT NULL,
    error_kind TEXT NOT NULL,
    error_message TEXT NOT NULL,
    context JSONB NOT NULL DEFAULT '{}'::jsonb,
    attempt_count INTEGER NOT NULL DEFAULT 1 CHECK (attempt_count >= 1),
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved BOOLEAN NOT NULL DEFAULT FALSE,
    resolved_at TIMESTAMPTZ,
    detail JSONB NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE lucidota_korpus.chrono_dead_letter IS
    'Retry-safe extractor failure ledger. Daemon records failures here and continues processing other files.';

CREATE TABLE IF NOT EXISTS lucidota_korpus.chrono_ranking_pass (
    ranking_pass_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pass_kind TEXT NOT NULL DEFAULT 'phase_c_verification',
    algorithm_version TEXT NOT NULL DEFAULT 'trust_weight_desc_timestamp_asc_created_desc_v1',
    source_view TEXT NOT NULL DEFAULT 'lucidota_korpus.resolved_chrono_timeline',
    total_files BIGINT NOT NULL DEFAULT 0 CHECK (total_files >= 0),
    total_claims BIGINT NOT NULL DEFAULT 0 CHECK (total_claims >= 0),
    changed_best_time_count BIGINT NOT NULL DEFAULT 0 CHECK (changed_best_time_count >= 0),
    disputed_files_count BIGINT NOT NULL DEFAULT 0 CHECK (disputed_files_count >= 0),
    ranking_violations BIGINT NOT NULL DEFAULT 0 CHECK (ranking_violations >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    detail JSONB NOT NULL DEFAULT '{}'::jsonb
);

COMMENT ON TABLE lucidota_korpus.chrono_ranking_pass IS
    'Append-only ranking pass ledger. Reranking creates a new pass; temporal_claim is archive, ranking results are projections.';

CREATE TABLE IF NOT EXISTS lucidota_korpus.chrono_ranking_result (
    ranking_result_uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ranking_pass_uuid UUID NOT NULL REFERENCES lucidota_korpus.chrono_ranking_pass(ranking_pass_uuid) ON DELETE CASCADE,
    file_uuid UUID NOT NULL REFERENCES lucidota_korpus.file_object(file_uuid) ON DELETE CASCADE,
    selected_claim_uuid UUID NOT NULL REFERENCES lucidota_korpus.temporal_claim(claim_uuid) ON DELETE RESTRICT,
    resolved_timestamp TIMESTAMPTZ NOT NULL,
    selected_evidence_source TEXT NOT NULL,
    selected_trust_weight NUMERIC(3, 2) NOT NULL,
    candidate_count INTEGER NOT NULL DEFAULT 0 CHECK (candidate_count >= 0),
    has_conflict BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    detail JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (ranking_pass_uuid, file_uuid)
);

COMMENT ON TABLE lucidota_korpus.chrono_ranking_result IS
    'Immutable per-file result for one chrono_ranking_pass. Every selected best time links back to temporal_claim.claim_uuid.';

CREATE UNIQUE INDEX IF NOT EXISTS idx_chrono_dead_letter_open_dedupe
    ON lucidota_korpus.chrono_dead_letter(file_uuid, extractor_name, error_kind)
    WHERE resolved = FALSE;

CREATE INDEX IF NOT EXISTS idx_chrono_dead_letter_file_uuid
    ON lucidota_korpus.chrono_dead_letter(file_uuid);

CREATE INDEX IF NOT EXISTS idx_chrono_dead_letter_unresolved
    ON lucidota_korpus.chrono_dead_letter(resolved, last_seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_chrono_replay_cursor_boundary
    ON lucidota_korpus.chrono_replay_cursor(last_file_first_seen_at, last_file_uuid);

CREATE INDEX IF NOT EXISTS idx_chrono_ranking_result_pass
    ON lucidota_korpus.chrono_ranking_result(ranking_pass_uuid);

CREATE INDEX IF NOT EXISTS idx_chrono_ranking_result_file
    ON lucidota_korpus.chrono_ranking_result(file_uuid);

CREATE INDEX IF NOT EXISTS idx_chrono_ranking_result_claim
    ON lucidota_korpus.chrono_ranking_result(selected_claim_uuid);

CREATE OR REPLACE FUNCTION lucidota_korpus.touch_chrono_replay_cursor_updated_at()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_touch_chrono_replay_cursor_updated_at'
          AND tgrelid = 'lucidota_korpus.chrono_replay_cursor'::regclass
    ) THEN
        CREATE TRIGGER trg_touch_chrono_replay_cursor_updated_at
        BEFORE UPDATE ON lucidota_korpus.chrono_replay_cursor
        FOR EACH ROW
        EXECUTE FUNCTION lucidota_korpus.touch_chrono_replay_cursor_updated_at();
    END IF;
END;
$$;

COMMIT;
