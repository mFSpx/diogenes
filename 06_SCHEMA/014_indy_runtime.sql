-- Indy_Reads runtime support tables.
-- Local, operator-supervised memory and quiet queue only; no ambient inbox/Drive reads.

CREATE SCHEMA IF NOT EXISTS lucidota_indy;

CREATE TABLE IF NOT EXISTS lucidota_indy.task_memory (
    memory_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kind text NOT NULL CHECK (kind IN ('task','decision','correction','note')),
    title text NOT NULL,
    body text NOT NULL DEFAULT '',
    source text NOT NULL DEFAULT 'operator',
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','done','archived')),
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS indy_task_memory_status_idx
    ON lucidota_indy.task_memory (status, created_at DESC);

CREATE TABLE IF NOT EXISTS lucidota_indy.side_queue (
    queue_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    item_type text NOT NULL CHECK (item_type IN ('calendar_intent','reminder','wiki_note','auth_inventory','review')),
    title text NOT NULL,
    body text NOT NULL DEFAULT '',
    urgency text NOT NULL DEFAULT 'normal' CHECK (urgency IN ('low','normal','high')),
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','reviewing','done','dismissed')),
    due_at timestamptz,
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS indy_side_queue_status_idx
    ON lucidota_indy.side_queue (status, urgency, created_at DESC);
