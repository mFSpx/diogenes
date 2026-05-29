-- FILE: 06_SCHEMA/064_queue_transition_law_v2.sql
-- PURPOSE: table-driven ABSURD queue transition law.

BEGIN;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lucidota_control;

CREATE TABLE IF NOT EXISTS lucidota_control.absurd_queue_transition_policy (
  old_status text NOT NULL,
  new_status text NOT NULL,
  actor_role text NOT NULL,
  allowed boolean NOT NULL DEFAULT true,
  reason text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY(old_status,new_status,actor_role)
);

DELETE FROM lucidota_control.absurd_queue_transition_policy;
INSERT INTO lucidota_control.absurd_queue_transition_policy(old_status,new_status,actor_role,reason)
SELECT old_status,new_status,actor_role,'queue transition law v2'
FROM (VALUES
  ('queued','leased'),('queued','running'),('queued','cancelled'),('queued','dead_lettered'),
  ('leased','running'),('leased','queued'),('leased','failed'),('leased','dead_lettered'),('leased','cancelled'),
  ('running','succeeded'),('running','failed'),('running','dead_lettered'),('running','queued'),('running','cancelled'),
  ('failed','queued'),('failed','dead_lettered'),('failed','cancelled'),
  ('succeeded','succeeded'),('dead_lettered','dead_lettered'),('cancelled','cancelled')
) AS s(old_status,new_status)
CROSS JOIN (VALUES ('foreman'),('worker'),('operator'),('system'),('auditor')) AS r(actor_role)
WHERE NOT (
  old_status IN ('succeeded','dead_lettered','cancelled') AND actor_role NOT IN ('auditor','operator')
) AND NOT (
  old_status='failed' AND actor_role NOT IN ('foreman','operator','system')
) AND NOT (
  old_status='running' AND actor_role NOT IN ('worker','foreman','operator','system')
) AND NOT (
  old_status IN ('queued','leased') AND actor_role NOT IN ('foreman','worker','operator','system')
);

CREATE OR REPLACE FUNCTION lucidota_control.absurd_queue_transition_allowed(
  old_status text,
  new_status text,
  actor_role text DEFAULT 'worker'
) RETURNS boolean
LANGUAGE sql
STABLE
AS $$
  SELECT old_status = new_status
         OR EXISTS (
           SELECT 1 FROM lucidota_control.absurd_queue_transition_policy p
           WHERE p.old_status=$1 AND p.new_status=$2 AND p.actor_role=$3 AND p.allowed
         );
$$;

COMMIT;
