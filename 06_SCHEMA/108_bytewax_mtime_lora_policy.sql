-- Bytewax temporal-fragility runtime guard.
-- Any signal packet/event/hint tagged with mtime_snapshot_v1 gets a 256-token
-- LoRA/context ceiling and is routed toward Treelite date extraction instead of
-- deep DeepSeek context chewing. This DB trigger protects already-running
-- Bytewax loops without requiring a daemon restart.

CREATE SCHEMA IF NOT EXISTS lucidota_learning;

CREATE OR REPLACE FUNCTION lucidota_learning.bytewax_mtime_lora_policy_json()
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
AS $$
  SELECT jsonb_build_object(
    'schema', 'lucidota.bytewax.temporal_lora_policy.v1',
    'fragile_temporal_source', 'mtime_snapshot_v1',
    'lora_swap_token_threshold', 256,
    'max_context_tokens', 256,
    'deepseek_context_mode', 'lean_mtime_snapshot_fast_path',
    'date_extraction_router', 'treelite',
    'treelite_artifact', '03_VAULT/router/treelite_router_v0.tl',
    'vram_policy', 'do_not_expand_deep_context_for_low_trust_mtime_snapshot'
  )
$$;

CREATE OR REPLACE FUNCTION lucidota_learning.bytewax_apply_mtime_event_policy()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
  haystack text;
  policy jsonb;
BEGIN
  haystack := coalesce(NEW.text_surface, '') || ' ' || coalesce(NEW.payload::text, '') || ' ' || coalesce(NEW.certainty_trace::text, '');
  IF position('mtime_snapshot_v1' in haystack) > 0 THEN
    policy := lucidota_learning.bytewax_mtime_lora_policy_json();
    NEW.payload := coalesce(NEW.payload, '{}'::jsonb) || jsonb_build_object(
      'lora_swap_token_threshold', 256,
      'treelite_date_router_enabled', true,
      'temporal_lora_policy', policy
    );
    NEW.certainty_trace := coalesce(NEW.certainty_trace, '{}'::jsonb) || jsonb_build_object(
      'lora_swap_token_threshold', 256,
      'treelite_date_router_enabled', true,
      'temporal_lora_policy', policy
    );
  END IF;
  RETURN NEW;
END
$$;

DROP TRIGGER IF EXISTS trg_bytewax_mtime_event_policy ON lucidota_learning.bytewax_abductive_event;
CREATE TRIGGER trg_bytewax_mtime_event_policy
BEFORE INSERT OR UPDATE ON lucidota_learning.bytewax_abductive_event
FOR EACH ROW EXECUTE FUNCTION lucidota_learning.bytewax_apply_mtime_event_policy();

CREATE OR REPLACE FUNCTION lucidota_learning.bytewax_apply_mtime_hint_policy()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
  haystack text;
  policy jsonb;
BEGIN
  haystack := coalesce(NEW.hypothesis, '') || ' ' || coalesce(NEW.detail::text, '') || ' ' || coalesce(NEW.certainty_trace::text, '');
  IF position('mtime_snapshot_v1' in haystack) > 0 THEN
    policy := lucidota_learning.bytewax_mtime_lora_policy_json();
    NEW.detail := coalesce(NEW.detail, '{}'::jsonb) || jsonb_build_object(
      'lora_swap_token_threshold', 256,
      'treelite_date_router_enabled', true,
      'temporal_lora_policy', policy
    );
    NEW.certainty_trace := coalesce(NEW.certainty_trace, '{}'::jsonb) || jsonb_build_object(
      'lora_swap_token_threshold', 256,
      'treelite_date_router_enabled', true,
      'temporal_lora_policy', policy
    );
  END IF;
  RETURN NEW;
END
$$;

DROP TRIGGER IF EXISTS trg_bytewax_mtime_hint_policy ON lucidota_learning.bytewax_abductive_hint;
CREATE TRIGGER trg_bytewax_mtime_hint_policy
BEFORE INSERT OR UPDATE ON lucidota_learning.bytewax_abductive_hint
FOR EACH ROW EXECUTE FUNCTION lucidota_learning.bytewax_apply_mtime_hint_policy();
