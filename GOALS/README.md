# GOALS

Crash-recovery notes for active goal work.

Rules: read CURRENT_HANDOFF at goal start; update X/N after every step; append concise handoffs to GOAL_LOG; keep final goal step prefixed with "Save This Prompt, Pass on this Handoff:".

RAC truth rule: before claiming work in a build/race session, read `lucidota_control.active_operation_mode`, `manual_current`, `root_orchestrator_current`, and `workload_audit_current`; GOALS is intent/handoff, Postgres/PostgREST is live truth, and every claim must have a receipt row or UNKNOWN debt.

Yap Trap: this is where bot-yap gets compressed, not expanded. Current handoff should stay tiny: goal, step, completed, next action, resume command, evidence. No essays.

Files: GOAL_HANDOFF_PROMPT.md, CURRENT_HANDOFF.md, GOAL_LOG.md, GOAL_PROMPTS.md, AGENT_ORCHESTRATION_POLICY.md, EXTERNAL_PLUGIN_BUILD_MODE.md, MODEL_FABRIC_AUDIT.md, FOSS_REUSE_AUDIT.md, plugin_build_mode_bootstrap.json, DEMO_25_STEP_LOG.md, ARCHITECTURE_AUDIT.md. No nested folder sprawl.
