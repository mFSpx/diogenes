# Codex Prompting Guide - LUCIDOTA Policy

- prefer_fast_search: Prefer rg/rg --files for search before slower shell scans.
- prefer_dedicated_tools: Use dedicated tools before raw shell when the tool exists.
- parallelize_independent_reads: Batch and parallelize independent reads/searches/tool calls.
- persist_to_verified_work: Gather context, implement, test, and refine end-to-end when feasible.
- suppress_upfront_preamble: Do not force upfront plans, preambles, or status chatter that can stop rollout.
- working_code_not_plan: Default to working code over plan-only output.
- root_cause_not_symptom: Cover the root cause/core ask, not only a narrow symptom.
- preserve_dirty_worktree: Never revert unrelated dirty worktree changes.
- use_apply_patch_for_edits: Use apply_patch for focused source edits when practical.
- plan_hygiene: Use update_plan for multi-step work and keep it current.
