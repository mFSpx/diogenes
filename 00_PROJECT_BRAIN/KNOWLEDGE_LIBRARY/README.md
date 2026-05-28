# LUCIDOTA Knowledge Library

Purpose: durable, readable learning cards for external repos/papers/patterns that should stay accessible to future agents without turning every useful idea into hard law.

Authority classes:

- `pseudolaw`: standing hygiene guidance; always consider, but still implement with receipts.
- `research_reference`: learned pattern/library; mine for implementation ideas, tests, and comparison baselines.
- `candidate_tool`: cloned/local artifact that can be adapted only after a bounded integration plan.

Rules:

1. A knowledge item must have a source URL, local path or citation, summary, adoption stance, and verification receipt.
2. Knowledge cards do not override `STATUS_LEDGER`, graph write barriers, or operator instructions.
3. Promote to pseudolaw only when the rule changes system behavior globally.
4. Otherwise keep it here: learned, contemplated, searchable, and TICKLETRUNK-indexed.
5. Validate with `python3 scripts/knowledge_library_check.py --check`.
