# Blueprint-First / PocketFlow Pseudolaw

Status: ACTIVE SYSTEM PSEUDOLAW / MORAL HYGIENE GUIDANCE.
Authority class: always-readable guidance for LUCIDOTA/CLAW work; use it before adding workflow/runtime code, routing logic, model orchestration, agent loops, queue workers, or audits.

## Source anchors

- Paper: arXiv:2508.02721, "Blueprint First, Model Second: A Framework for Deterministic LLM Workflow".
  - URL: https://arxiv.org/abs/2508.02721
  - HTML: https://arxiv.org/html/2508.02721
  - Submitted: 2025-08-01.
- Reference implementation yardstick: PocketFlow.
  - Repo: https://github.com/The-Pocket/PocketFlow
  - Local clone: `01_REPOS/PocketFlow/`
  - Core file: `01_REPOS/PocketFlow/pocketflow/__init__.py`
  - Local measured core size at adoption: 100 lines / 88 nonblank lines.

## The law

1. Blueprint first, model second.
   - The workflow path belongs in readable source code, schema, templates, queues, or explicit state machines.
   - A model may perform bounded interpretation/summarization/extraction at a named node.
   - A model must not be the hidden controller of the workflow path.

2. Determinism before cleverness.
   - Prefer fixed routes, visible guards, small routers, RETE/rules, schemas, typed packets, and receipts.
   - Use stochastic output only behind contracts that can be validated.
   - Never have an LLM do work that a deterministic workflow, parser, hash check, schema, router, or explicit state machine can do exactly and faster.
   - This is not model-zero doctrine: use LLMs for bounded ambiguity, messy language extraction, synthesis, adversarial ideation, drafting, and code generation when deterministic machinery is not the right tool.

3. Execution blueprints are first-class artifacts.
   - If the system says it can do a workflow, there should be a small runnable blueprint or queue worker that makes the path explicit.
   - The blueprint should be inspectable without needing to trust a prompt transcript.

4. PocketFlow is the simplicity mirror.
   - If a compact workflow core can express nodes, batches, async execution, and flows in about 100 lines, LUCIDOTA code needs a reason before growing large.
   - At 5x PocketFlow size: review for slop.
   - At 10x: split, template, or move behavior into data unless there is a clear reason.
   - At 20x: require explicit receipt-backed justification.
   - At 40x: treat as a blocker unless it is legacy surface area already queued for reduction.
   - These are not universal software laws. They are local smoke alarms. Cohesion, branching, coupling, and cognitive complexity matter too.

5. Receipts over claims.
   - A PASS requires runnable proof: command, output, receipt, schema, or check.
   - A markdown-only claim is not completion.

6. Templates over prose generation.
   - Operator-visible output should prefer deterministic templates with bounded context.
   - Free prose is commentary, not the contract.

7. Slop audit question.
   - Before adding complexity, ask: are we making the blueprint clearer, or are we hiding workflow inside more code/prompt fog?

8. Proportional gates.
   - Security/proof/logging that freezes work, floods storage, or blocks low-risk reversible action without proportional benefit is also slop.
   - Put heavy gates at graph materialization, destructive actions, external effects, sensitive exports, and slow-lane review.
   - Keep hot-lane proof small, fast, and sufficient.

9. Terminology truth.
   - Use established terms when they exist.
   - If a name is local, metaphorical, or provisional, say so.
   - Do not present invented labels as if they are domain concepts.

10. Temporal truth.
   - Current/latest/best-now questions require current verification.
   - Stale model memory is not evidence of present reality.

## Practical enforcement hooks

- Read this file through `AGENTS.md` and `00_PROJECT_BRAIN/ACTIVE_INSTRUCTION_INDEX.md`.
- Audit code with:
  - `python3 scripts/slop_audit_law.py --paths <file-or-dir>`
  - `./claw slop-audit --paths <file-or-dir>`
- Prefer receipts under `05_OUTPUTS/slop_audit/` when arguing that large code is justified.

## Non-goals

- This does not delete or shame the proof hoard.
- This does not ban large files automatically.
- This does not make the paper canonical truth about all software engineering.
- It is a standing hygiene lens: blueprint path visible, model bounded, receipts real, code small unless proven otherwise.
