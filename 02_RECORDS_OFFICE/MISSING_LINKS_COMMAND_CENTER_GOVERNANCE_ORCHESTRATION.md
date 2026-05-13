# Missing Links: Command Center, Governance, Orchestration

## Status

Working architecture intake note from the final pre-shower spec dump. These are actionable tracks, not blockers.

## 1. Big Board / Human-In-The-Loop Interface

Need:

- a high-fidelity command center for investigation state.
- visible case timelines, evidence graph, source cartridges, active workflows, model/runtime health, queue state, and provenance.
- clear escalation surfaces where northern.strike approves or rejects external writes, emails, filings, purchases, and other high-impact actions.

Candidate implementation paths:

- private Clawd TUI first for immediate command workflow.
- local web dashboard later for dense visualization.
- Next.js is a likely long-term UI shell if the app needs full graph/case visualization.
- Streamlit is acceptable only for throwaway internal panels, not final product architecture.

Open design rule:

- command center must show state, not vibe theater.
- Big Board is a control surface tied to DBOS/CKDOG1/Postgres events.

## 2. Governance / Privacy / Validation Layer

Need:

- prevent scrapers from wrecking IP reputation or violating source constraints.
- prevent mailroom/calendar automation from acting beyond current authority.
- prevent generated drafts from becoming filed/sent/signed external actions without explicit approval.
- preserve provenance and audit trails for casework.

Candidate implementation paths:

- custom validation gates first, because LUCIDOTA policy is project-specific.
- NeMo Guardrails can be evaluated, but must not become a generic refusal engine.
- DBOS workflows should require explicit policy checks before external writes.
- CKDOG1 policy contracts should expose who/what set the policy and whether it is user-controlled or reactive.

Immediate policy split:

- internal writes: allowed as needed.
- external reads/scrapes: allowed but rate-limited, logged, and source-aware.
- external writes/sends/buys/files: require explicit user approval until delegated otherwise.

## 3. Resilient Task Orchestration

Need:

- long-running case prep must resume after crashes.
- scrapes, ingest, embeddings, graph writes, and analysis loops must be idempotent.
- task state must be inspectable from the Big Board and Clawd.

Current decision:

- DBOS is the primary workflow conductor.
- Do not add Temporal/Airflow unless DBOS fails a real requirement.
- DBOS should own workflow checkpoints, retry policy, queue state, and transaction-bound state changes.

Temporal/Airflow remain references:

- Temporal: useful benchmark for durable execution semantics.
- Airflow: useful reference for batch DAG visibility, less attractive for interactive local intelligence workflows.

## Component Map

- PGVector + AGE: long-term vector/graph memory.
- River ML: live reflexes and online learning.
- Vault: raw evidence and source artifacts.
- DBOS: durable workflow skeleton.
- CKDOG1: kernel/state/policy/proof layer.
- Clawd/LUCIDOTA UI: operator interface and Big Board.
- Model stack: listener/router/tactician/heavy-hitter runtime.

## Action Items

1. Add DBOS workflow naming and status conventions.
2. Add local workflow/event tables for Big Board consumption.
3. Add governance gate schema.
4. Add scraper rate-limit and source-policy skeleton.
5. Add external-action approval queue.
6. Add Big Board wireframe after next planning session.
