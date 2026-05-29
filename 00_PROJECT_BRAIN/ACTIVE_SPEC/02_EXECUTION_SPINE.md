<!--
DEV NOTE (anti-slop provenance, 2026-05-24): A file/report/taxonomy is proof of the truth it actually declares: this document exists, has this title, and asserts this scoped content. It is not proof of unrelated external facts, not permission to rename its scope as "the manual," and not evidence that the operator personally coined a local label. Names are integral to their own statements and contexts; do not erase, swap, or flatten them. When an agent says "this is the user's rule," it must cite direct operator instructions or accepted doctrine/RFC/receipt lineage, state exactly how many source docs were consulted and why those docs were chosen, and distinguish direct evidence from inference or compression. If the system lacks receipts for a factual CLAIM, classify it as hypothesis, observation, inference, suggestion, or confidence-rated candidate instead.
-->

# LUCIDOTA Execution Spine

Status: ACTIVE SPEC SOURCE — execution path and graph-affecting authority  
Purpose: define the execution path every live workflow must map onto before it can affect queues, receipts, candidates, or graph materialization.

## 1. Spine in One Line

```text
operator intent or raw artifact
  -> custody / command envelope
  -> ABSURD/Postgres work order
  -> bounded worker
  -> receipt / event / dead letter
  -> evidence / claim / hypothesis candidate
  -> authority mapping
  -> graph promotion gate
  -> optional operator-confirmed materialization
  -> review/export/status surface
```

The spine is not one daemon. It is the rule that all daemons, scripts, workers, shells, model calls, parsers, scrapers, and live-coding tools must obey.

## 2. Main Objects

Keep the operator-facing graph model simple:

1. **OBJECT** — things: people, files, claims, models, tools, places, concepts, artifacts, templates.
2. **EVENT** — happenings: ingest, parse, hash, route, model call, operator decision, score, send attempt, promotion attempt.
3. **EDGE** — relationships: supports, contradicts, derived-from, used-by, owns, part-of, happened-before, mentions.

GO-25 / `OFFICIAL_ONTOLOGY.json` is the active ontology skin over that model. ROOT-414 remains archived reference material unless explicitly reactivated.

## 3. Authority Boundary

No ordinary worker writes canonical graph truth directly.

Canonical graph promotion requires:

1. evidence refs,
2. authority class,
3. graph-promotion preflight,
4. command envelope when operator confirmation/materialization is required,
5. guarded helper path,
6. graph journal append,
7. materialization receipt.

Raw evidence, parser output, GLiNER output, local model output, River score, Ahoy simulation, and Krampus/KORPUS component rows are candidates only.

## 4. Mutation Rights

Every component must declare one of these mutation classes:

- `read_only`: may inspect only.
- `receipt_only`: may write logs/reports/receipts under `05_OUTPUTS/`.
- `custody_writer`: may hash/store bytes or storage rows without truth promotion.
- `queue_writer`: may enqueue/dequeue work under worker-contract rules.
- `candidate_writer`: may stage evidence/claim/hypothesis/graph candidates.
- `authority_gate`: may approve/defer/reject according to registry/policy.
- `materializer`: may materialize canonical graph only through the guarded helper/journal path.
- `external_effect`: may touch outside-world systems only with explicit operator authorization and receipt.

If a component cannot name its mutation class, it is not allowed to mutate.

## 5. Workflow Law

A workflow is any repeatable path that turns input into a traceable result. Examples:

- document/evidence/facts/random material in -> structured evidence candidates out,
- questions to many people in parallel -> timed/polite outreach work orders and receipts,
- recursive pivot search -> network map candidates,
- all of the above chained into a case pipeline.

Every workflow must have:

- input contract,
- worker/route contract,
- output contract,
- failure/dead-letter behavior,
- receipt path,
- authority/mutation class,
- replay or audit handle.

## 6. Model Placement

Models sit at named nodes only. They may:

- extract spans,
- summarize with source anchors,
- draft prose,
- rank candidates,
- classify uncertain items,
- generate bounded suggestions.

They may not:

- secretly choose the workflow,
- claim facthood,
- bypass authority gates,
- materialize canonical graph,
- send external messages,
- erase uncertainty,
- claim completion without receipts.

## 7. Spine Enforcement Sources

Primary enforcement sources:

- `00_PROJECT_BRAIN/spine_authority_registry.json`
- `00_PROJECT_BRAIN/canonical_graph_write_allowlist.json`
- `06_SCHEMA/035_absurd_queue_spine.sql`
- `06_SCHEMA/082_absurd_worker_contract_registry_enforcement.sql`
- `scripts/absurd_queue_spine.py`
- `scripts/absurd_consume_one.py`
- `scripts/absurd_worker_contracts.py`
- `scripts/spine_authority_checker.py`
- `scripts/spine_kernel_authorization.py`
- `scripts/graph_promotion_gate.py`
- `scripts/canonical_graph_write_scanner.py`

## 8. Completion

Completion is not a mood. Completion means:

- receipt exists,
- relevant status/check command passes,
- failures are recorded or dead-lettered,
- graph writes are declared yes/no,
- outputs are inspectable by path/hash/query.
