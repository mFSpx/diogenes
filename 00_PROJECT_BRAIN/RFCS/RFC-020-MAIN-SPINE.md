# RFC-020: Main Spine / Graph-as-Memory Execution Model

Status: DRAFT  
Subject ID: `main_spine`  
Normative role: defines the executable backbone that keeps LUCIDOTA predictable in execution while allowing unpredictable investigative outcomes.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

LUCIDOTA's main spine is a durable, authority-gated execution path from operator intent or raw artifact to receipt-backed graph candidate and optional materialization. The graph is not just a database feature; it is the system's addressable memory. The spine is the bus that decides how anything gets permission to affect that memory.

The spine is correct when it is boring:

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

## 2. Why I Believe This Is True

I believe this model is true because multiple independent local sources converge on it:

- `00_PROJECT_BRAIN/ACTIVE_SPEC/02_EXECUTION_SPINE.md` states this exact path.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md` maps Diogenes, ABSURD, KORPUS, ontology, language membrane, models, and templates onto that path.
- `06_SCHEMA/035_absurd_queue_spine.sql` implements durable queue/job/event/dead-letter tables.
- `scripts/absurd_worker_contracts.py` refuses to execute queue work unless a live worker contract permits the queue/job kind.
- `scripts/spine_authority_checker.py` defines authority classes, allowed effects, permitted lanes, and evidence requirements.
- `scripts/graph_promotion_gate.py` exists as the graph-promotion boundary.
- `OFFICIAL_ONTOLOGY.json` defines GO as the active ontology and ROOT-414 as archived reference.

This is stronger than a design preference. It is a repeated implementation pattern: work moves as rows/events/receipts, not as hidden model intention.

## 3. External Source Anchors

- PostgreSQL documents row locking and `SKIP LOCKED`, including its queue-like use case for avoiding lock contention between consumers. That supports ABSURD's queue-worker pattern while warning not to treat skipped locked rows as a general consistent view: <https://www.postgresql.org/docs/current/sql-select.html>.
- W3C PROV-O supports representing data lineage through entities, activities, and agents. LUCIDOTA's OBJECT/EVENT/EDGE model is locally simpler but aligned with that provenance frame: <https://www.w3.org/TR/prov-o/>.
- SEPIO supports modeling claims together with evidence and provenance; this matches LUCIDOTA's refusal to collapse evidence, claim, hypothesis, and truth: <https://ohsu.elsevierpure.com/en/publications/sepio-a-semantic-model-for-the-integration-and-analysis-of-scient/>.
- Blueprint First / Model Second supports the rule that workflow path belongs in code/schema/queues/templates, not hidden model control: <https://arxiv.org/abs/2508.02721>.

## 4. Main Objects

Operator-facing graph memory SHOULD collapse to three primitives unless a specific subsystem proves need for more:

1. **OBJECT** — thing being reasoned about: file, person, artifact, model, claim, tool, source, concept.
2. **EVENT** — change/happening: ingest, parse, hash, model call, route, operator decision, promotion attempt.
3. **EDGE** — relation: supports, contradicts, derived-from, happened-before, uses, owns, mentions, part-of.

GO-25 is the active ontology skin over this model. ROOT-414 is no longer active repo gravity; it has been externalized to `/home/mfspx/Documents/ROOT_414/`.

## 5. Authority and Mutation

All runtime components MUST declare mutation class:

- `read_only`
- `receipt_only`
- `custody_writer`
- `queue_writer`
- `candidate_writer`
- `authority_gate`
- `materializer`
- `external_effect`

A component with no declared mutation class is read-only by default. A model has no materialization authority by default. A worker has no canonical graph authority by default.

Canonical graph materialization requires:

1. evidence refs,
2. authority class,
3. graph-promotion preflight,
4. command envelope when operator confirmation/materialization is required,
5. guarded helper path,
6. graph journal append,
7. materialization receipt.

## 6. Whole-System Interaction

- **ETL/KORPUS** supplies custody objects and component candidates.
- **Diogenes** supplies command authority and control-packet discipline.
- **ABSURD** supplies durable queue movement.
- **Models** supply draft/extraction/ranking at named nodes.
- **Language Membrane** supplies inspectable input/output lanes.
- **Indy_READs** supplies teammate synthesis and reading memory as candidates.
- **PercyphonAI** supplies deterministic scaffolding metadata.
- **Artifact Templates** produce reviewable outputs from candidates.
- **Authority Gate** decides whether anything can approach canonical truth.

The spine gives all organs a shared rule: useful outputs are allowed; unlabeled authority is not.

## 7. Benefit to the Whole System

The main spine makes LUCIDOTA snappy and predictable in execution because work can be enqueued, claimed, inspected, retried, dead-lettered, or replayed. It also keeps outcomes open: the operator may ask for a dumb workflow, sources may contradict, models may be wrong, and investigations may pivot. Predictability belongs to execution mechanics, not to the outcome.

This is why the system can be a LEGO kit for hypersystemia: each block has a connector, authority class, receipt path, and failure mode.

## 8. Correctness Argument

The main spine is correct to the degree that it makes every meaningful state transition nameable, replayable, and challengeable. A graph candidate is not trusted because a model said it; it is trusted only when its input artifact, worker, queue event, authority class, and receipt can be followed. That matches the local schemas and scripts: ABSURD queues produce observable job state, worker contracts limit execution, authority checking limits mutation, and graph promotion is a named gate rather than a vibe.

The correctness claim is therefore not “the graph is always right.” The claim is narrower and more defensible: the spine makes wrongness findable. If a bad claim lands in the system, the operator should be able to locate the source artifact, parser/model/template/workflow that produced it, the authority rule that allowed it, and the receipt that records it. That is the required property for a living abductive system.

## 9. Falsifiers

This RFC is wrong if:

- current useful workflows require direct graph writes from ordinary workers,
- evidence cannot be traced from graph candidate back to custody/source,
- the queue spine cannot represent live work without unacceptable friction,
- the ontology cannot collapse to OBJECT/EVENT/EDGE plus GO skin,
- model calls must control routing invisibly for the system to function.

Current evidence supports the opposite: direct graph writes are treated as dangerous, queues/events/dead letters exist, authority checker exists, and model output is explicitly candidate/draft scoped.

## 10. Filesystem / Runtime Consequences

- `ACTIVE_SPEC/02_EXECUTION_SPINE.md` stays short and canonical.
- Long arguments live here in `RFCS/`.
- Graph-affecting code MUST route through authority/gate scripts.
- New active workers SHOULD appear in worker contracts and queue schemas.
- Reference/lab material MUST be outside active spine unless explicitly promoted.
- Tests should prove authority/mutation boundaries, not merely imports.
