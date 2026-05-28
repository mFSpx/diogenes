# RFC-060: ABSURD Workflows / Durable Postgres Work

Status: DRAFT  
Subject ID: `absurd_workflows`  
Normative role: defines durable workflow execution for LUCIDOTA.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

ABSURD is the durable workflow spine for LUCIDOTA. It uses Postgres-backed queues, worker contracts, events, retries, and dead letters so that work is predictable in execution even when investigations and outcomes are uncertain.

ABSURD is not a vibe name. It is the local replacement/current target for legacy DBOS-era workflow language.

## 2. Why I Believe This Is True

Local evidence:

- `00_PROJECT_BRAIN/DURABLE_WORKFLOW_DECISION.json` says current durable workflow substrate is ABSURD and new work targets ABSURD/Postgres.
- `06_SCHEMA/035_absurd_queue_spine.sql` defines queues, jobs, events, and dead letters.
- `06_SCHEMA/082_absurd_worker_contract_registry_enforcement.sql` registers worker contracts.
- `scripts/absurd_queue_spine.py` is the queue implementation surface.
- `scripts/absurd_consume_one.py` consumes one job through the spine.
- `scripts/absurd_worker_contracts.py` rejects jobs without active contracts before side effects.

## 3. External Source Anchors

- PostgreSQL `FOR UPDATE` / `SKIP LOCKED` documentation explicitly supports queue-like consumers that avoid waiting on locked rows, with the caveat that it is not a general consistent view. ABSURD uses it for queue claims, where that caveat is appropriate: <https://www.postgresql.org/docs/current/sql-select.html>.
- Blueprint First / Model Second supports explicit workflow state in code/schema instead of prompt-driven hidden orchestration: <https://arxiv.org/abs/2508.02721>.
- W3C PROV-O supports event/activity provenance; ABSURD queue events provide the local activity log: <https://www.w3.org/TR/prov-o/>.
- RFC 2119 supports turning worker-contract requirements into explicit MUST/SHOULD language: <https://datatracker.ietf.org/doc/rfc2119/>.

## 4. Workflow Mechanics

ABSURD jobs carry:

- queue name,
- workflow name,
- job kind,
- idempotency key,
- payload,
- status,
- priority,
- attempt count,
- lease metadata,
- result/error,
- timestamps.

Events record transitions such as enqueued, leased, started, succeeded, failed, dead-lettered, retry scheduled, cancelled, or audit.

Dead letters preserve failed work instead of erasing it. This is not just operational hygiene; it is epistemic hygiene. A failed parser, missing dependency, contract rejection, or transient resource limit is evidence about the system. Erasing that evidence makes the operator repeat mistakes and makes agents overclaim completion.

The active Dev Library also shows why ABSURD is the right center: there are many old DBOS wrappers and legacy workflow scripts, but `00_PROJECT_BRAIN/DURABLE_WORKFLOW_DECISION.json` names ABSURD/Postgres as the current target. ABSURD absorbs the useful workflow pattern while demoting legacy names to provenance/compatibility.

## 5. State Machine Law

ABSURD work must move through named states rather than implied terminal output. Queued, leased, running, succeeded, failed, retry-scheduled, cancelled, and dead-lettered are different facts. A process exiting zero is not enough if the job row, event row, receipt, or output artifact disagrees. Conversely, a worker can crash and the system can still remain honest if the lease expires, attempts are counted, and dead-letter review has a record.

That is why ABSURD is a better fit than terminal-scroll automation. It turns execution into queryable state. It also lets the operator run a dumb workflow without the system pretending the outcome was smart; the execution can be predictable while the result remains the operator's problem.

## 6. Worker Contract Law

A worker MUST validate queue/job kind before side effects. If the contract table is missing, no fallback invention is allowed. If a job kind is not listed, the worker rejects and records the failure. This is central to making execution predictable.

## 7. Whole-System Interaction

- **ETL** enqueues derived work.
- **Diogenes** authorizes privileged enqueue/effects.
- **KORPUS** uses queue wrappers for ingestion/componentization health and slow work.
- **Graph Promotion** uses queued candidates and guarded materialization.
- **River/Bytewax/learning** use queue/event streams.
- **Templates/reports** can be generated from completed work and receipts.

## 8. Benefit to the Whole System

ABSURD gives LUCIDOTA a nervous system. It prevents live work from becoming a pile of one-off scripts whose state exists only in terminal scrollback. It also supports operator-level responsiveness: work can be queued, resumed, retried, cancelled, dead-lettered, and inspected.

## 9. Correctness Argument

Postgres is already the durable local control plane. For a single-machine sovereign system, Postgres queues are simpler and more inspectable than adding Redis/Kafka/gRPC as mandatory dependencies. `SKIP LOCKED` workers allow bounded concurrency while keeping job state queryable. The caveat in PostgreSQL's own documentation is exactly why ABSURD must treat queue claims as work distribution, not as a general consistent analytical view. Completion truth comes from job status, events, receipts, and dead-letter review together.

## 10. Falsifiers

This RFC is wrong if:

- Postgres queue performance cannot handle the workload,
- worker contract enforcement blocks necessary live repair paths without a safe override,
- dead-letter review becomes ignored junk,
- scripts continue bypassing ABSURD for active workflow state,
- old DBOS naming remains active architecture instead of provenance.

## 11. Filesystem / Runtime Consequences

- New durable workflows target ABSURD/Postgres.
- Legacy DBOS files may remain only as provenance or compatibility.
- Worker scripts must be contract-registered before trusted execution.
- Receipts/events/dead letters are required evidence for completion claims.
