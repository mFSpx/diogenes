# RFC-160: Automation / Scheduler / Watchers

Status: DRAFT  
Subject ID: `automation`  
Normative role: defines automation as conservative observation, idempotent work triggering, scoped remediation, and receipts — not a swarm of hidden autonomous agents.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

LUCIDOTA needs automation because a live sovereign exocortex must keep ingesting, watching, refreshing, and surfacing status while the operator is elsewhere. But automation is dangerous when it hides authority. Therefore LUCIDOTA automation must be boring: cron-safe, idempotent, observable, conservative, and scoped.

The current automation evidence shows the correct posture: ingestion watchdog observes and writes dashboards; Indy watcher starts a bounded book watcher; KRAMPUSCHEWING watcher waits for stable drops, routes probable chat/communication dumps, runs KORPUS, sidecars, rechrono, hard-math, and moves digested material only after pipeline success. The automation is not an unsupervised truth engine. It is an operating loop.

## 2. Sources

### Local sources

- `scripts/lucidota_ingest_watchdog.py` — conservative observation dashboard; explicitly avoids moving evidence, killing watcher parents, or broad schema DDL; only narrow safe remediation.
- `scripts/lucidota_start_indy_reads_watcher.sh` — idempotent PID-file guarded startup for Indy_READs watcher.
- `scripts/krampuschewing_watcher.sh` — stable-drop watcher that routes chat dumps, comm dumps, brain sidecar, KORPUS ingest, rechrono, hard truth math, and digested movement.
- `scripts/lucidota_observation_live_loop.sh` and `scripts/lucidota_observation_live_loop_ensure.sh` — flock/PID guarded dashboard updater loop.
- `06_SCHEMA/067_chrono_queue_event_bridge_automation.sql` — durable run records for idempotent ABSURD queue-event → Chrono bridge automation; appends temporal evidence and does not overwrite/delete claims.
- Dev Library query evidence for `watcher`, `automation`, and `board` — existing watchers and dashboards should be reused before new daemons are invented.

### External Source Anchors

- systemd timer documentation defines timer units activating services, `OnCalendar=`, and `Persistent=` catch-up behavior; this supports using explicit services/timers rather than hidden loops where appropriate: <https://www.freedesktop.org/software/systemd/man/latest/systemd.timer.html>.
- PostgreSQL queue/lock semantics support durable queue workers and multi-consumer patterns with `SKIP LOCKED`: <https://www.postgresql.org/docs/current/sql-select.html>.
- W3C PROV-O supports representing automated activities and agents in provenance chains: <https://www.w3.org/TR/prov-o/>.
- Blueprint-first design supports small explicit workflow blueprints over prompt-driven hidden routing: <https://arxiv.org/abs/2508.02721>.

## 3. Automation Law

Automation MUST declare:

- what it watches,
- what it may write,
- what it may move/delete/kill/restart,
- what it refuses to do,
- where logs and receipts go,
- how idempotency is maintained,
- how a human can inspect status.

If it cannot declare those, it is not automation; it is a gremlin.

## 4. Watcher Classes

LUCIDOTA currently needs these automation classes:

1. **Observation loops** — dashboards, status JSON/Markdown/HTML, health reports.
2. **Ingest loops** — drop-folder stability checks, KORPUS runs, chat/comm dump routing.
3. **Reading loops** — Indy book watcher and LoRA staging appenders.
4. **Queue bridge loops** — ABSURD queue events to Chrono/evidence appenders.
5. **Safe remediation loops** — narrow restarts or stale-child cleanup, never broad destructive cleanup.
6. **Ensurers** — PID/flock/systemd-style wrappers that keep one loop alive without duplication.

## 5. Whole-System Interaction

- **Main spine:** automation produces events, receipts, dashboards, and queued work; it does not bypass graph promotion.
- **KORPUS:** watchers feed drop material into custody-first ingest.
- **Indy_READs:** reading watcher keeps book/chunk/adaptation surfaces current.
- **Constant learning:** queue outcomes and watcher events become learning features.
- **Artifact templates:** dashboards and status surfaces are artifacts with sourceable state.
- **Filesystem law:** automation writes to `04_RUNTIME/` and `05_OUTPUTS/`, not random source paths.
- **Diogenes:** any irreversible or external effect must pass command authority, not just watcher impulse.

## 6. Benefit to the Whole System

Automation makes LUCIDOTA snappy. The operator should not have to manually remember every watcher, ingestion loop, dashboard refresh, and reading index. Conservative automation keeps the system alive while preserving operator authority.

The benefit is not autonomy for its own sake. The benefit is reduced context switching: the machine maintains boring loops, so the operator can spend attention on analysis, design, and decisions.

## 7. Correctness Argument

I believe this RFC is correct because the current scripts repeatedly state and implement conservative boundaries. The watchdog says it never moves evidence files or broadly mutates schemas. The observation loop uses `flock`; the Indy watcher checks PID liveness before starting; the KRAMPUS watcher waits for file stability and logs pipeline stages; the Chrono bridge schema records run facts and forbids overwrites/deletes of existing temporal claims.

The correctness property is controlled liveness. An automation component is correct when it keeps the system moving while making every action observable and reversible or bounded. If it hides mutation, it is wrong even if it is useful.

## 8. Falsifiers

This RFC is wrong if:

- watcher loops can start duplicate uncontrolled instances,
- automation moves/deletes evidence without receipts,
- automation writes canonical graph truth directly,
- logs/receipts are missing for automated runs,
- remediation kills parent/watch/control processes broadly,
- cron/systemd/shell loops cannot be inspected or stopped.

## 9. Filesystem / Runtime Consequences

- Runtime logs/PIDs go under `04_RUNTIME/`.
- Operator-facing dashboards and receipts go under `05_OUTPUTS/`.
- Durable automation run schemas live under `06_SCHEMA/`.
- New watchers should reuse existing PID/flock/log patterns.
- Destructive automation requires explicit command authority and receipts.
