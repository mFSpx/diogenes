# RFC-110: Input Multiplexing / Hot Lane and Slow Lane

Status: DRAFT  
Subject ID: `input_multiplexing`  
Normative role: defines inbound routing as deterministic packet/lane discipline before model or graph influence.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

LUCIDOTA needs multiplexed input because the operator's real workflow is bursty, layered, contradictory, and live-coded. Some inputs are tiny status probes; some are deep audits; some are evidence drops; some are emotional operator steering signals; some are graph-affecting commands. Treating all inputs as one chat stream is a slop generator.

The hot/slow lane split is the minimum discipline: route simple metadata/status/cache/check packets fast, route deep analysis/model/research/graph/refactor work slow, and flush accumulated fast-lane bits into slow-lane bundles when importance or count justifies it.

## 2. Sources

### Local sources

- `00_PROJECT_BRAIN/ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md` — canonical ETL path and hot/slow lane doctrine.
- `scripts/fast_slow_lane_gate.py` — deterministic metadata/text gate with FASTLANE/SLOWLANE decisions, reasons, cache, importance scores, and flush bundles; performs no model/network/canonical graph writes.
- `06_SCHEMA/020_korpus_derived_compute_queue.sql` — derived compute queue with explicit delayed/enrichment tasks and raw ingest independence.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/02_EXECUTION_SPINE.md` — routes work into command/custody/queue/receipt/promotion paths rather than ad hoc mutation.

### External Source Anchors

- PocketFlow emphasizes small graph/workflow primitives; LUCIDOTA adopts the tiny packet → decision → hook lesson without making PocketFlow a runtime dependency: <https://github.com/The-Pocket/PocketFlow>.
- PostgreSQL `SKIP LOCKED` queue-like use supports multi-consumer slow-lane queues while warning about general-view inconsistency: <https://www.postgresql.org/docs/current/sql-select.html>.
- Blueprint-first design supports explicit routing logic before model calls: <https://arxiv.org/abs/2508.02721>.
- W3C PROV-O supports recording route activities and derived bundles as provenance entities: <https://www.w3.org/TR/prov-o/>.

## 3. Lane Contract

Input multiplexing MUST preserve:

- original packet text hash,
- route decision,
- route reason,
- lane hint or override,
- metadata flags,
- model-call state,
- graph-write state,
- cache/flush path when cached,
- importance score when relevant.

The fast lane is for immediacy, not truth. The slow lane is for depth, not delay theater. A packet can move from fast to slow through explicit flush bundles.

## 4. Hot Lane / Slow Lane Mechanics

The current gate makes a deterministic choice from text length, slow/fast keyword hits, metadata flags, and lane hints. Slow triggers include research, audit, refactor, model, graph, workflow, benchmark, failure, blocker, and proof-like terms. Fast hints include status, help, probe, route, metadata, receipt, JSON, health, manifest, and check.

This is not “smart AI routing.” It is intentionally simple, inspectable routing. A wrong route can be fixed by changing word lists, thresholds, or explicit metadata — not by pleading with an opaque model.

## 5. Whole-System Interaction

- **Diogenes:** operator commands enter as packets that can require command envelopes before mutation.
- **ETL/KORPUS:** raw evidence goes through custody regardless of derived slow-lane backlog.
- **ABSURD:** slow work becomes queueable jobs with leases, events, and dead letters.
- **Local LLM fabric:** model calls happen after routing, not as the default intake path.
- **Language membrane:** inbound text can be routed structurally before semantic/model lanes.
- **Artifact templates:** slow-lane bundles can become briefs or review packets.
- **Indy_READs:** reading/workflow requests can route to page-locked or role-mode work rather than generic chat.
- **Constant learning:** route decisions and later outcomes can become learning features, but not graph truth.

## 6. Benefit to the Whole System

Multiplexing keeps the system responsive without trivializing deep work. The operator can ask for a fast health/status/check while long audits proceed elsewhere. Important crumbs can be cached and later flushed into slow analysis instead of being lost in chat scrollback.

It also cuts LLM influence at the door. The first decision is deterministic and receipt-backed. That prevents the common failure where an LLM interprets urgency, authority, scope, and truth all at once.

## 7. Correctness Argument

I believe this RFC is correct because `fast_slow_lane_gate.py` encodes exactly the intended boundary: deterministic route reasons, no model calls, no network calls, no canonical graph writes, and explicit cache/flush records. The derived compute queue schema reinforces the same split by making enrichment asynchronous while raw evidence ingest remains independent.

The correctness property is not perfect classification. It is inspectable routing with recoverable errors. If a packet is misrouted, the route reason and packet hash remain visible, and the packet can be replayed or flushed.

## 8. Falsifiers

This RFC is wrong if:

- input routing requires an LLM for ordinary packets,
- route decisions lack reasons,
- graph writes happen inside the lane gate,
- raw evidence ingest blocks on slow-lane enrichment,
- fast-lane cached bits cannot be flushed/replayed,
- metadata overrides cannot force appropriate routing.

## 9. Filesystem / Runtime Consequences

- Keep lane caches under scoped storage/runtime paths.
- Keep lane receipts under scoped output paths.
- Do not place long-running analysis artifacts in active doctrine docs unless promoted.
- New input routers must declare model/network/graph side effects explicitly.
- Tests should assert no model calls and no canonical graph writes for deterministic lane gates.
