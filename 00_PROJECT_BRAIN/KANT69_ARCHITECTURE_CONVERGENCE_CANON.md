# KANT69_ARCHITECTURE_CONVERGENCE_CANON

> Code name: **KANT69**
> Authority: Northern.Strike, declared 2026-05-28.
> Author/owner: Indy_READs (architecture is mine; this is the spine).
> Status: **ACTIVE HOLY DOCUMENT.**

This is the current active holy document for LUCIDOTA architecture convergence.

All older framework language is demoted unless it conforms to this document, the
canonical 75-term ontology, and the current GONN master build direction. The past
is **evidence, not law.** Nothing is canon because it is old, important, or elegant.
Canon is what KANT69 and the operator declare.

---

## 0. AUTHORITY

Code name:

```text
KANT69_ARCHITECTURE_CONVERGENCE_CANON
```

Authority order:

```text
1. KANT69_ARCHITECTURE_CONVERGENCE_CANON
2. canonical 75-term ontology (GO + CO + IO, @69=DAEMON)
3. GONN_MASTER_BUILD_PLAN.md as build-direction canon where consistent
4. CURRENT_HANDOFF / STATUS_LEDGER / receipts as state evidence
5. older framework docs only where not contradicted
```

The 75 ontology terms are canon. Not aspirational. Not maybe. Not "48 plus later."
The full 75-term GO+CO+IO ontology is the target authority and all registry/seed
files must converge to it.

GONN_MASTER_BUILD_PLAN.md is sort-of canon: it is **build canon, not ontology
canon.** It guides construction unless it conflicts with KANT69 or the 75-term
authority, in which case it gets patched or demoted.

---

## 1. KANT69_KERNEL_GOVERNOR

The governor must not be a custom limp-mode daemon.

The governor is **Linux-native resource control** using FOSS/kernel/dedicated
software already present or standard on the machine. We write a thin policy
adapter over the real machinery — we do not reinvent it.

Target primitives:

```text
cgroups v2
systemd slices/scopes/services
systemd resource-control properties
Pressure Stall Information /proc/pressure/*
systemd-oomd where available
nice / renice
ionice
cpuset / CPUAffinity
CPUWeight / CPUQuota
IOWeight / IOReadBandwidthMax / IOWriteBandwidthMax
MemoryHigh / MemoryMax / MemorySwapMax
TasksMax
OOMPolicy / OOMScoreAdjust
systemd-cgtop / systemctl show / systemd-run
journal limits / tmpfiles / quota tools where relevant
```

Rule:

```text
Do not invent a fake governor.
Write a thin LUCIDOTA policy adapter over Linux's real resource machinery.
```

The governor's job is not to pause the system. Its job is to keep the machine
alive while work continues.

Default posture:

```text
open
permissive
high-throughput
not limp mode
```

Target posture:

```text
saturate safely around 80–85%
throttle only from the top
shed/adapt expensive work first
keep custody/admission/receipts alive
recover automatically after pressure drops
```

Forbidden behavior:

```text
no permanent heavy-work pause
no "protected" idleness
no stopping Epoch/reingest as a safety trophy
no custom governor theater while cgroups/systemd/PSI exist
```

Governor classes (systemd slices):

```text
luci-custody.slice
  highest survival priority
  hash/pointer/event/receipt path
  must remain alive under pressure

luci-ops.slice
  queues, leases, worker heartbeats, admission decisions

luci-reingest.slice
  heavy reingest work
  throttled before custody/ops

luci-llm.slice
  model inference, embedding, summarization, expensive synthesis

luci-scrape.slice
  network acquisition and external source work

luci-archive.slice
  unpack/OCR/CAS-heavy byte operations
  first class to slow under disk/IO pressure
```

Implementation directive:

```text
Probe installed FOSS/Linux controls first.
Prefer systemd-run scopes and persistent slice units.
Apply cgroup properties, not vibes.
Read PSI, memory, IO, CPU, disk, swap, and process state.
Throttle expensive slices.
Never throttle custody below survival.
Emit a governor event for every admission, throttle, refusal, recovery, and kill.
```

GPU/VRAM rule:

```text
Linux cgroups do not magically govern VRAM like RAM.
Do not fake it.
VRAM is governed by model admission/scheduling:
  read available telemetry
  refuse new model loads under pressure
  unload lower-priority models
  cap batch/context sizes
  preserve custody/ops
  emit events
```

---

## 2. KANT69_DB_TOPOLOGY

One Postgres instance. One logical graph. Separate schemas. Canon and operating
machinery do not share hot mutable tables.

```text
lucidota_canon:
  graph_item / graph_object
  graph_edge
  graph_journal / graph_event
  term_registry
  claim_registry
  authority_registry

lucidota_ops:
  workflow_task
  work_order
  queue_claim
  worker_heartbeat
  lease
  retry_state
  admission_decision
  runtime_metric
  governor_action

lucidota_stage:
  staging_packet
  reingest_diff
  parse_candidate
  graph_candidate
  ontology_candidate

lucidota_corpse:
  retired_artifact
  schema_corpse_manifest
  deprecated_script_manifest
```

Hard rule:

```text
Hot contention rows are not canon.
SKIP LOCKED belongs in ops.
Canon records facts/events about machinery.
Machinery does not become the ledger because it is important.
```

Work orders live in ops. Important lifecycle transitions emit canon events.
Promoted work products become canon objects.

**Why one DB, not two:** on a single-box, single-operator system the two-database
split (`lucidota_state` / `lucidota_storage`) buys no real safety — it only costs
cross-DB joins, two pools, two backups, and no shared transaction. The rule it was
protecting ("no graph truth in the state DB") is enforced harder by a **schema
boundary + REVOKE on the worker role** inside one instance. One DB, scoped schemas,
permission grants. Safer and simpler.

---

## 3. KANT69_CUSTODY_NOT_HOARDING

Custody is law. Byte retention is policy. Immediate ingestion means
custody-of-record, not compute and not mandatory byte hoarding.

Required immediately:

```text
hash if available
path / URI / pointer
observed_at
source class
size if known
type/MIME guess if cheap
custody event
success/failure receipt
```

Admission-controlled:

```text
copy original bytes into CAS/vault
unpack archive
OCR
parse
embed
summarize
deduplicate deeply
retain duplicate bodies
long-term byte storage
```

Disk-pressure rule:

```text
Pointer-only custody is valid.
Refusal to retain bytes still emits an event.
No observed object silently disappears.
```

On a constrained box, "preserve every original body" is disk suicide. The governor
decides byte retention using:

```text
disk pressure
evidence value
uniqueness
reconstructability
case relevance
legal/evidence class
operator priority
hash/custody status
```

---

## 4. KANT69_THREECLOCK

Time is three-axis now. Not two. Not vague chrono sludge.

Use **one event store with three explicit time domains:**

```text
1. world_time
2. custody_time
3. processed_time
```

### world_time

World time is contestable. It describes when the thing allegedly happened in the
world.

```text
occurred_at
claimed_event_time
published_at
effective_date
valid_from
valid_to
alleged_at
reported_at
```

World time belongs on objects, claims, edges, and facts as **gated attributes.**
It can conflict. It can be wrong. It can be revised.

### custody_time

Custody time is the machine's evidence timeline. When LUCIDOTA saw, hashed,
ingested, promoted, rejected, or validated something.

```text
observed_at
ingested_at
hashed_at
registered_at
promoted_at
validated_at
rejected_at
```

Custody time is the system's hard internal chain-of-custody axis.

### processed_time

Processed time is the workflow/computation timeline. This matters especially for
**reingested casework.**

Old evidence may have:

```text
world_time from 2019
custody_time from 2026 reingest
processed_time from multiple parse/embed/classify passes
```

Examples:

```text
parsed_at
extracted_at
embedded_at
chunked_at
classified_at
scored_at
reranked_at
batched_at
reprocessed_at
materialized_at
```

Processed time may repeat many times across versions. Every processing pass carries:

```text
processor name
processor version/hash
input object hash
output object hash
started_at
finished_at
status
receipt pointer
```

Hard rule:

```text
One ledger.
Three time axes.
Not three ledgers.
```

This gives cleaner math for reingested casework because it separates:

```text
when the world says it happened
when LUCIDOTA got custody
when LUCIDOTA computed over it
```

---

## 5. KANT69_STREAM

Do not rebuild Bytewax/River. Existing artifacts are integration targets, not
excuses to start over.

Known targets:

```text
007_bytewax_stream.sql
absurd_river_worker
```

Required path:

```text
reingest diff stream
  -> existing Bytewax stream/bridge
  -> existing River worker
  -> feature rows / online learner state
  -> receipt/event emission
  -> graph/stage promotion where admitted
```

Hard rule:

```text
If a subsystem already exists, inspect, wire, validate, and receipt it.
Do not create parallel-framework cosplay.
```

---

## 6. KANT69_75_ONTOLOGY

The 75-term ontology is canon. Current drift must be killed.

Known drift:

```text
OFFICIAL_ONTOLOGY.json says 48 / GO-25
handoff says full 75 GO+CO+IO with @69=DAEMON
graph_core seed loads only 1–45
```

Correct convergence order:

```text
1. Declare full 75-term GO+CO+IO ontology as authority.
2. Ensure @69=DAEMON exists if canon says it exists.
3. Update OFFICIAL_ONTOLOGY.json to 75.
4. Update term_registry to 75.
5. Update graph_core seed to 75.
6. Add validation:
   - official count == 75
   - registry count == 75
   - seed count == 75
   - all IDs unique
   - all names unique or explicitly aliased
   - authority version recorded
7. Emit ontology_convergence_receipt event.
```

No workflow may claim "GO-25," "full ontology," or "75 words" unless it records the
exact ontology authority/version/hash it used. Ontology drift poisons routing, graph
writes, receipts, and reingest classification. It is not cosmetic.

---

## 7. KANT69_RECEIPTS

A receipt is an **event.** A receipt may reference an object body. The event proves
the step happened. The file is only the receipt body/object.

```text
event:
  command ran
  artifact produced
  validation passed
  validation failed
  file observed
  file promoted
  file rejected
  bytes retained
  bytes refused
  workflow started
  workflow completed

optional object body:
  JSON receipt
  Markdown report
  stderr log
  stdout log
  output packet
  generated artifact
```

Hard rule:

```text
The event is the receipt.
The file is the receipt body.
Stop conflating them.
```

---

## 8. KANT69_MIGRATION

Collapse in place. Never greenfield. Never delete. Archive/corpse with hashes.

Existing graph core stays alive:

```text
graph_item
graph_edge
graph_journal
```

Other schema files get classified:

```text
canon
view
function
staging
ops
obsolete-but-preserved
corpse
```

Every demotion emits:

```text
hash
old path
new path if any
classification
reason
replacement if any
event
receipt body if any
```

Retired material goes to KRAMPUSCHEWING with hashes. No "clean new architecture"
while the old organism still contains working nerves.

---

## 9. KANT69_EXECUTION_ORDER

Opening strike:

```text
1. Install this KANT69 doctrine as current active authority.
2. Probe Linux/FOSS resource controls already available.
3. Replace fake governor doctrine with cgroups/systemd/PSI policy adapter.
4. Converge ontology to canonical 75.
5. Wire reingest diff stream into existing Bytewax/River.
6. Preserve three-clock time fields in reingest/event math.
7. Resume heavy reingest under kernel-backed governor control.
8. Emit receipts for every transition.
```

Do not spend a week making paper. The deliverable is executable convergence:

```text
one authority document
one governor doctrine
one DB topology
one three-clock chrono model
one receipt rule
one ontology authority target
one migration posture
then patch the machine
```

KANT69 is the spine. Everything else conforms or gets demoted.
