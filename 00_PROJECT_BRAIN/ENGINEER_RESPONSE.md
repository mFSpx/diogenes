# Response to Engineer Friend
_2026-05-30_

---

## Props — Yeah, You're Right

**Relational-Graph Hybrid:** You nailed it. The whole point of running `age` + `pgvector` + `pg_trgm` inside the same Postgres cluster is exactly what you said — no cross-cluster consistency theater. One txn, one lock domain, one DSN. 306 tables across two databases (`lucidota_state` for control plane, `lucidota_storage` for canonical truth) with hard schema contracts in `06_SCHEMA/` numbered 001–126. The `age` extension is installed in `lucidota_storage`, though worth noting: the primary GO graph (`lucidota_go.graph_item`, `graph_edge`) is actually relational — `ternary_valency INTEGER` columns, UUID keys, status enums, jsonb payload. Cypher only shows up in a legacy sidecar script (`lucidota_age_edges.py`). The native relational path is cheaper and already works; Cypher is available when graph traversal gets complex enough to need it.

**ABSURD Queue + Outbox:** Correct on the mechanics. `FOR UPDATE SKIP LOCKED` on `lucidota_control.workflow_event`, outbox trigger fires on INSERT and fans into `event_outbox` atomically. No Kafka, no Redis, no broker to babysit. The worker contract registry (`absurd_worker_contract` table) enforces that handlers must be declared before they can consume — so no anonymous workers touching queues.

**Deterministic Streaming:** Also correct. 26 Treelite stumps generated at Python import time from 5-feature float32 vectors. The entire scoring layer runs in-process with zero disk I/O. Decoupled from model servers entirely.

---

## Critique — Partially Right, But Missing Context

**On the VRAM "suicide mission":** You're right about `swappiness=180` being a disaster — that needs to be 10, not 180. You're right that the 1650's 4GB is tight. But the BGE fleet freeze isn't a design problem, it's a regression bug in one line of one script.

The design is: BGE fleet runs CPU-only (NGL=0), 16 instances share mmap'd weights (~606MB once in RAM, not 16× loaded), each gets ~195MB compute buffers, total ~3.7GB RAM. That fits. The Mamba lanes and BGE fleet are never meant to run simultaneously — there's a load governor (`lucidota_runtime.load_governor_decision` with `budget_vram_mb`, `observed_free_mb`, `headroom_mb`, `decision` columns) specifically to prevent that.

The actual bug: `lucidota_bge_fleet.sh` line 41 has `NGL="${LUCIDOTA_BGE_NGL:-99}"`. The safe-ops env exports `LUCIDOTA_BGE_NGL=0` — but the script's own fallback of `:-99` fires when the var is **unset**, not empty. If you run the script without sourcing safe-ops first, NGL=99 activates GPU mode, loads the model into VRAM per-instance (not mmap'd), and blows the 4GB ceiling. Plus there's no earlyoom installed — no OOM killer fires, so `swappiness=180` cascades into a full swap storm and the machine locks cold. Five hard reboots, no logs.

The "enterprise server" framing isn't quite right either. This is a layered-on-demand stack, not everything hot simultaneously. The model lane facts in PG (`runtime_facts`) track which lanes are `always_hot` vs preemptible. Most aren't always-hot.

---

## Design Optimizations — Where I Agree and Where I Don't

**A. Alembic for the bandit tables:** Don't need it. We already have `06_SCHEMA/` as a numbered SQL contract layer — 126 files, each a canonical DDL contract. The `bytewax_rete_bandit_decision` and `bytewax_bandit_policy` tables exist only as inline SQL strings inside `bytewax_abductive_blender.py` because they were written fast and never promoted to `06_SCHEMA/`. The fix is a one-liner: create `06_SCHEMA/127_bytewax_rete_bandit.sql`, extract the DDL, run it. Adding Alembic wraps a Python migration layer around something that already works declaratively. The numbered SQL files are the migration history.

**B. anyio capacity limiters:** Valid point technically. In practice the fix is simpler — `--concurrency 3 --http-batch 16` on the command line combined with running through `lucidota_capped_run.sh` (cgroup MemoryMax=4.5G). The anyio limiter approach is cleaner code but the problem is a config knob, not an architecture gap. Worth doing if we ever expose this as a service rather than a CLI invocation.

**C. xdotool for window telemetry:** This is actually the right call and I hadn't thought of it. `diogenes.py` has the psutil hardware loop (CPU/RAM/GPU/VRAM/thermals) — it just has no window context feed because the ActivityWatch server isn't running. `xdotool getactivewindow getwindowname` in a 30-line background loop dumping to `lucidota_control.hw_telemetry_log` with a `window_title TEXT` column is exactly the right lightweight solution. Will implement.

---

## Black Boxes — Answered

**Percyphon Village:**

Deterministic procedural identity scaffolds. Not agents, not simulation actors — more like zero-VRAM synthetic personas for routing substrate testing. Each villager is derived entirely from SHA-256 of a seed string. 6 archetypes: `ledger`, `runner`, `witness`, `archivist`, `carrier`, `scribe`. 128 slots total — 28 fixed identity slots mirroring CKDOG1's soul structure, 100 procedural verbosity slots. Fields: `vuuid`, `name` (Villager-0000 through 9999), `persona`, `alias`, `ternary_state` (-1/0/+1), `slots` JSONB, `relevance_confidence_bps`, `authority` (always `'procedural_scaffold_candidate_not_truth'`, enforced by constraint). The 5000 cap is a soft ceiling — a village curator worker evicts lowest-relevance rows beyond that. Mutation class is `candidate_writer` only, no canonical graph writes. They're essentially a test/simulation substrate that generates rich deterministic identities without touching model servers.

**LoRA Policy mtime Fragility:**

Not a gradient loss function — it's a routing gate implemented as a Postgres trigger. When an event's text contains `mtime_snapshot_v1` (a tag stamped during file ingestion when the source artifact's modification time is the primary evidence), the trigger fires on INSERT/UPDATE and: (1) caps the LoRA context ceiling to 256 tokens instead of 512, (2) routes date extraction to the Treelite fast path instead of DeepSeek, (3) writes the policy envelope into the event's `payload` and `certainty_trace` as JSON (`lora_swap_token_threshold: 256`, `treelite_date_router_enabled: true`). The logic is: mtime-derived signals are low-trust — a file timestamp can be wrong, modified by a copy operation, etc. So we cut the context budget and force the deterministic router rather than an expensive model call. It's a trust-adjusted resource allocation, not weight adaptation.

**CKDOG1 Kernel — 9 gRPC methods:**

This is a soul/routing kernel, not a hardware abstraction layer. The full service surface from `kernel.proto`:

1. `CreateSnapshot` — captures a graph snapshot by node UUIDs, returns snapshot ID + root hash
2. `SubmitAuthorizedJob` — submit action + endpoint + payload with idempotency key; this is the signed command envelope path
3. `GenesisInit` — one-time kernel bootstrap: choose `USER_CONTROLLED` or `SELF_REACTIVE` policy mode; immutable after first call
4. `SoulCreate` — create a villager/soul with ID (1–5000) and symbolic term
5. `TraitSet` — assign one of 12 core trait slots
6. `DomainSlotSet` — populate one of 88 domain slots with target/hash/weight in basis points
7. `StateShift` — shift ternary state (-1/0/+1) along an axis with intensity
8. `Route` — deterministic routing query: given soul + global symbol → returns trace hash + route JSON
9. `DeltaApply` — apply a temporal delta (conversation bytes) to soul's domain slots; returns delta hash + projection hash

The Ed25519 signatures aren't guarding raw hardware ops — they're preventing injection into the routing and soul-mutation path. All decisions hash-deterministic, policy-mode-gated. No English-word intent parsing anywhere in the kernel.

---

## What You Want to Know More About

**AGE Graph Layout:**

The primary GO graph is relational, not Cypher-first. `graph_item` (nodes): `uuid`, `term`, `label`, `status`, `ternary_valency INTEGER`, `payload JSONB`, `operator_uuid`, location fields, approval timestamps. `graph_edge`: `edge_uuid`, `source_uuid`, `target_uuid`, `edge_type`, `term`, `relationship_family`, `status`, `ternary_valency INTEGER`, `valid_from/to`, `detail JSONB`. `ternary_valency_summary`: `layer`, `graph_item_count`, `positive_count`, `neutral_count`, `negative_count`, `net_spatial_polarity`, `kleene_k3_state`. The ternary isn't floating-point sentiment — it's balanced ternary logic: -1 (negated/contradicted), 0 (indeterminate), +1 (affirmed). `kleene_k3_state` maps these to Kleene's three-valued logic. Cypher via AGE is available for traversal queries but the write path is strictly relational with the promotion gate enforced via `SET LOCAL lucidota.graph_promotion_path='on'` — a Postgres session var the write barrier checks before allowing inserts into canonical graph tables.

**River ML Serialization:**

The serialization target already exists and is already used — just not in the Bytewax reflex script. `lucidota_korpus.river_model_blob` in `lucidota_storage` has `model_key TEXT`, `model_kind TEXT`, `payload BYTEA`, `sample_count BIGINT`, `detail JSONB`. `scripts/korpus_krampii.py` already reads and writes this table (SELECT payload on load, INSERT on save). The gap is specifically `lucidota_river_reflex.py` — the Bytewax reflex uses an ephemeral `Pipeline(OneHotEncoder → LogisticRegression)` that resets on every run. Fix is: on init, `SELECT payload FROM lucidota_korpus.river_model_blob WHERE model_key='bytewax_reflex_lr'` → `pickle.loads(payload)` → warm start. On shutdown, `pickle.dumps(model)` → upsert back. The infrastructure is there, the wiring isn't.

**lucidota_capped_run.sh + load_governor_decision back-pressure loop:**

The columns in `lucidota_runtime.load_governor_decision` are: `loadout_id`, `target_gpu`, `budget_vram_mb`, `observed_used_mb`, `observed_free_mb`, `estimated_required_mb`, `headroom_mb`, `decision TEXT`, `rationale`, `action_plan JSONB`, `gpu_observation JSONB`. The back-pressure loop you're describing is the right design and it's almost fully plumbed. The missing piece is the read side in the embed workers: on startup, query `SELECT decision, headroom_mb FROM lucidota_runtime.load_governor_decision ORDER BY created_at DESC LIMIT 1` — if `decision = 'defer'` or `headroom_mb < 300`, start with `--concurrency 1`. The write side (diogenes → hw_telemetry_log → governor decision insert) needs to be wired to run on a polling interval. When we tie that to the xdotool window telemetry loop, diogenes becomes the single source of system state for both the hardware interlock and the concurrency back-pressure. That's a clean closed loop and it's about 60 lines of Python.
