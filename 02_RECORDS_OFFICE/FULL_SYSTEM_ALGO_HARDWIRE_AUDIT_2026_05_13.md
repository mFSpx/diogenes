# Full System Algorithm Hardwire Audit

Date: 2026-05-13
Scope: local filesystem audit of LUCIDOTA source/docs/schemas/scripts plus nested CKDOG1/Clawd source. No Drive access.

## File coverage

- Audited source-like files: 2068
- `ALGOS/`: 39 discrete algorithm modules + `__init__.py`
- `scripts/`: 32 operational scripts
- `06_SCHEMA/`: 12 SQL schemas
- `00_PROJECT_BRAIN/`: 12 project brain docs
- `02_RECORDS_OFFICE/`: 41 records/import docs
- `01_REPOS/doggystyle`: 1422 kernel files
- `01_REPOS/claudecode/rust/crates`: 53 interface/runtime files

## math.zip intake status

`/home/mfspx/Downloads/math.zip` was copied into `02_RECORDS_OFFICE/imports/math_zip_2026_05_13/math.zip`, extracted under `.../extracted/`, and manifest-recorded. The useful primitives were reimplemented as clean, dependency-light, discrete modules under `ALGOS/`.

Added math.zip-derived modules:

- `bayes_update.py`
- `shannon_entropy.py`
- `gini_coefficient.py`
- `rsa_cipher.py` — demo integer RSA only; not production crypto.
- `doomsday_calendar.py`
- `cockpit_metrics.py`
- `bandit_router.py`
- `regret_engine.py`
- `privacy.py`
- `sketches.py`
- `perceptual_dedupe.py`
- `endpoint_circuit_breaker.py`
- `label_foundry.py`
- `semantic_neighbors.py`
- `temporal_motifs.py`
- `counterfactual_effects.py`

Algorithm smoke now covers 39 modules.

## Hardwire opportunities by subsystem

### 1. Wake Bus / workflow_event / event_outbox

Files:
- `scripts/lucidota_wake_bus.py`
- `06_SCHEMA/010_wake_bus.sql`
- `06_SCHEMA/001_lucidota_control.sql`

Hardwire pattern:
- Schoolfield gate: convert event pressure into warm/cold/overheat admission.
- Thanatosis: allow low-probability exploratory delivery/retry while decaying retries into dormancy.
- Endpoint circuit breaker: stop pounding broken workflow targets.
- Gini coefficient: detect unfair delivery starvation across workflow IDs.

Bigbrain combo:
```text
Schoolfield normalized_activity(queue_pressure)
  × Thanatosis acceptance_probability(retry_penalty, temp)
  × EndpointCircuitBreaker.allow()
  -> wake delivery/admission score
```

Use: avoid blind `deliver_pending(limit)` ordering. Rank pending outbox rows by heat, recovery odds, fairness debt, and circuit status.

### 2. Hydra capture / evidence diff

Files:
- `scripts/lucidota_hydra_capture.py`
- `scripts/lucidota_hydra_evidence.py`
- `scripts/lucidota_hydra_policy.py`
- `scripts/lucidota_hydra_browser_capture.py`
- `06_SCHEMA/011_hydra_capture.sql`

Hardwire pattern:
- Perceptual dedupe + SSIM: replace screenshot byte-hash placeholder with stable pHash/dHash + SSIM review buckets.
- Shannon entropy: flag captures that are mostly boilerplate/low-information.
- Bayes update: update source-trust posterior after repeated stable/changed captures.
- Chelydrid ambush: burst-capture high-value deltas immediately, then return dormant.
- MinHash: near-duplicate content across capture sources.

Bigbrain combo:
```text
Possum cluster suppression on similar sources
  + MinHash/SSIM near-duplicate evidence grouping
  + Bayes posterior(source_truth | stable_hashes, policy_decisions)
  -> capture priority and reviewer queue
```

Use: Hydra stops treating every URL equally; it captures novel/high-signal evidence first.

### 3. Scout / Hop Pivot / authorized extractors

Files:
- `scripts/lucidota_scout.py`
- `scripts/lucidota_hop_pivot.py`
- `scripts/lucidota_extractor_registry.py`
- `06_SCHEMA/003_scout_protocol.sql`
- `06_SCHEMA/008_hop_pivot.sql`
- `06_SCHEMA/012_authorized_extractors.sql`

Hardwire pattern:
- Possum filter: suppress clustered duplicate pivots by host/category/path/address where metadata exists.
- Pheromone + Regret + Bandit: choose next pivots from past reward, current utility, and exploration need.
- OPOSSUM/RBF surrogate: cheaply estimate expensive scrape payoff before running browser/extractor levels.
- Hoeffding bound: promote a source only after enough stream evidence beats uncertainty.
- Capybara movement: population-style exploration of route/search parameters.

Bigbrain combo:
```text
RBF surrogate predicts pivot payoff
  -> Bandit chooses extractor/route
  -> Pheromone reinforces successful graph edges
  -> Hoeffding gates promotion when evidence gap is statistically enough
```

Use: turn Hop Pivot from bounded BFS into a learning crawler without losing safety gates.

### 4. CAS / Vault / GC

Files:
- `scripts/lucidota_cas_index.py`
- `scripts/lucidota_cas_gc.py`
- `scripts/lucidota_cas_journal.py`
- `06_SCHEMA/005_cas_manifest.sql`

Hardwire pattern:
- Count-min sketch: cheap frequency hints for hot CAS blobs.
- HyperLogLog-lite: rough unique-source volume by run/source.
- MinHash: duplicate text artifacts.
- Cockpit honesty + audit debt: do not show green if CAS GC has unreviewed orphan/quarantine candidates.
- RSA file stays demo-only; use it for educational smoke, not vault encryption.

Bigbrain combo:
```text
CAS manifest refs
  + count-min hotness
  + MinHash duplicate clusters
  + audit_debt(orphan_candidates)
  -> no-delete GC priority report
```

Use: safer GC triage before any future destructive mode.

### 5. River / Bytewax / Treelite reflex spine

Files:
- `scripts/lucidota_river_reflex.py`
- `scripts/lucidota_bytewax_mini.py`
- `scripts/lucidota_treelite_router.py`
- `06_SCHEMA/004_learning_reflex.sql`
- `06_SCHEMA/007_bytewax_stream.sql`
- `06_SCHEMA/009_treelite_router.sql`

Hardwire pattern:
- Temporal motifs: mine repeated workflow sequences and failure motifs.
- Shannon entropy: route/event diversity; low entropy means stuck loop.
- Fold-change detection: detect sudden spike in failures/captures/events.
- Hoeffding: decide when enough stream observations justify a route split.
- Counterfactual effects: did routing policy actually improve outcome, or did it just correlate?

Bigbrain combo:
```text
Bytewax stream -> temporal motifs + entropy + fold-change
  -> River online scores
  -> Treelite route hints
  -> CounterfactualEffect audit on policy changes
```

Use: make the reflex team self-auditing instead of just producing hints.

### 6. Clawd / operator UI / Big Board

Files:
- `scripts/lucidota_big_board.py`
- `01_REPOS/claudecode/rust/crates/claw-cli/src/main.rs`
- `00_PROJECT_BRAIN/BUILD_PLAN_AUDIT.md`

Hardwire pattern:
- Cockpit honesty: every green cell must be backed by a smoke/test/source counter.
- Anti-slop ratio: claims in reports need evidence pointers.
- Audit debt: every export/demo with missing audit step adds visible debt.
- Serpentina self-righting: if UI/session state flips, recover from local state snapshot before full reset.

Bigbrain combo:
```text
CockpitHonesty(displayed_ok, unknown_ok)
  + AntiSlopRatio(claims_with_evidence, claims_total)
  + AuditDebt(exports_missing_audit)
  -> Big Board truthfulness banner
```

Use: prevent fake-green dashboards.

### 7. Model runtime / local brain

Files:
- `scripts/lucidota_runtime_smoke.py`
- `scripts/lucidota_model_registry.py`
- `scripts/lucidota_record_runtime_inventory.py`
- `06_SCHEMA/002_model_runtime.sql`

Hardwire pattern:
- Schoolfield: model preheat/load admission based on activity and VRAM pressure.
- Regret engine: choose local model route by expected value minus cost/risk.
- Bandit router: learn best model/tool action per context.
- Gini: fairness across resident model slots / avoid one model starving others.
- Endpoint circuit breaker: provider/model health fallback.

Bigbrain combo:
```text
Schoolfield(VRAM/activity temperature)
  + RegretWeightedStrategy(EV-cost-risk)
  + Bandit reward updates from workflow_event outcome
  -> model load/route governor
```

Use: model runtime stops being static inventory and becomes policy-governed.

### 8. CKDOG1 kernel / ontology / semantic route

Files:
- `01_REPOS/doggystyle/**`
- `scripts/lucidota_age_edges.py`

Hardwire pattern:
- HDC: symbolic route embeddings / fast approximate memory.
- Semantic neighbors: local enclave retrieval.
- Bayes update: claim belief updates from evidence.
- Label foundry: weak labels for imported records.
- Voronoi / Fisher localization: partition attention/search space.

Bigbrain combo:
```text
HDC bind(entity, evidence)
  + Bayes posterior over claim truth
  + SemanticNeighbors for recall
  -> CKDOG1 route confidence with provenance
```

Use: bridge current smoke route into evidence-aware recall.

### 9. Security / auth / privacy

Files:
- `scripts/lucidota_security_scan.py`
- `02_RECORDS_OFFICE/CYBERSECURITY_RISK_REGISTER.md`
- `00_PROJECT_BRAIN/DECISIONS.md`

Hardwire pattern:
- Privacy anonymization before indexing.
- Reconstruction risk score for quasi-identifiers.
- Entropy scan for suspicious token-like strings.
- Honey-snapping is defensive-only: detect inbound suspicious events, capture/quarantine locally, never retaliate.

Bigbrain combo:
```text
Entropy(token-likeness)
  + Privacy reconstruction risk
  + Cockpit audit debt
  -> release sanitization gate
```

Use: block commits/exports that leak auth material or fake sanitized status.

## Best immediate hardwire targets

1. `lucidota_wake_bus.py`: add algorithmic ranking/admission without changing truth schema.
2. `lucidota_hop_pivot.py`: Possum + Bandit + RBF + Hoeffding for pivot selection/promotion.
3. `lucidota_hydra_evidence.py`: pHash/SSIM/MinHash/Bayes evidence priority.
4. `lucidota_big_board.py`: cockpit honesty / anti-slop / audit debt banner.
5. `lucidota_runtime_smoke.py` + model registry: Schoolfield + regret model route governor.

## Rule

Algorithms are advisory unless promoted through DBOS/Postgres schema. They may rank, gate, explain, or warn; canonical state remains in durable tables and source files.
