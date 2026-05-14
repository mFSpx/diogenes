# Adversarial Audit — 2026-05-13

Status: active

## Scream List

### CRITICAL: Wake Bus was seed-coupled, not transaction-coupled

- Finding: outbox rows could be created after the fact instead of atomically with workflow events.
- Risk: missed wake refs if process died before seed pass.
- Fix: Postgres trigger `workflow_event_outbox_trigger` now writes `event_outbox` in the same DB transaction.
- Proof: harness inserts `workflow_event` and fails unless matching outbox row exists.

### CRITICAL: Hydra v0 JSON output lied when graph write rolled back

- Finding: Hydra wrote capture metadata in graph DB, then attempted workflow event in graph DB where control schema does not exist. The caught exception left transaction state unsafe and capture rows were not persisted despite JSON success.
- Risk: fake-green evidence capture.
- Fix: graph DB transaction commits capture metadata first; state DB workflow event emit is separate best-effort.
- Proof: psql row count confirms capture row; Hydra policy evaluator reads it.

### HIGH: CAS manifest is not authoritative semantic ownership

- Finding: GC initially counted `cas_manifest` as a reference, hiding phantom blobs created by dual-write failure.
- Risk: orphan bytes could be misclassified as owned.
- Fix: GC compares only authoritative metadata refs; manifest is byte index only. Local CAS journal adds recovery context.
- Proof: harness phantom blob test expects `orphan_candidates: 1`.

### HIGH: Tracked docs carried sensitive/trigger-prone Drive metadata

- Finding: tracked notes named private archive categories and old noncanonical record details.
- Risk: metadata leakage / classification noise.
- Fix: granular metadata moved to ignored vault; tracked docs summary-only.
- Proof: redaction scanner green.

### MEDIUM: Hydra visual policy did not exist yet

- Finding: capture existed without profile-specific alert policy.
- Risk: all changes become noise, or visual changes get ignored globally.
- Fix: watcher profiles + decision table + policy evaluator added.
- Proof: `lucidota_hydra_policy.py` returns decision row.

## Active Remaining Risks

- Browser screenshot/DOM capture not yet implemented; Hydra v0 is HTTP-body only.
- Visual hash/SSIM/bounding-box channels are planned, not implemented.
- DBOS does not yet own every side effect.
- Least-privilege DB roles not implemented.
- CAS encryption-at-rest not implemented.
- Drive imports remain category-mapped, not fully private-vault ingested.

### HIGH: Browser fallback tried to become a dependency sink

- Finding: distro Chromium attempted to route through a broken Snap wrapper and stalled the build path.
- Risk: heavy dynamic capture dependency undermines adapter-first/local-light posture.
- Fix: browser capture is optional and reports `skipped` for missing/unusable browser; adapter registry makes browser fallback explicit and low-priority.
- Proof: full harness passes with browser fallback skipped safely.

### MEDIUM: Extractor strategy needed canonical priority

- Finding: without an adapter registry, browser automation could become the lazy default.
- Risk: brittle, slow, noisy extraction.
- Fix: `lucidota_extract.adapter` registry: static/schema/file adapters first, browser fallback candidate priority 90.
- Proof: `lucidota_extractor_registry.py` reports `browser_default: false`.

### MEDIUM: Executable paths needed plain-language enforcement

- Finding: code/schema comments can drift into project-lore phrasing.
- Risk: unclear production behavior and avoidable review friction.
- Fix: `lucidota_code_language_scan.py` scans executable/schema/harness paths; project records and immutable ontology are out of scope.
- Proof: scanner is wired into full harness and passes.

### MEDIUM: Local algorithm primitives needed isolated wrappers

- Finding: HDC, NLMS, SSIM, and MinHash are useful but should not enter core runtime as framework dependencies.
- Fix: added small standalone modules under `ALGOS/` plus smoke test.
- Proof: `lucidota_algos_smoke.py` passes inside full harness.
