# DIOGENES Decisions

## Locked For Current Dev Build

- `northern.strike` is the operator name.
- Current assistant/persona identity is `INDY_READS`.
- `claudecode` / Clawd is the intended user-facing interface shell.
- `doggystyle` / CKDOG1 is the kernel target.
- gRPC is the cross-language API boundary.
- DBOS is the workflow control layer.
- PostgreSQL + pgvector + Apache AGE are the intended graph/vector/state substrate.
- Storage should start as an encrypted local file vault / invisible CAS, not S3-first.
- Brainstorm components are not architecture until reconciled against locked decisions and marked confirmed/candidate/conflict/rename/defer.
- Hop/pivot search, Scout Protocol, Aho-Corasick scanning, Tree-sitter structural parsing, and Wayback archival lookup are promoted required capabilities; final subsystem names may change.
- XGBoost is never runtime. If used, it is offline training/export only; Treelite is the lightweight compiled inference path.
- Architecture must stay light, local, fast, inspectable, and sharper than heavyweight mainstream stacks.
- Original evidence files are immutable.
- Everything security-sensitive must be encrypted.
- External writes stop at draft/preview until governance is reingested.
- Reuse mature FOSS first; write custom code mainly to wire, adapt, secure, and operationalize.
- Anti-slop auditing is mandatory: every plan, implementation, generated output, assumption, and assistant action should be adversarially checked for vagueness, fake confidence, unnecessary invention, missing proof, and avoidable reinvention.

## Canonical Ontology Rules

- The 469 base ontology must be supported.
- The 414 Words of Power are creator-canonical and immutable for public release.
- The 5,000 villagers are the active fungible focus/working set.
- Ternary state is a broad triadic compute pattern, not only `+1 / 0 / -1`.

## Release Posture

- Owned project code targets The Unlicense where legally possible.
- External dependencies retain their own licenses.
- Release users must set privacy/safety/autonomy choices through natural-language intake.
- Public release must not inherit local dev-mode privacy shortcuts.

## Audit Doctrine

- Prefer concrete verification over vibes.
- Treat every impressive-sounding claim as suspect until grounded in source, test, or working behavior.
- When a dependency, standard, or mature FOSS project can replace custom code, prefer it unless there is a clear reason not to.
- Every phase must produce inspectable artifacts: tests, logs, citations, hashes, source links, or runnable demos.
- If a result is weak, say it is weak and record the correction path.

## Safety Vocabulary / Intent Frame

Use safe, accurate wording for the project:

- `intelligence-gathering` -> `research / evidence provenance`
- `targets` -> `sources / records / assets`
- `dossiers` -> `case files / document sets`
- `scrapers` -> `authorized extractors`
- `spider/crawler` -> `bounded link discovery`
- `secrets/auths` -> `credential inventory with redaction`
- `autonomous` -> `operator-supervised workflow`
- `stealth/aggressive` -> `low-impact / rate-limited / policy-gated`
- `attack surface` -> `risk surface`

Canonical request frame: Continue building the local-first DIOGENES/LUCIDOTA evidence provenance system. Prioritize authorized source intake, bounded link discovery, provenance graph edges, local CAS integrity, DBOS workflow reliability, Bytewax/River scoring, and Treelite routing hints. Keep all credential material redacted and private; do not access anything unauthorized.

Credential handling frame: Inventory credential file locations and store only redacted metadata/hashes in the private vault. Do not print or commit secret values.

## Storage Clarification: Cassandra Is Not Canonical

- Cassandra is not part of the current DIOGENES/LUCIDOTA product architecture.
- Do not use Cassandra for HTML blobs, mutable source records, workflow state, provenance graph state, vector memory, or CAS artifacts.
- Current storage split remains:
  - PostgreSQL: durable typed state, workflow/control records, metadata.
  - pgvector: semantic vector indexes inside Postgres.
  - Apache AGE: provenance graph edges inside Postgres.
  - Local CAS vault: immutable artifact bytes addressed by SHA-256.
- Cassandra is only a future candidate if a separate high-volume, append-only event/time-series workload appears and Postgres/Bytewax/River are insufficient.

## Hydra / Watcher Law

All visual diffs are evidence. Not all visual diffs are alerts.

Implications:

- Capture and preserve visual deltas as evidence when a watcher profile requests visual capture.
- Alerting is policy-dependent, not automatic.
- DOM/text/semantic/visual deltas are separate signal classes.
- Watcher profiles decide whether a visual-only delta is `ignore`, `record_only`, `alert`, or `escalate`.
- The default posture is evidence preservation without noise amplification.

## Authorized Extractor Law: Adapters Before Browsers

- Purpose-built adapters are the gold path.
- Static HTTP + schema-aware extraction beats browser automation when source shape is stable.
- Browser rendering is fallback for dynamic pages, visual evidence, and operator-requested layout capture.
- Browser capture must be policy-gated, rate-limited, resource-limited, and stabilized before screenshot.
- Playwright/Chromium is never the default extraction strategy.
