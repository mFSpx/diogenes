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
