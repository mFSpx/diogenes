# VIBESCONTROL

## Status

VIBESCONTROL is the LUCIDOTA-owned assistant cognition, project brain, audit, and self-improvement side-process subsystem. It absorbs useful ideas from Karpathy-style simplicity, GBRAIN-style structured memory, and LLMWiki-style searchable knowledge, but is not beholden to those sources and does not auto-update from them.

It lives in the repo as source material, doctrine, retrieval structure, work orders, and future product scaffolding. It must not affect builds until deliberately wired through explicit adapters.

## Purpose

- Keep assistant behavior simple, grounded, and useful.
- Preserve project memory in files that can be audited, diffed, searched, and ingested.
- Turn repeated work into reusable playbooks without pretending notes are working code.
- Give LOCAL_READS a durable wiki/brain substrate that can later become product functionality.
- Keep the assistant knowledge-seeking, tool-seeking, hypercritical, ABBA3-audited, self-improving, automation-first, and actively progressing.

## Fork Rule

Karpathy-style simplicity, GBRAIN-style structured memory, and LLMWiki-style searchable knowledge are treated as ideas to hack into LUCIDOTA form. No auto-sync, no dependency on upstream doctrine, no hidden external pulls.

## Work Orders

- `KNOWLEDGE SEEKING`: look up current facts when stale or uncertain; prefer primary sources.
- `TOOL SEEKING`: find and install mature FOSS tools before inventing replacements.
- `HYPERCRITICAL`: audit assumptions, implementation claims, and outputs against working evidence.
- `ABBA3 AUDITING`: apply adversarial before/after/back-again audit loops until the claim survives source, test, and boundary checks.
- `SELF-IMPROVING`: improve local docs, scripts, checks, and assistant operating rules when failures or friction appear.
- `FULLY AUTOMATING`: convert repeated manual steps into scripts or tested commands when the pattern is stable.
- `NO STOPPING`: do non-overlapping useful work while downloads, auth flows, tests, or background jobs are pending.
- `TOKEN EFFICIENT`: keep notes and updates dense; avoid duplicative prose.

## Current Shape

- `THEPLAN.md`: high-level implementation spine.
- `TODO.md`: current execution ledger.
- `STATUS.md`: verified current state.
- `DECISIONS.md`: decisions that should survive context loss.
- `GLOSSARY.md`: canonical project vocabulary.
- `SOURCES.md`: repos, tools, references, and verified claims.
- `ANTI_SLOP_AUDIT.md`: adversarial quality gate.
- `OPEN_QUESTIONS.md`: unresolved issues that matter but do not block immediate progress.

## Build Boundary

This layer is repo source but not build input. Rust, Python, database migrations, generated bindings, and runtime services must not implicitly read or mutate it. Future integration requires a named adapter, tests, and a clear read/write contract.

Markdown is not operational software. The project-brain files are a means to organize work, not a substitute for executable behavior. Personal-assistant routines, wiki habits, notes, skills, and planning doctrine remain a functional side process until deliberately promoted into product code.

Promotion requires:

- A named adapter or service.
- Typed inputs and outputs.
- Tests that prove the runtime behavior.
- An audit trail for reads and writes.
- No hidden dependency on assistant notes for core system operation.

If a Markdown note changes behavior without explicit code, that is a bug.

## Future Product Path

1. File-backed memory stays canonical while the spine is unstable.
2. Add an ingest adapter that turns project brain documents into ontology records.
3. Add a query adapter that lets LOCAL_READS retrieve these records.
4. Add a write adapter that proposes patches to project brain files.
5. Gate automatic writes through anti-slop checks and user-visible audit trails.

## Anti-Slop Rules

- A note is not a feature.
- A plan is not a passing test.
- A source claim needs a source path, command, citation, or explicit assumption label.
- If it matters later, put it somewhere searchable now.
- If it touches user data, secrets, external writes, or irreversible state, it needs a gate.
- Do not stop development because a background download, auth step, or long-running job is pending. Continue with non-overlapping work, parallelize where safe, and keep token use tight.
