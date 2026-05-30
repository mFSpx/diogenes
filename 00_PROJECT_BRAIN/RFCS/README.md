# LUCIDOTA RFC Program

Status: ACTIVE RFC WORKBENCH  
Purpose: master-thesis-tier explanation and audit of the real LUCIDOTA system.

This directory is where deep arguments live. The five minimum docs remain the short canonical map. RFCs prove and refine that map with sources, implementation evidence, tradeoffs, and falsifiers.

## Required RFC Shape

Every RFC MUST answer:

1. **What is the claim?**
2. **Why do I believe it is true?**
3. **What local evidence proves or weakens it?**
4. **What external sources support the design pattern?**
5. **How does it interact with the whole system?**
6. **How does it benefit the whole system?**
7. **How can it be wrong, and how would we know?**
8. **What filesystem/runtime organization follows from it?**

## Current Files

- `RFC-000-MASTER-THESIS-PROGRAM.md` — governing RFC for the RFC effort itself.
- `RFC-001-SYSTEM-THESIS.md` — first whole-system argument tying the major organs together.
- `RFC-010-SLOP-LAWS.md` — anti-slop / anti-bullshit law.
- `RFC-020-MAIN-SPINE.md` — graph-as-memory execution spine.
- `RFC-030-FULL-ETL-PIPELINE.md` — custody-first ingest pipeline.
- `RFC-040-DEV-LIBRARY.md` — indexed reuse without authority leakage.
- `RFC-050-DIOGENES-KERNEL.md` — command authority and permission boundary.
- `RFC-060-ABSURD-WORKFLOWS.md` — durable Postgres workflow spine.
- `RFC-070-KRAMPUS-KORPUS.md` — deterministic KORPUS/Krampus ingest and externalization boundaries.
- `RFC-080-LOCAL-LLM-FABRIC.md` — local model/RAM/VRAM sovereignty fabric.
- `RFC-090-PERCYPHONAI.md` — zero-VRAM procedural scaffolding organ.
- `RFC-100-ARTIFACT-TEMPLATES.md` — deterministic source-anchored artifact generation.
- `RFC-110-INPUT-MULTIPLEXING.md` — hot-lane/slow-lane input routing.
- `RFC-120-OUTPUT-HYPERPLEXING.md` — lane-preserving output weaving.
- `RFC-130-INDY-READS.md` — Indy_READs teammate model.
- `RFC-140-CONSTANT-LEARNING.md` — bounded learning and adaptation lanes.
- `RFC-150-BOARD-GAME-LAB.md` — externalized simulation/game lab boundary.
- `RFC-160-AUTOMATION.md` — conservative watchers/schedulers/automation.
- `RFC-170-ACTIVE-ONTOLOGY.md` — GO-25 / OBJECT-EVENT-EDGE active ontology.
- `RFC-180-ABBA63.md` — abductive-not-credulous decision heuristics.
- `RFC-190-SELF-SOVEREIGNTY-OSINT.md` — lawful self-sovereignty / OSINT domain.
- `RFC-200-FILESYSTEM-LAW.md` — filesystem authority and promotion/archive law.
- `GOAL_REQUIREMENT_MATRIX.json` — requirement-by-requirement machine matrix for the user objective.
- `GOAL_COMPLETION_AUDIT.md` — generated current-state audit summary from `scripts/lucidota_goal_audit.py`.
- `RFC_SUBJECT_REGISTRY.json` — machine-readable list of subject RFCs, source paths, and status.
- `SOURCES.md` — external and local source bibliography seed.

## Relationship to Minimum Docs

Minimum docs are small enough to obey:

1. `../ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md`
2. `../ACTIVE_SPEC/02_EXECUTION_SPINE.md`
3. `../ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md`
4. `../ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md`
5. `../ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md`

RFCs may be long. The minimum docs must stay short.
