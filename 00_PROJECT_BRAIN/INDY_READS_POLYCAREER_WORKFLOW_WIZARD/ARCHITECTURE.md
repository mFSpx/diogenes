# Architectural Design Document: INDY_READs Polycareer Workflow Wizard

**Subproject:** `00_PROJECT_BRAIN/INDY_READS_POLYCAREER_WORKFLOW_WIZARD`  
**Runtime target:** `indy-polycareer-workflow-wizard`  
**System owner:** LUCIDOTA / CLAWD / INDY_READs  
**Status:** architecture + wiring seed  
**Date:** 2026-05-15

## 1. Executive summary

INDY_READs already exists as a page-locked reading companion, book watcher, chunk embedder, and LoRA-cartridge staging surface. This subproject extends that foundation into a **polycareer workflow wizard** for a one-person investigative force.

The goal is not to make Indy a generic chatbot. The goal is to make her a disciplined role-router that combines:

- institutional workflow standards from intelligence, forensics, fraud examination, legal support, journalism, research, security, sales, marketing, and organizing;
- LUCIDOTA custody primitives: hashes, CAS, Postgres, graph items, workflow events, KORPUS, and dashboards;
- anomaly hunting: identifying people and methods that "glow" because they outperform institutions, produce receipts, and provoke status-quo backlash.

North star:

> **Boring custody. Weird insight. Ruthless routing. Beautiful reports.**

## 2. Existing foundation

Existing Indy_READs footprint found in this repo:

- `BOOKS/.indy_reads/` — state, page caches, parser caches, judgment CSV.
- `scripts/indy_reads.py` — page-locked reading game.
- `scripts/lucidota_indy_reads_watcher.py` — book watcher.
- `scripts/lucidota_indy_library_ingest.py` — extraction/chunking/embedding/LoRA staging.
- `scripts/lucidota_indy_lora_train.py` — LoRA training readiness lane.
- `scripts/lucidota_indy_contract.py` — runtime contract renderer.
- `06_SCHEMA/017_indy_reads_library.sql` — `lucidota_indy` book/chunk/embedding/training tables.
- `04_RUNTIME/indy_reads_persona_config.json` — persona config.
- `04_RUNTIME/lora_cartridges/indy_reads__*` — per-book adapter/cartridge outputs.
- `lucidota_control.workflow_registry` already contains:
  - `indy-reads-book-watch`
  - `indy-reads-game`
  - `lora-cartridge-status`

Therefore this project should **extend** Indy_READs, not replace her.

## 3. Problem statement

A one-person investigative force must perform work normally split across many jobs:

- investigator
- fraud examiner
- OSINT analyst
- legal clerk/paralegal
- journalist/editor
- researcher/librarian
- mailroom/document tech
- executive assistant
- market analyst
- B2B salesperson/strategist
- activist/organizer
- protective risk analyst
- poet/editor/narrative designer

Current system failure mode: workflows are over-eager and under-contracted. A chaotic input can trigger too many stages at once, dashboard state can confuse historical errors with current failures, and malformed data can look like system failure.

Indy must make this boring.

## 4. Scope and safety boundary

### In scope

- Lawful OSINT and research.
- Evidence intake and organization.
- Fraud/case support without guilt declarations.
- Journalism-style verification and editing.
- Legal-clerk/paralegal-style organization without legal advice.
- Market/sales/strategy analysis.
- Advocacy/political power mapping and messaging.
- Protective risk assessment at the planning/accountability level.
- Workflow design, routing, briefing, red-team review, and product generation.

### Out of scope

- Unauthorized access.
- Illegal surveillance.
- Impersonation.
- Credential theft.
- Doxxing or targeted harassment.
- Covert HUMINT tasking or manipulation.
- Violence, tactical harm, evasion, or weaponization.
- Legal advice pretending to be counsel.
- Publication of unverified claims as fact.

## 5. Reference workflow standards distilled

Indy should use these as reference patterns, not as rigid doctrine.

| Domain | Standard pattern | Indy design translation |
|---|---|---|
| Intelligence | requirements → collection → processing → analysis → dissemination → evaluation | Every task starts with an intelligence requirement / decision question. |
| Digital forensics | collection → examination → analysis → reporting | Preserve first, process second, analyze third, report last. |
| eDiscovery/legal | identify → preserve → collect → process → review → analyze → produce → present | Case-file workflow with custody and production boundaries. |
| Fraud examination | assignment → evidence/documents/interviews → findings report | Findings only; do not declare guilt or innocence. |
| Investigative journalism | hypothesis, source development, verification, contradiction handling, publication review | Publishable outputs need source matrix, right-of-reply, and risk flags. |
| Research | preregister/scope, search, screen, include/exclude, synthesize, limitations | Research outputs need inclusion/exclusion and reproducibility notes. |
| Mailroom/document ops | receive, scan/OCR, index, classify, route, audit | Every drop gets a routing slip and document state. |
| Executive assistant | inbox/calendar/task triage, meeting prep, follow-up | Indy turns chaos into briefs, deadlines, and next actions. |
| Marketing/sales | audience/pain, positioning, qualification, decision process, next action | Market/product/sales products get stakeholder and decision maps. |
| Activism/strategy | target, power map, coalition, message, tactic ladder, evaluation | Advocacy mode must map power and escalation without unsafe tactics. |
| Protective/security risk | context, threat/vulnerability/risk, mitigation, accountability | Risk mode produces registers, triggers, mitigations, after-action reviews. |

## 6. System architecture

```text
                         +-----------------------+
                         |        CLAWD          |
                         | local operator login  |
                         +-----------+-----------+
                                     |
                                     v
+----------------+        +----------+-----------+       +------------------+
| KRAMPUS drop   | -----> | Intake / Classifier  | ----> | Workflow Router  |
| files/dirs/etc |        | preserve + inventory |       | role-mode select |
+----------------+        +----------+-----------+       +--------+---------+
                                     |                            |
                                     v                            v
                           +---------+----------+        +--------+----------+
                           | Evidence Vault    |        | Role Mode Workers |
                           | CAS/Postgres/GO   |        | Indy modes       |
                           +---------+----------+        +--------+----------+
                                     |                            |
                                     v                            v
                           +---------+----------+        +--------+----------+
                           | KORPUS / Graph    | <----> | Product Factory  |
                           | entities/timeline |        | briefs/reports   |
                           +---------+----------+        +--------+----------+
                                     |                            |
                                     v                            v
                           +---------+----------+        +--------+----------+
                           | Dashboard / DBOS  |        | Human Operator   |
                           | workflow_event    |        | approve/act      |
                           +--------------------+        +-------------------+
```

## 7. Core runtime components

### 7.1 Intake / Classifier

Receives any drop and emits a stable intake manifest.

Responsibilities:

- path inventory
- size/timestamp capture
- hash/CAS where possible
- archive/directory manifest support
- duplicate detection
- parser suitability
- risk/sensitivity hints
- workflow routing suggestions

Failure mode contract: malformed input becomes `partial`, `deferred`, or `quarantined`, not a crash.

### 7.2 Evidence Vault

Maintains the chain between raw source, extracted text, derived entities, summaries, and reports.

Required fields:

- `source_path`
- `source_sha256` or directory manifest hash
- `cas_uri`
- `mime/file_kind`
- `ingested_at`
- `parser_version`
- `custody_note`
- `extraction_status`
- `confidence`

### 7.3 Role Router

Selects one or more Indy modes from `ROLE_MODES.json`.

Routing features:

- GO-25 route terms
- file/document class
- user command verbs
- case/project context
- confidence and urgency
- explicit operator override

Example:

```json
{
  "input": "organize this complaint package into a case timeline and publication memo",
  "routes": ["LEGAL_CLERK", "NEWS_EDITOR", "EVIDENCE_VAULT", "OSINT_ANALYST"]
}
```

### 7.4 Product Factory

Turns analyzed material into products.

Standard product headers:

- mission/question
- source set
- custody state
- findings
- confidence
- contradictions
- gaps
- risk flags
- next actions
- appendix/evidence references

### 7.5 Glow Hunter

A dedicated anomaly-detection role for people/methods/workflows.

Glow means:

- unusual method
- unusually high output quality
- receipt-backed claims
- repeatability
- internet/institutional backlash because the method threatens incumbents
- teachability for a one-person force
- ethical survivability

Glow Hunter output:

```json
{
  "subject": "name/org/method",
  "domain": "journalism|OSINT|fraud|sales|organizing|research|security|art",
  "glow_score": 0-100,
  "method_pattern": "what they do differently",
  "evidence_of_quality": [],
  "backlash_map": [],
  "transferable_moves": [],
  "risks": [],
  "should_indy_learn": true
}
```

## 8. Role-mode specification

The machine-readable seed lives in `ROLE_MODES.json`. Initial modes:

1. `INTAKE_CLERK`
2. `EVIDENCE_VAULT`
3. `OSINT_ANALYST`
4. `FRAUD_EXAMINER`
5. `LEGAL_CLERK`
6. `NEWS_EDITOR`
7. `RESEARCH_LIBRARIAN`
8. `EXEC_ASSISTANT`
9. `MAILROOM_TECH`
10. `MARKET_ANALYST`
11. `SALES_STRATEGIST`
12. `ACTIVIST_ORGANIZER`
13. `RISK_ANALYST`
14. `POET_EDITOR`
15. `GLOW_HUNTER`

## 9. Data model proposal

This design can begin as files + workflow registry. Later, add SQL tables if needed:

```sql
lucidota_indy.workflow_mode
  mode_id text primary key
  mission text
  safety_boundary text
  input_schema jsonb
  output_schema jsonb
  status text

lucidota_indy.workflow_run
  run_uuid uuid primary key
  mode_id text
  source_ref jsonb
  question text
  status text
  output_ref jsonb
  confidence_bps integer
  created_at timestamptz
  finished_at timestamptz

lucidota_indy.glow_profile
  profile_uuid uuid primary key
  subject text
  domain text
  glow_score integer
  method_pattern text
  evidence jsonb
  backlash jsonb
  transferable_moves jsonb
  risks jsonb
```

Do **not** add these tables until the file-mode prototype proves useful.

## 10. Integration with current LUCIDOTA/CLAWD

Current integration points:

- CLAWD runtime: `scripts/lucidota_clawd_runtime.py`
- Workflow registry: `lucidota_control.workflow_registry`
- Workflow events: `lucidota_control.workflow_event`
- Indy library: `lucidota_indy.*`
- KORPUS mass ingest: `lucidota_korpus.*`
- Evidence/CAS: `lucidota_spine.*`, `lucidota_go.*`, local vault outputs
- Dashboard: `05_OUTPUTS/ingestion_live_dashboard.html`
- Big drops: `KRAMPUSCHEWING/`

Target new registry row:

- `indy-polycareer-workflow-wizard`

Initial command can remain doc/prototype until implementation exists:

```text
scripts/lucidota_basic_workflows.py --json indy-polycareer-status
```

or future:

```text
scripts/lucidota_indy_polycareer.py route --input <text|file|case>
```

## 11. Implementation phases

### Phase 0 — Architecture seed

- Create this subproject.
- Register `indy-polycareer-workflow-wizard` as prototype.
- Emit workflow event.

### Phase 1 — File-mode prototype

- Load `ROLE_MODES.json`.
- Add a script: `scripts/lucidota_indy_polycareer.py`.
- Commands:
  - `status`
  - `modes`
  - `route --message TEXT`
  - `glow-score --subject NAME --notes FILE`
  - `product-template --mode MODE`

### Phase 2 — Integration with intake

- KORPUS/KRAMPUS can ask Indy for role routing after classification.
- Intake dashboard shows top suggested mode per drop.
- Bad/malformed files are routed to `MAILROOM_TECH` / `EVIDENCE_VAULT` instead of error theater.

### Phase 3 — Product factory

Generate first useful products:

- case chronology
- evidence matrix
- OSINT source matrix
- publication risk memo
- research synthesis
- market/competitor brief
- power map
- glow profile

### Phase 4 — Learning loop

- Operator grades products.
- Store judgments in Indy runtime tables or `.indy_reads` CSV.
- Promote effective patterns into role-mode prompts/templates.
- Maintain ethical boundary tests.

## 12. Product templates

### Investigation brief

```markdown
# Investigation Brief

## Mission
## Key findings
## Evidence table
## Timeline
## Actors/entities
## Contradictions
## Gaps
## Confidence
## Risks
## Next actions
```

### Legal clerk memo

```markdown
# Clerk Memo

## Question presented
## Short answer / working view
## Facts from record
## Issues
## Authorities / rules to check
## Evidence index
## Deadlines
## Questions for counsel/operator
```

### News editor memo

```markdown
# Publication Review Memo

## Story thesis
## Verified facts
## Unsupported claims
## Source matrix
## Right-of-reply targets
## Defamation/privacy/safety risk
## Required edits
## Publication recommendation
```

### Glow profile

```markdown
# Glow Profile

## Subject
## Domain
## Why they glow
## Method pattern
## Evidence of quality
## Backlash / screech map
## Transferable moves for Indy
## Risks / ethics
## Should Indy learn this?
```

## 13. Success metrics

Indy is useful if she reduces operator load in these ways:

- Time from drop to manifest.
- Percent of malformed files handled without crash.
- Number of products generated with evidence references.
- Reduction in stale/fake-running batches.
- Operator acceptance rate of routed modes.
- Number of useful Glow Hunter patterns extracted.
- Fewer “what the hell is happening?” moments in dashboard.

## 14. Seed references

- Intelligence cycle: https://www.intelligence.gov/index.php/how-the-ic-works
- FBI intelligence cycle graphic: https://www.fbi.gov/image-repository/intelligence-cycle-graphic.jpg/view
- NIST SP 800-86 digital forensics: https://csrc.nist.gov/pubs/sp/800/86/final
- EDRM model / discovery lifecycle: https://edrm.net/resources/frameworks-and-standards/edrm-model/
- ACFE report writing manual/product page: https://www.acfe.com/training-events-and-products/books-and-manuals/product-detail?s=Report-Writing-Manual
- ACFE fraud report writing article: https://www.acfe.com/fraud-magazine/all-issues/issue/article?s=2002-septoct-strong-written-reports
- SPJ Code of Ethics: https://www.spj.org/spj-code-of-ethics/
- GIJN citizen investigations planning: https://gijn.org/resource/citizen-investigations-planning-and-carrying-out-an-investigation/
- PRISMA 2020 checklist: https://www.prisma-statement.org/prisma-2020-checklist
- HubSpot inbound marketing: https://www.hubspot.com/inbound-marketing
- MEDDICC/MEDDPICC methodology: https://meddicc.com/meddpicc-sales-methodology-and-process
- NEA Power Mapping 101: https://www.nea.org/professional-excellence/student-engagement/tools-tips/power-mapping-101
- ISO 18788 private security operations management: https://www.iso.org/standard/63380.html

## 15. Design mantra

Indy_READs should feel like this:

> She opens the mail, preserves the receipts, knows which desk it belongs on, writes the first useful memo, spots the weird genius, and never lets the operator drown in the pile.

## 16. Built-in hook: Glow Watch

The first runtime hook for this subproject is now:

```text
scripts/lucidota_indy_polycareer.py watch-once
```

Purpose:

- watch CLAWD workflow events;
- watch local Claw agent artifacts in `01_REPOS/claudecode/rust/.claw-agents`;
- optionally watch explicit scratch files passed with `--path`;
- route observed text into Indy role modes;
- score whether the artifact contains a "glow" pattern: a strange, high-output, teachable method with receipts/backlash/repeatability;
- write findings to `05_OUTPUTS/indy_polycareer/glow_watch_findings.jsonl` and `05_OUTPUTS/indy_polycareer/glow_watch_latest.md`;
- emit `lucidota_control.workflow_event` rows under `indy-polycareer-glow-watch`.

This is deliberately deterministic and local. It does not call external LLMs. It is a hook for discovering candidate methods while the operator is working, not an automatic promotion mechanism.

Example commands:

```bash
scripts/lucidota_indy_polycareer.py --json status
scripts/lucidota_indy_polycareer.py --json route --message "organize this evidence drop into a legal/newsroom case memo"
scripts/lucidota_indy_polycareer.py --json glow-score --subject "weird operator" --text "They use public records differently, with receipts, and everyone gets mad because it works." --write
scripts/lucidota_indy_polycareer.py --json watch-once --since-hours 24 --threshold 35
```

Promotion rule:

> Glow Watch may discover and preserve candidates. Indy only learns/adopts a method after operator review.
