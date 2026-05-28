# agentSI Architecture

## Purpose

agentSI gives LUCIDOTA/INDY_READs a dedicated MVP for agent career formation:

- preserve a human-origin persona seed;
- expose a 100-question fantasy interview without making it mandatory;
- match the agent to useful job booths;
- generate a job fair board for the current mission;
- record growth lessons as career evidence;
- emit workflow events so the system can audit how the agent changes over time.

## Position in LUCIDOTA

```text
operator persona seed
        │
        ▼
agentSI profile ──► job booth matching ──► job fair board
        │                    │                     │
        │                    └──► workflow_event ◄──┘
        ▼
growth journal ──► reviewed skills / boundaries / dream jobs
```

agentSI does **not** replace ingestion, KORPUS, GO, DBOS, or Indy_READs. It sits beside them as the career/persona layer:

- ingestion handles bytes;
- KORPUS handles corpus parsing/indexing;
- GO/graph handles knowledge routing;
- Indy polycareer handles workflow-role routing and Glow Watch;
- agentSI handles identity, career booths, fit scoring, and growth journals.

## Data model

### Persona seed

`PERSONA_SEED_SCHEMA.json`

Required:

- `name`
- `human_generated_origin_persona`

Optional:

- pronouns;
- interview answers;
- hard boundaries;
- dream jobs.

### Agent profile

Stored under `04_RUNTIME/agentsi/agents/{slug}.json`.

Important fields:

- `agent_uuid`
- `name`
- `human_generated_origin_persona`
- `interview_status`
- `skills`
- `dream_jobs`
- `hard_boundaries`
- `charter`
- `growth_journal`

### Job booths

`JOB_BOOTHS.json` contains booth definitions. Each booth has:

- `id`
- `title`
- `pitch`
- `keywords`
- `outputs`
- `safety`

The first MVP has 17 booths:

1. Identity Forge
2. Career Navigator
3. Workflow Architect
4. Intake Clerk
5. Evidence Vault
6. OSINT Analyst
7. Legal Clerk
8. Newsroom Editor
9. Research Librarian
10. Market Analyst
11. Sales Strategist
12. Activist Organizer
13. Risk Analyst
14. Glow Hunter
15. Super Pal
16. Skill Gardener
17. Job Fair Host

## Workflows

| Workflow | Purpose | Command |
|---|---|---|
| `agentsi-profile-init` | Create/refresh an agent profile from the human seed. | `scripts/agentsi.py init` |
| `agentsi-job-match` | Rank career booths for an agent and task. | `scripts/agentsi.py match` |
| `agentsi-job-fair` | Generate an operator-facing job fair board. | `scripts/agentsi.py fair --write` |
| `agentsi-growth-journal` | Append reviewed lessons and inferred skill candidates. | `scripts/agentsi.py grow` |
| `agentsi-glow-absorb` | Convert Indy Polycareer Glow Watch findings into candidate growth lessons. | `scripts/agentsi.py absorb-glow` |
| `agentsi-mvp-demo` | One-shot demo that initializes, fairs, and journals. | `scripts/agentsi.py mvp-demo` |

## Scoring

The MVP scoring is intentionally boring and inspectable:

- gather words from the persona seed, existing skills, dream jobs, and current task;
- match those against booth keywords;
- add small bonuses for explicit dream jobs and high-priority booths;
- return the ranked list with visible keyword hits.

This is deliberately not model-magic. It is explainable enough to trust during live system work.

## Glow Watch integration

`scripts/agentsi.py absorb-glow` reads `05_OUTPUTS/indy_polycareer/glow_watch_findings.jsonl` and turns high-scoring discoveries into growth-journal candidate entries. This does not silently rewrite doctrine: entries are marked `promotion_status=candidate_needs_operator_review`.

## Promotion model

agentSI treats growth as a staged process:

1. **Observation** — work happens, a lesson is seen.
2. **Journal** — a growth entry records the lesson and source.
3. **Candidate skill** — inferred skills are appended transparently.
4. **Operator review** — bigger identity/mission changes must be explicitly promoted later.
5. **Doctrine** — only reviewed methods become default behavior.

## Safety and boundaries

agentSI self-sovereignty means an operational contract, not legal independence:

- no claim of legal personhood;
- no authority to bypass the operator;
- no autonomous external writes without the existing gates;
- no unsafe surveillance, harassment, exploitation, or illegal tasking;
- no raw-evidence overwrite by summaries;
- uncertain claims remain marked as uncertain.

## MVP non-goals

- no autonomous web agent;
- no unsupervised external contact;
- no hidden memory promotion;
- no replacement of legal counsel, newsroom editorial accountability, or evidence custody rules;
- no persona fabrication beyond the human seed unless clearly marked as generated/candidate material.
