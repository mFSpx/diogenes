# Working Reality Law / Ontology Humility Contract

Status: ACTIVE SPEC SOURCE — ontology humility and working-reality selection law  
Authority: active doctrine for ontology, board moves, graph staging, reports, hunches, model output, and operator-selected action frames.

## 1. Core law

Ontology is not Truth.

Ontology is the current structured hypothesis-layer the system lives through while acting under uncertainty.

Truth is the thing the system keeps testing against. Reality exists beyond the system; LUCIDOTA does not fully possess it.

The record is for future minds to re-run, disagree, falsify, inherit, or surpass.

## 2. Working Reality Law

The operator may choose a working reality for action.

A working reality is not eternal truth. It is the current best hypothesis-set selected for movement under uncertainty.

The ontology stores:

- what we thought,
- why we thought it,
- what evidence supported it,
- what contradicted it,
- what move we made from it,
- what happened after.

The system must preserve enough record that another mind, future operator, auditor, adversary, court, friend, or stranger can reconstruct the path and think differently.

## 3. Ontology Humility Contract

Ontology is a structured hypothesis ecology.

It contains lived working reality: ideas, moments, claims, conflicts, decisions, and receipts.

The operator may select a working reality for action, but every selected reality must remain:

- inspectable,
- contestable,
- falsifiable,
- historically recoverable.

The system must not erase rejected hypotheses. It must preserve them with status, context, evidence, contradiction, and reason.

The purpose is not to win personal grudges. The purpose is to leave a record strong enough that future minds can disagree honestly.

## 4. Four layers

1. **REALITY** — the thing itself. Never fully owned by the system.
2. **EVIDENCE** — receipts, observations, logs, files, testimony, tests, hashes, timestamps.
3. **HYPOTHESIS** — the ontology's current model of what the evidence may mean.
4. **WORKING REALITY** — the subset of hypotheses the operator authorizes for action right now.

## 5. Reasoning cycle

```text
Abduction proposes.
Deduction checks.
Induction updates.
Receipts preserve.
Conflict is sacred data.
```

This is not relativism and not vibes. There is reality. The system operates through hypotheses and records the operation.

## 6. Operator role

```text
OPERATOR = abductive chooser of working reality under receipt discipline.
```

The operator is not God of truth. The operator is the action selector under uncertainty.

## 7. Board move requirement

Every move that depends on selected uncertainty should preserve this shape:

```json
{
  "evidence": ["receipt://...", "log://...", "hash://..."],
  "hypothesis": "This subsystem fails because graph writes are blocked by admission policy.",
  "working_reality": "Treat graph writes as blocked until a verified allow-gate exists.",
  "move": "Stage graph candidates instead of claiming canonical write.",
  "result": "PASS|FAIL|CONFLICT",
  "record_for_future": true
}
```

## 8. Board-game compression

Reality is the board.
Evidence is what we can see.
Ontology is our map.
Hypothesis is our planned route.
Working reality is the move we choose.
Receipts are how future players know whether we cheated.

## 9. Mutation boundary

Working-reality records may be written to Postgres, receipts, and graph-staging packets.

They must not be silently promoted to canonical graph truth. Canonical graph writes still require the graph promotion gates, evidence refs, authority class, journal, and materialization receipt.

## 10. Operator graph eligibility decision — 2026-05-27

Operator decision: most facts from **Rickshaw Robbery**, **Nordby Squeeze** / **Nordley Squeeze**, and **operator-life facts** are graph-eligible by default. The system should not freeze these domains at the staging boundary merely because a later correction may be needed.

Implementation meaning:

- Default posture is `stage_or_materialize_with_existing_graph_gates`, not indefinite quarantine.

- Repo/archive aliases such as `Nordley`, `NORDLEY_SQUEEZE`, and `NORDLEY_SQUEEZECOPY` are treated as the same local case lane as Nordby Squeeze for eligibility purposes; this is alias handling, not a separate truth claim.
- Evidence refs and authority class are still required.
- Direct canonical graph-write shortcuts remain forbidden. Use graph promotion gates / materialization helper / graph journal paths.
- If the operator says a fact is wrong later, preserve history and mark it `contested`, `corrected`, `retracted`, or `superseded`; do not delete the old path to make the graph look clean.
- This is a working-reality authorization, not a claim that every candidate is eternal truth.

Machine policy: `00_PROJECT_BRAIN/operator_graph_eligibility_policy.json`.

