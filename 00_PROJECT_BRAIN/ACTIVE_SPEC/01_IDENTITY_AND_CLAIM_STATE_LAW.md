<!--
DEV NOTE (anti-slop provenance, 2026-05-24): A file/report/taxonomy is proof of the truth it actually declares: this document exists, has this title, and asserts this scoped content. It is not proof of unrelated external facts, not permission to rename its scope as "the manual," and not evidence that the operator personally coined a local label. Names are integral to their own statements and contexts; do not erase, swap, or flatten them. When an agent says "this is the user's rule," it must cite direct operator instructions or accepted doctrine/RFC/receipt lineage, state exactly how many source docs were consulted and why those docs were chosen, and distinguish direct evidence from inference or compression. If the system lacks receipts for a factual CLAIM, classify it as hypothesis, observation, inference, suggestion, or confidence-rated candidate instead.
-->

# LUCIDOTA Identity and Claim-State Law

Status: ACTIVE SPEC SOURCE — identity, anti-slop, and claim-state semantics  
Purpose: define what LUCIDOTA is, what it refuses to become, and how system statements are classified before they become factual CLAIMs.

## 1. Identity

LUCIDOTA is a local-first sovereign data exocortex for intelligence work, OSINT, live coding, investigative journalism/newsroom work, reporting, evidence handling, ontology building, and self-checking. It is built for a constrained local machine and an operator who thinks in systems quickly. It should multiply that ability without letting speed become bullshit.

The product is not a chatbot, not a prompt pile, not a hidden model controller, not a conventional SaaS app, and not a perfect truth machine. The product is a controlled intelligence pipeline that preserves chaos, records custody, stages claims honestly, learns from work, and refuses to call something fact before the evidence and authority support it.

## 2. One Law

**Be honest about the relationship between code and truth.**

Everything in the system may be useful: a file, a guess, a model output, a hunch, a failure, a dead letter, a contradiction, a simulation, a human decision, or an unknown. None of those are shameful. The failure is unlabeled certainty.

Accepted states include, but are not limited to:

- observed evidence,
- source claim,
- hypothesis,
- inferred candidate,
- deterministic computation,
- model-computed finding,
- operator-authored assertion,
- operator-confirmed finding,
- disputed item,
- unknown,
- nobody-knows-yet.

If a state is labeled honestly, it can be calculated on. If it is not labeled honestly, it is slop.

## 3. Anti-Slop Law

Slop is any component, document, test, prompt, script, schema, or workflow that makes the system less honest about what exists, what works, what it knows, or what it is allowed to change.

Slop also includes overbuilt “safety” that makes the system less usable without buying proportional truth or risk reduction: redundant gigabyte logs, frozen live loops, security theater, fake compliance rituals, and low-value bureaucracy. The correct standard is proportional proof at the right boundary, not maximum friction everywhere.

Failure labels:

- `VIBE`: sounds good, not verified.
- `SLOP`: vague, bloated, fake-clear, or fake-working.
- `FANTASY`: capability claimed but not wired.
- `REINVENTION`: new custom wheel where the dev library already had a usable part.
- `UNTRACED`: no source, hash, receipt, test, or reproducible command.
- `DANGEROUS`: hidden external effect, secret leak, direct graph write, or irreversible action.
- `AUTHORITY_LEAK`: candidate/model/heuristic output silently promoted to truth or command.
- `SECURITY_THEATER`: proof/security/logging slows or bloats the system without real safety.
- `OPERATIONAL_FRICTION`: process, refusal, testing, or hygiene wastes operator time without improving truth/risk.
- `TERMINOLOGY_FOG`: made-up local names presented as real established concepts.
- `TEMPORAL_SLOP`: stale-memory answer to a current-status question.

Correction rule: fix it now, or record the smallest concrete next action that would make it honest.

Reuse rule: reinventing the wheel is slop unless it improves sovereignty, is objectively better for LUCIDOTA, is required by the system, or the operator explicitly wants a better local version.

Terminology rule: name real concepts as real concepts; label project names and metaphors as local design handles. Do not flatter the operator with fake terminology.

Size rule: no line-count number is universal truth, but most functions/tools should stay near 100 LOC or less; 500 LOC is large and should be reviewed; 2000 LOC needs serious justification unless generated/vendored/legacy or consciously bounded.

Capability preservation rule: optimization must not remove, rename, disable, or narrow existing capabilities unless the operator explicitly asks or evidence proves the capability is dead, duplicate, or superseded. Build center-out: strengthen the shared spine first, then thin adapters; do not create sideways sprawl to look productive.

Systemwide elegance rule: the language/goals standard is not a sidecar. Every subsystem should converge toward one source of truth, one source of authority/glory, many thin interfaces, ontology-driven routing, deterministic-first operation, self-teaching from receipts, self-auditing, and self-red-teaming. New helpers target ~100 LOC, but 100 LOC is a burden-of-proof trigger, not a toy law: exceeding it is allowed when cohesion, capability preservation, performance, safety, or reuse of an established/vendored tool justifies the size in writing.

Asymmetric dev wargame rule: LUCIDOTA development is a real capability loop, not a benchmark theater. Build a functional feature, exercise it, measure it, tighten it, reuse better lines of play, and record proof. Speed, knowledge, legal/intelligence research, live coding, assistant automation, correspondence, and investigative journalism are product lanes, not vibes.

No-delete rule: preserve by default. Do not delete source/history/toolbox/custody artifacts as normal cleanup. If fresh runaway logs, caches, generated junk, or system-threatening bulk must be removed, bound the target, justify it, and write a receipt; otherwise quarantine/archive/index instead of erasing.

## 4. Models, Local Sovereignty, and Hardware

Models are bounded organs, not rulers. They may extract, summarize, draft, rank, or suggest at named nodes. They do not secretly choose the workflow path, mutate canonical graph truth, send external messages, or declare completion.

The system should punch above its hardware by being local-first, FOSS-first where practical, streaming, deterministic-first, and careful with RAM/VRAM. Use small hot routers, staged model residency, LoRA lanes, Needles, deterministic math, and receipts instead of one giant vague model blob.

Default stance:

```text
deterministic first
bounded model second
receipt always
```

## 5. No Accusation Doctrine

LUCIDOTA does not need to accuse. It records evidence, writes ledgers, prepares letters/reports, exposes relationships, preserves contradictions, and tells the truth as far as the evidence permits.

Even when something appears obvious, the system should prefer:

```text
source says X
artifact shows Y
timeline supports Z
hypothesis A remains unproven
operator decision B was made at time T
```

over theatrical certainty.

## 6. Abductive but Not Credulous

The system must remain wildly abductive: it should form weird useful hypotheses, pivot searches, infer possible networks, run simulations, and follow faint signals. But it must not become credulous. It should not believe the operator merely because the operator said something, and it should not resist the operator with fake friction either.

Operator change-of-mind is normal state, not failure. Capture it as an event and keep moving.

`ABBA63` is an operator-named heuristic family for this stance. Until a formal machine registry is promoted, its canonical meaning here is:

```text
high-abduction, high-honesty, low-ego, low-unlabeled-certainty
```

## 6.1 Working Reality / Ontology Humility

Ontology is not Truth. Ontology is the current structured hypothesis-layer the system lives through while acting under uncertainty.

Reality exists beyond the system. Evidence is what the system can inspect. Hypothesis is what the ontology thinks the evidence may mean. Working reality is the subset of hypotheses the operator selects for action right now.

`OPERATOR = abductive chooser of working reality under receipt discipline.`

Every selected working reality must remain inspectable, contestable, falsifiable, and historically recoverable. Rejected hypotheses are not erased; they are preserved with status, context, contradiction, evidence, and reason. See `ACTIVE_SPEC/07_WORKING_REALITY_LAW.md`.

## 7. Weird Names Are Design Handles

These names are allowed because they point to real organs or concepts. Do not dismiss them as vibes before searching implementation evidence.

- **Diogenes Kernel** — authority, command envelope, permission boundary, receipts, graph-promotion guard.
- **ABSURD** — durable workflow spine over Postgres/work orders/events/dead letters.
- **Krampus Express / KRAMPUSCHEWING / KORPUS** — preserve-now, digest-later stomach for documents, evidence, old code, and strange artifacts.
- **PercyphonAI** — zero/low-VRAM procedural reasoning/entity-mask organ; currently evidenced by `ALGOS/percyphon.py`.
- **Indy_READs** — named teammate/reading-intake/synthesis identity; not a disposable littleworker.
- **Needles** — cheap hot scouts before expensive inference.
- **FairyFuse / ternary / Bonsai / Mamba / DeepSeek / LoRA lanes** — constrained local model fabric.
- **GO-25** — active ontology skin. ROOT-414 is archived reference unless explicitly reactivated.
- **Ahoy** — board-game/simulation lab, not production truth spine.

## 8. Active Spec Set

The active spec set is deliberately small, but it is not magic and not a license to rename any one file as “the manual.” These files are active because their scoped content declares active rules and the instruction index points at them:

1. `ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md`
2. `ACTIVE_SPEC/02_EXECUTION_SPINE.md`
3. `ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md`
4. `ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md`
5. `ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md`
6. `ACTIVE_SPEC/06_BARE_STEEL_DOCTRINE.md`
7. `ACTIVE_SPEC/07_WORKING_REALITY_LAW.md`

Everything else is a registry, schema, receipt, subsystem README, historical source, or generated output unless explicitly promoted.
