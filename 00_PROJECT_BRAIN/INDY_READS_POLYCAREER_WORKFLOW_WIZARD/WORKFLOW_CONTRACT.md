# INDY_READs Workflow Contract

This is the boring contract the system must obey before it gets clever.

## Non-negotiables

1. **Accept drop. Never crash.**
   - Bad input becomes `deferred`, `partial`, or `quarantined`, never an unhandled traceback.

2. **Preserve before interpretation.**
   - Store path, size, timestamps, SHA256/CAS URI when possible, source, and custody notes before analysis.

3. **Separate raw evidence, extracted facts, claims, and inference.**
   - Do not let summaries overwrite evidence.

4. **Classify and route.**
   - Every object gets a class: document, media, comms, legal, market, person/entity, event, claim, artifact, unknown.

5. **Queue stages separately.**
   - Intake must not block on brain mapping, report writing, OCR, embeddings, rechrono, or hardmath.

6. **Quarantine malformed or dangerous material.**
   - Keep custody. Record parser error. Continue.

7. **Dashboard shows current truth, not historical panic.**
   - Historical stack traces remain in logs, but the UI must distinguish current active error vs old scar.

8. **Every batch reconciles.**
   - `running` with no live process becomes stale/failed with reason and timestamp.

9. **Lawful/authorized workflow only.**
   - No illegal surveillance, unauthorized access, impersonation, doxxing, targeted harassment, or evasion workflows.

10. **Human decision remains human.**
   - Indy can prepare, route, score, and red-team. Operator decides legal, safety, publication, and contact actions.

## Default product format

Every output should declare:

- Mission / question
- Source set
- Custody state
- Findings
- Confidence
- Contradictions
- Gaps
- Risk flags
- Next actions
- Appendix / evidence links

## Glow Watch hook contract

1. Glow Watch observes local artifacts; it does not automatically act on people.
2. A glow candidate is not truth. It is a review item.
3. New methods are promoted only after operator review.
4. The hook must record source path/event, excerpt, score, signals, and transferable moves.
5. High weirdness without receipts is inspiration, not doctrine.

## Auto-start rule

When the operator enters through `./claw`, the Glow Watch daemon should be alive unless explicitly stopped. It is allowed to observe local workflow events and Claw agent artifacts, but it is not allowed to execute external collection or promote new doctrine without operator review.

## IronClaw runtime binding

IronClaw is the operator-facing shell for Indy_READs when the local stack is live.

- IronClaw may route reading, triage, and workflow prompts into Indy_READs 24/7.
- Groq is the default cloud reasoning fallback for hard or broad turns.
- Local model lanes remain the first-choice execution path when already admitted.
- Email, Signal, and other outbound contact hooks stay operator-gated and must not auto-send without an explicit receipt-backed action.
- Book intake, watcher loops, and LoRA staging stay custody-first: preserve, classify, queue, then promote.
- External effects require a recorded receipt and an allowed action path.
## Hunch Record + Subtle Knife Binding

Runtime binding: `04_RUNTIME/indy_percyphon_hunch_subtleknife_binding.json`.

- Operator-labeled hunches from the recorded hunch instrument are high-priority evidence leads, not generic speculation and not automatic truth.
- Indy_READs must preserve hunch source paths/hashes, separate hunch/evidence/inference/truth, and route proof work before promotion.
- Subtle Knife cuts are allowed only inside current authority and must be sealed by receipt/log; unsealed cuts are invalid.
- This binding also applies to PercyphonAI procedural slots; Percyphon may scaffold leads, never factualize them.
