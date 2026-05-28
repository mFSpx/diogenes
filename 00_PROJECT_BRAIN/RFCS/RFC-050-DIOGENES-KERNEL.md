# RFC-050: Diogenes Kernel / Command Authority

Status: DRAFT  
Subject ID: `diogenes_kernel`  
Normative role: defines the authority kernel that keeps LUCIDOTA from treating live scripts, model suggestions, or operator impulses as ambient permission.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

The Diogenes Kernel is not one monolithic daemon. It is the authority organ: command envelopes, control packets, permission boundaries, receipts, proof custody, and graph-promotion guardrails. Its purpose is to let the operator act quickly while preventing unlabeled authority from entering queues, graph state, or external effects.

## 2. Why I Believe This Is True

Local evidence:

- `00_PROJECT_BRAIN/ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md` defines Diogenes as authority, command envelope, permission boundary, receipts, graph-promotion guard.
- `scripts/kernel_control_packet.py` creates and verifies domain-separated control packets.
- `scripts/spine_kernel_authorization.py` and `scripts/spine_authority_checker.py` enforce authority checks.
- `scripts/proof_kernel.py` is byte custody without truth promotion or graph mutation.
- `scripts/graph_promotion_gate.py` is the graph boundary.
- `00_PROJECT_BRAIN/spine_authority_registry.json` defines authority classes and effects.

## 3. External Source Anchors

- W3C PROV-O supports recording agents/activities/entities; Diogenes provides the local authority and agent/action layer: <https://www.w3.org/TR/prov-o/>.
- RFC 2119 supports making authority requirements explicit: <https://datatracker.ietf.org/doc/rfc2119/>.
- Blueprint First / Model Second supports moving command authority into inspectable contracts instead of model improvisation: <https://arxiv.org/abs/2508.02721>.
- SEPIO's claim/evidence/provenance separation supports Diogenes refusing to let candidate outputs become truth without authority and evidence: <https://ohsu.elsevierpure.com/en/publications/sepio-a-semantic-model-for-the-integration-and-analysis-of-scient/>.

## 4. Command Authority Model

A privileged action SHOULD pass through a command envelope or control packet when it can enqueue work, alter authority, promote graph state, or cause external effects.

Control packets in `scripts/kernel_control_packet.py` require:

- lane,
- valid action,
- authorized_by,
- domain-separated hash,
- optional previous packet hash,
- optional expiry.

Invalid or expired packets fail verification. That matters because live coding creates many tempting shortcuts: a shell command works, a model suggests a fix, a queue row exists, or a human is impatient. Diogenes says those facts are not the same as permission. Permission must be represented as a packet, authority class, operator confirmation, or policy row that later workers can inspect.

The kernel is therefore intentionally distributed. `scripts/kernel_control_packet.py` handles packet hashing/verification. `scripts/spine_kernel_authorization.py` and `scripts/spine_authority_checker.py` handle enforcement. `scripts/proof_kernel.py` preserves byte custody. `scripts/graph_promotion_gate.py` handles the graph boundary. The organ is the contract across these parts, not a mascot daemon.

## 5. Authority Classes and Effects

The kernel should reason in terms of effects, not vibes. Reading a file, writing a receipt, enqueueing a job, staging a graph packet, materializing a canonical graph row, exporting a bundle, and sending an external message are different effects. They require different authority. A local model may propose any of them as text, but it owns none of them as permission.

This is why `00_PROJECT_BRAIN/spine_authority_registry.json` matters: authority is data that can be checked, not tone that must be inferred. It also explains why graph promotion is separate from proof custody. Byte custody says “these bytes existed and were stored.” Graph promotion says “this claim/relation is allowed to enter memory under this evidence and authority.” Mixing those would create authority leakage.

## 6. Whole-System Interaction

- **ABSURD** receives authorized work instead of ambient scripts.
- **Main Spine** uses Diogenes for authority boundaries.
- **Graph Gate** depends on authority/evidence checks.
- **ETL/KORPUS** can preserve data but cannot escalate to truth.
- **Models** may suggest actions but cannot authorize themselves.
- **External Effects** require explicit operator authorization.

## 7. Benefit to the Whole System

Diogenes gives the operator speed without turning speed into permission. It lets a live-coding system stay powerful while keeping a receipt chain for why an action was allowed. That is essential for self-sovereignty: the operator stays sovereign, not the model, not the script, not a stale workflow.

## 8. Correctness Argument

The kernel model is correct because authority is orthogonal to usefulness. A model output can be useful but unauthorized. A parser output can be accurate but not graph-truth. A script can work but lack permission. Diogenes separates capability from authority. That separation is the only way to make a fast local system safe: a component can be powerful, experimental, or useful while still being unable to perform privileged mutation. If this feels like friction, the answer is not to bypass it; the answer is to make the command envelope cheap, repeatable, and visible.

## 9. Falsifiers

This RFC is wrong if:

- command envelopes add friction without preventing real authority leaks,
- most useful workflows cannot name their authority class,
- receipt/control-packet hashes are not checked by workers,
- external effects remain possible without operator authorization,
- graph materialization bypasses the gate in normal operation.

## 10. Filesystem / Runtime Consequences

- Kernel/authority helpers remain active enforcement code, not just docs.
- New privileged workers must state required authority and evidence.
- Control packet receipts should be linked to ABSURD jobs when used.
- Proof kernel custody must not be described as truth promotion.
