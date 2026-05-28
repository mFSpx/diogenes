# RFC-090: PercyphonAI

Status: DRAFT  
Subject ID: `percyphon_ai`  
Normative role: defines PercyphonAI as a deterministic zero-VRAM procedural scaffolding organ for identity/routing/persona slots, not a truth engine and not a generic chatbot.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

PercyphonAI is important because it solves a real LUCIDOTA problem that ordinary agents keep misunderstanding: not every “AI teammate” behavior should cost a model call. PercyphonAI creates stable procedural entity masks and fluid slots using deterministic integer/hash logic. It gives the system named scaffolding for roles, villagers, witnesses, scribes, and routing surfaces without spending VRAM or pretending the generated masks are facts.

The current implementation returns 12 stable procedural slots plus configurable fluid slots, seeded by a bounded villager list and adjusted by simple psyche telemetry ratios. That is not cosmetic. It is a cheap identity lattice that lets the rest of the system coordinate weird human-scale workflows without invoking an LLM for every bit of narrative structure.

## 2. Sources

### Local sources

- `ALGOS/percyphon.py` — zero-VRAM procedural entity generator using SHA-256-derived names, aliases, UUID-like identifiers, personas, ternary offsets, and fluid slots.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md` — defines PercyphonAI as a zero/low-VRAM procedural entity/reasoning mask organ whose authority is scaffold/candidate authority only.
- `core/language_membrane.py` — consumes routing/context lanes where procedural scaffolds can help organize output without owning truth.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/02_EXECUTION_SPINE.md` — requires state transitions and graph-affecting claims to pass authority gates, which prevents Percyphon masks from becoming truth by style.

### External Source Anchors

- Python `hashlib` documents SHA-256 availability through the standard library, matching Percyphon's deterministic seed/hash implementation: <https://docs.python.org/3/library/hashlib.html>.
- NIST FIPS 180-4 anchors SHA-256 as part of the Secure Hash Standard family: <https://csrc.nist.gov/pubs/fips/180-4/upd1/final>.
- W3C PROV-O's distinction between entities, activities, and agents supports treating generated masks as attributed scaffolding rather than factual persons: <https://www.w3.org/TR/prov-o/>.
- Blueprint-first design supports moving repeatable structure into code instead of spending model calls on hidden ad hoc role invention: <https://arxiv.org/abs/2508.02721>.

## 3. PercyphonAI Contract

PercyphonAI MUST be understood as procedural scaffolding:

- It may generate stable slot names, aliases, personas, UUID-like IDs, and fluid references.
- It may use operator telemetry-like ratios to set simple offsets.
- It may help route, label, stage, or present work.
- It may support game-like or village-like operator interfaces.
- It must not claim a generated slot is an external person.
- It must not write canonical graph facts without separate evidence and authority.
- It must not require model weights.

The function's returned `zero_vram: True` is the central design clue.

## 4. Whole-System Interaction

- **Local LLM fabric:** PercyphonAI reduces model pressure by generating stable scaffolds without inference.
- **Language membrane:** procedural slots can label lanes, voices, witnesses, or work queues while output remains draft-only.
- **Indy_READs:** Indy can be more-than-agent while Percyphon supplies supporting cast/slot structure without confusing her identity with model output.
- **ABSURD workflows:** work packets can carry procedural slot labels for routing/triage, but queue state remains in Postgres/receipts.
- **KORPUS:** artifacts and components may be assigned procedural handling slots without changing source custody.
- **Ontology:** Percyphon roles can be GO-routed candidates; GO remains the active ontology and evidence remains separate.
- **Artifact templates:** templates can render stable slot labels for review surfaces and simulations.
- **Board-game lab:** procedural slots are useful for simulations, but board-game-specific material belongs outside active repo unless promoted.

## 5. Benefit to the Whole System

PercyphonAI gives LUCIDOTA cheap coherence. It lets the system preserve the operator's need for lively, abductive, role-rich interfaces while avoiding the two common failures: burning model cycles on identity decoration, or letting generated persona language masquerade as evidence.

The benefit is especially high on constrained hardware. A deterministic scaffold can be reused, diffed, cached, tested, and audited. It also makes live coding smoother: the operator can improvise workflows around stable procedural handles while the canonical spine remains boring.

## 6. Correctness Argument

I believe this RFC is correct because the implementation is small, deterministic, and explicit. `ALGOS/percyphon.py` imports `hashlib`, `json`, dataclasses, and typing; it does not import model libraries. Slot names and UUID-like values are generated from SHA-256 of the seed/index. The returned note states that identity masks are procedural and not model-generated.

The correctness property is repeatability plus non-authority. Given the same villager seed and parameters, Percyphon should produce the same scaffold. Given evidence-free procedural output, no canonical truth should change. That makes PercyphonAI a useful organ rather than a slop source.

## 7. Falsifiers

This RFC is wrong if:

- Percyphon starts importing/loading model weights,
- generated identities are treated as real-world facts,
- procedural slots mutate graph truth without evidence,
- slot output is unstable for identical inputs,
- downstream systems cannot tell Percyphon scaffolding apart from sourced evidence,
- the organ becomes a catch-all persona chatbot instead of a deterministic scaffold.

## 8. Filesystem / Runtime Consequences

- Keep PercyphonAI implementation in `ALGOS/` unless it grows into a runtime service with a new contract.
- Store Percyphon runtime receipts/outputs only when a workflow needs replayable state.
- Any graph/template/workflow use must label Percyphon output as procedural scaffold.
- Percyphon slots may be used by board-game/simulation labs, but lab assets stay outside active repo unless promoted.
## 9. Hunch Record + Subtle Knife Binding

Runtime binding: `04_RUNTIME/indy_percyphon_hunch_subtleknife_binding.json`.

PercyphonAI may attach deterministic procedural slots to operator-labeled hunch leads and Subtle Knife cut records so the board can route them cheaply. This is scaffold authority only: hunches still require evidence paths before truth promotion, and every autonomous cut must point to a receipt/log seal. Percyphon output remains `procedural_scaffold_candidate_not_truth`.

