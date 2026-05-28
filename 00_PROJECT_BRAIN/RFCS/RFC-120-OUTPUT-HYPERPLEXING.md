# RFC-120: Output Hyperplexing / Sequencing of Language

Status: DRAFT  
Subject ID: `output_hyperplexing`  
Normative role: defines outbound language as lane-tagged, source-aware, draft-only weaving rather than one undifferentiated model answer.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

LUCIDOTA should not output a single blob of confident prose when the system internally knows that different parts came from different authority lanes. Output hyperplexing means sequencing language by lane: deterministic template, exact quote/retrieval, model synthesis, smoothing, timeline/circadian context, warning/falsifier, and operator action surface. The user sees a coherent artifact, but the system keeps the seams visible enough to audit.

This is the language-side mirror of input multiplexing. Input separates hot and slow lanes; output weaves lanes back together without erasing provenance.

## 2. Sources

### Local sources

- `core/language_membrane.py` — deterministic intake plus output weaving across `tera_template`, `rag_quotes`, `deepseek_q4`, and `fairyfuse_smoothing`, with outbound state `draft_only`.
- `scripts/hypertimeline_engine.py` — temporal ingestion/sequencing engine for chat/social/email/SMS-like events and circadian/hypertimeline work.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md` — defines language membrane/multiplexing/hyperplexing as deterministic templates, retrieval, model synthesis, and smoothing lanes with draft/surface authority only.
- `scripts/template_contract.py` — deterministic artifact rendering lane.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/02_EXECUTION_SPINE.md` — keeps outbound artifacts away from canonical truth unless promoted.

### External Source Anchors

- Jinja's official documentation anchors the template-lane idea of rendering final text from passed data and placeholders: <https://jinja.palletsprojects.com/>.
- W3C PROV-O supports tracking generated artifacts and derivation between source entities and output entities: <https://www.w3.org/TR/prov-o/>.
- Blueprint-first design supports explicit workflow structure around model calls: <https://arxiv.org/abs/2508.02721>.
- SEPIO supports separating evidence, claims, assertions, and provenance rather than flattening them into prose: <https://ohsu.elsevierpure.com/en/publications/sepio-a-semantic-model-for-the-integration-and-analysis-of-scient/>.

## 3. Hyperplexing Contract

Output hyperplexing SHOULD label or internally preserve:

- deterministic template lane,
- exact quote/source lane,
- model synthesis lane,
- smoothing/style lane,
- timeline/order lane when applicable,
- confidence/falsifier lane,
- operator-action lane,
- outbound authority state.

`draft_only` is the default state. A polished artifact can still be draft-only. Beauty does not grant authority.

## 4. Sequencing of Language

The “sequencing of language” problem is that prose order changes belief. If a model synthesis appears before source quotes, the reader may over-trust the synthesis. If caveats appear only at the end, they may be ignored. Hyperplexing makes ordering a design surface.

A LUCIDOTA output SHOULD generally sequence:

1. decision/question,
2. source/custody frame,
3. strongest grounded facts,
4. contradictions/gaps,
5. synthesis/hypothesis,
6. recommended next action,
7. receipt/export refs.

That sequence can vary by artifact type, but it must be chosen deliberately.

## 5. Whole-System Interaction

- **Input multiplexing:** output can refer back to packet/route/lane metadata.
- **KORPUS/ETL:** exact quote and component refs supply grounded lanes.
- **Local LLM fabric:** model synthesis is one lane, never the whole artifact.
- **PercyphonAI:** procedural roles can help label output perspectives without becoming evidence.
- **Artifact templates:** provide deterministic skeletons and final packaging.
- **Indy_READs:** teammate outputs can preserve page locks, judgments, and role-mode headers.
- **Constant learning:** learned verdicts/anomaly flags can become a lane, not an invisible tone shift.
- **Main spine:** outbound surfaces remain drafts until action/promotion authority is satisfied.

## 6. Benefit to the Whole System

Hyperplexing cuts slop by preventing source, synthesis, style, and authority from collapsing into one voice. It lets LUCIDOTA produce beautiful, useful, fast outputs while still showing what came from where.

This directly supports the user's live operating style. The system can produce a readable surface now, then let the operator drill into exact quotes, model drafts, receipts, or falsifiers without rerunning the whole thought process.

## 7. Correctness Argument

I believe this RFC is correct because `core/language_membrane.py` already expresses the architecture: inbound structural routing, exact quote RAG, deterministic template lane, model synthesis lane, FairyFuse smoothing lane, and `draft_only` outbound state. `hypertimeline_engine.py` adds temporal sequencing for human-message/event streams. The runtime organs doc explicitly limits the membrane to draft/surface authority.

The correctness property is lane preservation. If an output can tell which part was template, quote, model synthesis, smoothing, or timeline structure, the operator can evaluate it. If those lanes disappear into fluent prose, LUCIDOTA has lost the plot.

## 8. Falsifiers

This RFC is wrong if:

- output lanes cannot be recovered or inspected,
- model synthesis is presented as sourced fact,
- exact quotes lack source refs,
- outbound state can silently upgrade from draft to fact/action,
- timeline order is fabricated or untraceable,
- style smoothing changes claim content without detection.

## 9. Filesystem / Runtime Consequences

- Keep output receipts with lane metadata where practical.
- Keep templates separate from model drafts and final exports.
- Preserve exact quote/source refs in review artifacts.
- Treat outbound text as draft-only unless an explicit authority gate changes state.
- Tests should assert `draft_only` defaults for language membrane output.
