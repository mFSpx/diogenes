# RFC-140: Constant Learning / River, Bytewax, GLiNER, LoRA, SONA, RuVector

Status: DRAFT  
Subject ID: `constant_learning`  
Normative role: defines learning as bounded, receipt-backed adaptation over candidates and features, not autonomous truth mutation.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

LUCIDOTA should constantly learn, but “constant learning” must mean controlled update loops, not a self-mutating belief blob. The correct architecture is ABSURD/Postgres as durable workflow substrate; River/Tree-style learners as math/logic judges over bounded features; Bytewax-style streams as dataflow execution; GLiNER/local models as extraction candidates; SONA/MicroLoRA as neural repair/adaptation lanes; RuVector-like ideas as future vector/compression/witness concepts. Every crossing happens through records, verdicts, receipts, and promotion gates.

The hard law: no direct neural output to canonical graph; no direct River/Tree judge to SONA mutation; no direct SONA to ABSURD queue mutation. Everything crosses through typed records.

## 2. Sources

### Local sources

- `scripts/absurd_river_worker.py` — ABSURD wrapper for River/Bytewax/GLiNER extraction; safety laws say it writes ABSURD receipts and learning staging rows only, not Chrono temporal claims, KORPUS custody rows, or canonical graph tables.
- `00_PROJECT_BRAIN/RUVECTOR_ABSURD_SONA_RIVERML_NOTES.md` — captures RiverML/Treelite as math judge, SONA/MicroLoRA as neural/embedding surgeon, ABSURD/Postgres as durable substrate, plus hard boundaries.
- `06_SCHEMA/073_absurd_river_claim_packet_job.sql` — registers GLiNER claim-packet extraction as candidate-only with `truth_status: not_truth_claim_candidate`.
- `06_SCHEMA/004_learning_reflex.sql`, `06_SCHEMA/007_bytewax_stream.sql`, `06_SCHEMA/038_absurd_river_wrapper.sql` — learning/streaming wrapper schemas used by the worker.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md` — places constant learning in the organ map with candidate/learning authority.

### External Source Anchors

- River is an online/streaming machine learning library in Python; this matches LUCIDOTA's need for incremental judges over streams: <https://github.com/online-ml/river> and paper <https://arxiv.org/abs/2012.04740>.
- Bytewax documents Python-native stateful streaming/batch dataflows and persistent state patterns: <https://docs.bytewax.io/v0.20.1/guide/concepts/dataflow-programming.html>.
- GLiNER provides lightweight zero-shot/open NER extraction; LUCIDOTA uses that class of tool for candidate spans, not truth: <https://github.com/urchade/GLiNER> and paper <https://arxiv.org/abs/2311.08526>.
- W3C PROV-O supports recording the activities/agents/entities behind learning verdicts and repairs: <https://www.w3.org/TR/prov-o/>.

## 3. Learning Layer Contract

Constant learning MUST be separated into layers:

1. **Observation layer:** source packets, components, queue events, failures, labels, judgments.
2. **Feature layer:** bounded numeric/text features, hashes, residuals, route outcomes, extraction spans.
3. **Judge layer:** River/Treelite-style online scores, anomaly flags, verdicts.
4. **Neural repair layer:** GLiNER/LoRA/SONA candidates for extraction/style/embedding repair.
5. **Witness layer:** receipts, hashes, ancestry, branch/delta refs.
6. **Promotion layer:** graph/artifact/runtime promotion only after authority checks.

A layer may feed the next through records. It may not secretly mutate the next layer's state.

## 4. Whole-System Interaction

- **ABSURD:** durable queues/events/dead letters are the learning substrate.
- **KORPUS:** components/entities/concepts provide bounded learning inputs.
- **Indy_READs:** reading judgments and adapter candidates provide supervised-style feedback.
- **Local LLM fabric:** model/adapters participate as candidate extractors or synthesis lanes, never as untracked learners.
- **Input multiplexing:** route decisions/outcomes can become features.
- **Output hyperplexing:** learned verdicts/anomaly flags can become explicit output lanes.
- **Artifact templates:** learning reports can show verdicts, witness refs, and recommended repairs.
- **Ontology:** learning can propose GO mappings; authority gates decide promotion.

## 5. Benefit to the Whole System

Constant learning lets LUCIDOTA improve while staying sovereign and inspectable. It can learn which queues fail, which parsers misbehave, which entity labels recur, which claim packets look anomalous, which Indy judgments imply an adapter, and which workflow cuts deserve repair.

This is the difference between a living system and a static script pile. But because learning is record-mediated, the system can be wild without becoming self-corrupting.

## 6. Correctness Argument

I believe this RFC is correct because the local worker and notes explicitly encode the needed boundaries. `absurd_river_worker.py` names canonical graph tables and temporal/KORPUS tables it must not mutate, while listing learning tables it may write. The RuVector/SONA notes say RiverML/Treelite is the math judge, SONA is the neural surgeon, ABSURD/Postgres is the substrate, and no live direct wiring was performed. The claim-packet job schema declares outputs as candidates and not truth.

The correctness property is bounded adaptation. A learner can change a score, verdict, adapter candidate, or repair branch; it cannot silently change what the system says is true. That is exactly the boundary a sovereign, local, proof-hoarding system needs.

## 7. Falsifiers

This RFC is wrong if:

- learning workers write canonical graph tables directly,
- GLiNER spans become facts without review/promotion,
- SONA/LoRA updates mutate runtime behavior without receipts,
- River/Treelite verdicts call neural repair lanes directly,
- witness/provenance records are absent for learning transitions,
- dead-letter repair overwrites originals instead of branching or preserving ancestry.

## 8. Filesystem / Runtime Consequences

- Keep learning outputs in `05_OUTPUTS/absurd/`, learning schemas, or scoped runtime paths.
- Keep future RuVector/SONA implementation as a separate ordered port, not accidental vendoring.
- Store adapter candidates under runtime/model cartridge paths with provenance.
- Require worker contracts for learning workers.
- Promote learned changes only through authority gates with evidence refs and receipts.
