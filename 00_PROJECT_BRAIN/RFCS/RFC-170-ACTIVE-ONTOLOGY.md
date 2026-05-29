# RFC-170: Active Ontology / GO-25 / OBJECT-EVENT-EDGE

Status: DRAFT  
Subject ID: `active_ontology`  
Normative role: defines GO-25 plus extension terms and OBJECT/EVENT/EDGE graph shape as the active ontology layer; ROOT-414 is archived reference unless explicitly reactivated.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

LUCIDOTA's graph is RAM and storage only if the ontology is small enough to use and strict enough to prevent slop. The active ontology is GO-25 plus extensions, grounded in three graph primitives: OBJECT, EVENT, and EDGE. GO terms are the skin and routing vocabulary; the graph storage model is the durable memory.

The system should be wildly expressive through layers, payloads, evidence refs, and staging packets — not through uncontrolled term explosion. ROOT-414 is preserved outside the repo as historical/reference material. GO is active.

## 2. Sources

### Local sources

- `OFFICIAL_ONTOLOGY.json` — declares active ontology name/short name, term count, GO terms, graph SQL schema, graph policy, archived ROOT-414 path, and staging policy.
- `BOOKS/GO_ACTIVE_TERMS.json` — machine-readable GO-25 plus extension terms.
- `06_SCHEMA/016_go_graph_core.sql` — term registry, graph layers, graph items, graph edges, graph journal, staging packets, approval checks, and evidence-note requirements.
- `scripts/operator_ontology_fidelity_guard.py` — checks outputs for forbidden softened labels and blocks graph promotion when ontology fidelity fails.
- `scripts/ontology_staging_contract.py` — creates staging candidates and explicitly blocks direct truth promotion.
- `scripts/graph_promotion_gate.py` — evidence + authority + preflight gate before graph packet creation/materialization.

### External Source Anchors

- W3C OWL describes ontology languages as ways to represent knowledge about things, groups of things, and relations; LUCIDOTA implements a simpler local operational ontology rather than full OWL complexity: <https://www.w3.org/OWL>.
- W3C PROV-O supports provenance chains among entities, activities, agents, and derivations, aligning with LUCIDOTA's evidence/claim/journal needs: <https://www.w3.org/TR/prov-o/>.
- SEPIO supports claim/evidence/provenance modeling, matching LUCIDOTA's separation between claim, evidence, hypothesis, and truth: <https://ohsu.elsevierpure.com/en/publications/sepio-a-semantic-model-for-the-integration-and-analysis-of-scient/>.
- PostgreSQL constraints and schemas support enforcing term/status/evidence rules at storage time; the active graph schema uses those patterns directly: <https://www.postgresql.org/docs/current/ddl-constraints.html>.

## 3. Ontology Contract

GO's active core terms include ENTITY, ATTRIBUTE, RELATIONSHIP, FRICTION, LEVERAGE, VISIBILITY, ACTION, EVENT, TIME, PATTERN, HYPOTHESIS, CLAIM, EVIDENCE, ATOMIC_ID, SIGNAL, GLOW, TERM, TOOL, ALGORITHM, NAUGHTY, NICE, GROUP, OPERATOR, MODE, and COMMENT.

The graph collapses operational memory into:

- **OBJECT / graph_item** — things, claims, artifacts, people, concepts, tools, sources.
- **EVENT / journal/staging/workflow event** — changes, observations, approvals, ingests, actions.
- **EDGE / graph_edge** — relationships, support, contradiction, derivation, location, influence.

GO terms classify; graph rows remember; evidence refs justify; journals make mutation auditable.

## 4. Authority and Promotion

Parser/model/worker outputs stage first. Active ontology law requires:

1. staged packet or candidate,
2. term fidelity check where applicable,
3. evidence refs,
4. authority-class decision,
5. graph promotion gate,
6. operator confirmation when materialization requires it,
7. graph journal.

The schema itself enforces part of this: approved graph items require approval metadata; approved CLAIM items require evidence/proof notes; inferred HYPOTHESIS items require canonical confidence values.

## 5. Whole-System Interaction

- **Main spine:** GO graph is the memory substrate the spine protects.
- **KORPUS/ETL:** extracts files/components/entities/concepts into GO-routable candidates.
- **Indy_READs:** uses GO terms for page-locked reading and judgments.
- **PercyphonAI:** can generate procedural scaffolds but not ontology truth.
- **Language membrane:** labels output lanes without softening operator ontology labels.
- **Constant learning:** may propose mappings and anomaly flags; promotion remains gated.
- **Artifact templates:** should cite GO terms and evidence refs in reports.
- **Filesystem law:** ROOT-414 remains externalized archive; GO files remain active repo references.

## 6. Benefit to the Whole System

The active ontology makes the system legible. It gives the operator a compact vocabulary for chaotic reality without pretending every situation needs a giant academic ontology. It also enables storage constraints: CLAIM requires evidence, HYPOTHESIS can remain provisional, staging packets can wait for review, and graph journals can explain why memory changed.

The benefit is that the graph can be RAM and storage: a living working memory where uncertainty is typed instead of hidden.

## 7. Correctness Argument

I believe this RFC is correct because the local ontology files and graph schema agree. `OFFICIAL_ONTOLOGY.json` names GO as active and ROOT-414 as archived external reference. `GO_ACTIVE_TERMS.json` lists the active terms. `016_go_graph_core.sql` enforces storage tables and approval constraints. `operator_ontology_fidelity_guard.py`, `ontology_staging_contract.py`, and `graph_promotion_gate.py` implement the process boundary from extraction to promotion.

The correctness property is not “GO is metaphysically complete” and not “ontology is truth.” It is “GO is the active operational ontology that can type uncertainty and govern graph memory while preserving the distinction between reality, evidence, hypothesis, and operator-selected working reality.” That is the right goal for LUCIDOTA.

## 8. Falsifiers

This RFC is wrong if:

- active code still treats ROOT-414 as canonical ontology,
- graph writes bypass staging/promotion gates,
- CLAIM rows can be approved without evidence/proof notes,
- ontology fidelity checks allow softened replacement labels into promotion surfaces,
- GO terms are too small to express active workflows even with payloads/layers,
- graph memory cannot represent source, claim, hypothesis, evidence, action, and event separately.

## 9. Filesystem / Runtime Consequences

- Keep active ontology files in `OFFICIAL_ONTOLOGY.json`, `BOOKS/GO_ACTIVE_TERMS.json`, and graph schemas.
- Keep ROOT-414 outside active repo at `/home/mfspx/Documents/ROOT_414/`.
- New ontology candidates stage under `05_OUTPUTS/ontology/` or explicit staging tables.
- Working-reality selections use `06_SCHEMA/117_working_reality_law.sql` and `scripts/working_reality_record.py` so future minds can replay the evidence, hypothesis, move, result, and contradictions.
- Graph-affecting changes must pass promotion gates and journals.
- Tests should cover fidelity, staging, graph write blockers, and approval constraints.
