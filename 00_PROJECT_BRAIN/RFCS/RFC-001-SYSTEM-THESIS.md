# RFC-001: LUCIDOTA System Thesis

Status: DRAFT-RFC  
Scope: whole-system explanation before per-subject deep RFCs.  
Claim: LUCIDOTA is best modeled as a local ontology-actuated intelligence operating system, not a chatbot, script pile, or conventional app.

## 1. What I Believe Is True

I believe the system is coherent when viewed as this spine:

```text
operator intent / raw artifact
  -> custody and command authority
  -> durable workflow
  -> deterministic hot lane
  -> adaptive slow lane
  -> candidate evidence / claim / hypothesis
  -> ontology/graph staging
  -> authority gate
  -> receipt-backed output
```

This belief is not based on the names. It is based on the repeated structure found in current files:

- `ACTIVE_SPEC/02_EXECUTION_SPINE.md` defines the execution path.
- `ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md` defines raw storage before derived cognition.
- `ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md` maps named organs to concrete source files.
- `06_SCHEMA/035_absurd_queue_spine.sql` and `scripts/absurd_queue_spine.py` show durable queue-spine intent.
- `OFFICIAL_ONTOLOGY.json` identifies GO as active ontology and ROOT-414 as archived.
- `ALGOS/percyphon.py` proves PercyphonAI is a deterministic zero-VRAM component, not just a name.
- `core/language_membrane.py` proves the language membrane has lane behavior, not just prose.

## 2. Why This Is Correct

The system’s core problem is not ordinary CRUD. It is uncertainty under live work: evidence arrives messy, operator intent changes, models may help but may lie, and old artifacts may be valuable later. A conventional app model would over-normalize too early. A pure chatbot model would hide control flow in prompts. A script-hoard model would preserve flexibility but lose authority and predictability.

The correct architecture is therefore a graph-and-workflow exocortex:

- The graph provides addressable memory: objects, events, and edges.
- The workflow spine provides predictable execution.
- The custody/ETL layer prevents interpretation from destroying source truth.
- The authority gate prevents candidates from becoming truth silently.
- The local model fabric gives abductive power without surrendering sovereignty.
- The Dev Library supports live coding without blessing every artifact as production.

This matches external design anchors: W3C PROV-O supports modeling data lineage and provenance; SEPIO supports treating claims with evidence/provenance as first-class; PostgreSQL `SKIP LOCKED` supports queue-like concurrent consumers; RFC 2119 supports explicit requirement levels; Blueprint First / Model Second supports keeping workflow in code/schema rather than prompt fog.

## 3. Whole-System Interactions

- **Slop law** keeps every organ honest about authority and evidence.
- **Main Spine** is the bus each organ must map onto.
- **ETL/Krampus/KORPUS** feed the graph memory without pretending to be truth.
- **Diogenes** constrains commands and privileged effects.
- **ABSURD** moves work predictably.
- **PercyphonAI** provides cheap deterministic scaffolding.
- **Indy_READs** provides stable teammate identity and reading/synthesis memory.
- **Local LLM fabric** supplies bounded extraction/drafting/ranking.
- **Language membrane** sequences output lanes so templates, retrieval, model synthesis, and smoothing remain inspectable.
- **Ontology** gives compact symbolic handles.
- **Receipts** make memory auditable.

## 4. Benefit to the Whole System

This model benefits the whole system because it lets the operator stay fast without surrendering control:

- live coding can reuse parts instead of reinventing them,
- OSINT work can preserve evidence without premature accusation,
- reports can be generated from traceable candidates,
- models can contribute without becoming rulers,
- failures become dead letters instead of hidden rot,
- weird ideas remain explorable without being promoted as facts.

## 5. Falsifiers

This RFC is wrong if inspection shows any of the following are unavoidable:

- the actual runtime depends on hidden model control flow,
- ordinary workers must write canonical graph truth directly,
- evidence cannot be traced back to source/custody,
- the named organs have no executable/source evidence,
- the filesystem cannot be reorganized without destroying required behavior,
- active ontology is actually ROOT-414 rather than GO-25.

Current evidence does not prove those falsifiers. It does prove incomplete implementation and scattered authority, which is why this RFC program exists.

## 6. Filesystem Consequence

This RFC justifies the current cleanup direction:

- keep the five minimum docs short,
- put deep arguments under `00_PROJECT_BRAIN/RFCS/`,
- externalize ROOT-414 and Ahoy because they are reference/lab material, not active spine,
- keep Dev Library as indexed reuse, not authority,
- use receipts for every move and every claim of completion.
