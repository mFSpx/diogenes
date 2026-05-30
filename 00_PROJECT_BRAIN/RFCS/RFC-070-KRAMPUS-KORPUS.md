# RFC-070: Krampus Express / KORPUS Port

Status: DRAFT  
Subject ID: `krampus_korpus`  
Normative role: defines KORPUS as the deterministic mass-ingestion and componentization organ, not a model-thinking organ and not a dumping ground for external board-game or ROOT-414 gravity.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

KORPUS is the ported, repo-native core of the Krampus Express idea: reclaim the jungle by hashing, locking, deduplicating, componentizing, routing, and indexing artifacts before any interpretive layer gets to tell stories about them. Its job is not to be clever in the LLM sense. Its job is to be relentlessly mechanical so the rest of LUCIDOTA can be abductive without losing custody.

The correct shape is:

```text
path/root -> file hash -> content-addressed storage / occurrence
  -> file object -> component object -> entity/concept candidate
  -> GO routing / tags / minhash / embedding status
  -> ABSURD event or derived compute queue
  -> graph-promotion candidate, not automatic graph truth
```

This preserves the user's proof-hoard doctrine: index the jungle, do not pave it. Strange material can remain strange, but it becomes addressable.

## 2. Sources

### Local sources

- `scripts/korpus_krampii.py` — states the workflow as scan → SHA-256 → UUID → CAS lock → exact dedupe → componentize → entity/concept extraction → GO routing → Postgres indexes → ABSURD event, with “No LLM calls.”
- `scripts/spine_krampus_worker.py` — observation/health wrapper that writes receipts and invokes storage-only KORPUS componentization rather than mutating custody or graph truth directly.
- `06_SCHEMA/019_korpus_krampii.sql` — storage schema for ingest batches, file objects, occurrences, components, tags, entities, concepts, minhash signatures, embeddings, and graph candidate fields.
- `06_SCHEMA/020_korpus_derived_compute_queue.sql` — post-ingest derived compute queue for replay, near-duplicate scans, graph promotion, deferred parsing, and deep media extraction; explicitly says raw evidence ingest must not depend on the queue draining.
- `05_OUTPUTS/filesystem_organization/*root414*` and `*ahoy*` receipts — prove ROOT-414 and Ahoy were moved outside active repo gravity.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md`, `00_PROJECT_BRAIN/ACTIVE_SPEC/02_EXECUTION_SPINE.md`, and `00_PROJECT_BRAIN/ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md` — place KORPUS inside the ETL/spine organ map.

### External Source Anchors

- NIST FIPS 180-4 defines the SHA-2 family, including SHA-256, which supports the use of content hashes as stable artifact identity anchors: <https://csrc.nist.gov/pubs/fips/180-4/upd1/final>.
- W3C PROV-O models provenance through entities, activities, agents, and derivation chains; KORPUS locally implements the same pattern with files/components/events/receipts: <https://www.w3.org/TR/prov-o/>.
- PostgreSQL `SELECT` documentation explains `SKIP LOCKED` as useful for queue-like tables while warning about inconsistent general views; that matches KORPUS derived-compute queue semantics: <https://www.postgresql.org/docs/current/sql-select.html>.
- SEPIO's evidence/provenance/claim separation supports LUCIDOTA's refusal to treat extraction candidates as truth: <https://ohsu.elsevierpure.com/en/publications/sepio-a-semantic-model-for-the-integration-and-analysis-of-scient/>.

## 3. What KORPUS Is

KORPUS is a storage and decomposition organ. It performs:

1. file discovery with exclusion rules,
2. byte hashing,
3. deterministic UUID assignment,
4. CAS/locked storage references,
5. exact dedupe and occurrence tracking,
6. content classification,
7. line/token-bounded componentization,
8. heuristic entity/concept extraction,
9. GO routing/tagging,
10. minhash and embedding status preparation,
11. event/receipt emission,
12. enqueueing of derived work that can fail or lag without corrupting raw ingest.

That last clause matters. KORPUS can ingest now and enrich later. A huge project needs this because blocking raw custody on every expensive interpretation step would recreate slop as latency.

## 4. What KORPUS Is Not

KORPUS is not:

- a direct canonical graph writer,
- a hidden LLM reasoning lane,
- a place to keep unrelated board-game labs,
- a place to keep ROOT-414 primitive-cries as active ontology,
- a magic “knowledge base” that erases source custody,
- an excuse to mutate sovereign toolbox originals.

Ahoy now belongs under `/home/mfspx/BOARD_GAMES/AHOY/`. ROOT-414 now belongs under `/home/mfspx/Documents/ROOT_414/`. KORPUS may reference external archives as evidence or historical material; it must not let them become active repo gravity again.

## 5. Whole-System Interaction

- **Full ETL pipeline:** KORPUS supplies the middle layer between raw drop/custody and higher-level analysis.
- **Main spine:** KORPUS produces candidates/events that can move through ABSURD and graph-promotion gates.
- **Diogenes kernel:** KORPUS jobs should be invoked by command envelopes or worker contracts when mutation is involved.
- **ABSURD workflows:** derived enrichment and health checks are queue/receipt work, not invisible daemon magic.
- **Local LLM fabric:** models may summarize or classify KORPUS components later, but KORPUS identity and custody do not depend on them.
- **PercyphonAI:** procedural scaffolding can name or route work around KORPUS artifacts without consuming VRAM or changing truth.
- **Artifact templates:** reports/case packets should cite KORPUS file/component/candidate refs, not free-floating prose.
- **Indy_READs:** reading judgments and book chunks can become KORPUS-addressable artifacts while preserving page boundaries.
- **Constant learning:** River/Bytewax/GLiNER can learn over KORPUS-derived features as candidates; learning outputs still require gates.
- **Ontology:** KORPUS routes toward GO; it does not resurrect ROOT-414 as canonical active ontology.

## 6. Benefit to the Whole System

KORPUS lets LUCIDOTA ingest large, weird, contradictory, partially broken sprawl without demanding immediate understanding. That is exactly what a hypersystemia-amplifying tool needs. The operator can throw the jungle at it; the system answers with stable handles, components, occurrence records, and candidate extractions.

This benefits the whole system by making every later organ cheaper and safer:

- LLM prompts can cite component IDs instead of whole blobs.
- Graph promotion can demand evidence refs.
- Templates can include source manifests.
- Dead-letter repair can target one component rather than one mystery pile.
- Duplicate detection reduces wasted model/runtime cycles.
- Externalized labs stay available without polluting active runtime semantics.

## 7. Correctness Argument

I believe this RFC is correct because the implementation converges with the doctrine. `scripts/korpus_krampii.py` explicitly says it is deterministic and performs no LLM calls. The schema gives independent tables for batches, files, occurrences, components, tags, entities, and concepts instead of collapsing everything into a single “knowledge” blob. The derived queue explicitly decouples post-ingest enrichment from raw evidence ingest. The spine worker limits itself to observation/receipts and storage-mode componentization.

The correctness property is not “KORPUS extracts perfect meaning.” It is “KORPUS makes source material stable, addressable, deduplicated, and challengeable before meaning is argued.” That is the right property for LUCIDOTA because the project wants wild abductive motion on top of boring custody.

## 8. Falsifiers

This RFC is wrong if:

- active KORPUS ingest requires an LLM call to succeed,
- raw custody cannot be traced back from a component or claim candidate,
- useful KORPUS workflows require direct canonical graph mutation,
- ROOT-414 or Ahoy artifacts reappear as active repo dependencies,
- derived enrichment failure blocks raw evidence preservation,
- component IDs are unstable across repeat ingest of identical content.

## 9. Filesystem / Runtime Consequences

- Keep KORPUS runtime code in active repo only where it directly supports ETL/spine operation.
- Keep board-game labs under `/home/mfspx/BOARD_GAMES/` unless explicitly promoted.
- Keep ROOT-414 reference mass under `/home/mfspx/Documents/ROOT_414/` unless explicitly promoted.
- Keep KORPUS outputs under scoped output/runtime/storage directories with receipts.
- Prefer storage-only graph staging unless an authority gate explicitly materializes a graph change.
- New KORPUS workers must declare mutation authority and failure behavior.
