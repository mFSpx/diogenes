# RFC Source Bibliography Seed

Status: ACTIVE SOURCE SEED  
Purpose: establish what kinds of evidence are allowed in LUCIDOTA RFCs.

## Local Canonical Sources

- `00_PROJECT_BRAIN/ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md`
- `00_PROJECT_BRAIN/ACTIVE_SPEC/02_EXECUTION_SPINE.md`
- `00_PROJECT_BRAIN/ACTIVE_SPEC/03_CUSTODY_ETL_PIPELINE.md`
- `00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md`
- `00_PROJECT_BRAIN/ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md`
- `OFFICIAL_ONTOLOGY.json`
- `BOOKS/GO_ACTIVE_TERMS.json`
- `00_PROJECT_BRAIN/spine_authority_registry.json`
- `00_PROJECT_BRAIN/canonical_graph_write_allowlist.json`
- `06_SCHEMA/035_absurd_queue_spine.sql`
- `06_SCHEMA/023_etl_pipeline.sql`
- `06_SCHEMA/016_go_graph_core.sql`
- `ALGOS/percyphon.py`
- `core/language_membrane.py`
- `scripts/absurd_queue_spine.py`
- `scripts/absurd_consume_one.py`
- `scripts/absurd_worker_contracts.py`
- `scripts/korpus_krampii.py`
- `scripts/indy_reads.py`
- `00_PROJECT_BRAIN/gpu_model_runtime_registry.json`
- `00_PROJECT_BRAIN/RUVECTOR_ABSURD_SONA_RIVERML_NOTES.md`

## External Sources

- RFC requirement language: IETF RFC 2119, <https://datatracker.ietf.org/doc/rfc2119/>.
- Provenance ontology: W3C PROV-O, <https://www.w3.org/TR/prov-o/>.
- Queue locking primitive: PostgreSQL `SELECT` docs for `FOR UPDATE` / `SKIP LOCKED`, <https://www.postgresql.org/docs/current/sql-select.html>.
- Evidence/provenance/claims model: SEPIO overview, <https://ohsu.elsevierpure.com/en/publications/sepio-a-semantic-model-for-the-integration-and-analysis-of-scient/>.
- Blueprint-first workflow source: arXiv `Blueprint First, Model Second`, <https://arxiv.org/abs/2508.02721>.
- PocketFlow simplicity mirror: local clone `01_REPOS/PocketFlow/` and upstream <https://github.com/The-Pocket/PocketFlow>.
- Local inference backend: llama.cpp upstream, <https://github.com/ggml-org/llama.cpp/blob/master/README.md>.
- GGUF / llama.cpp model format workflow: Hugging Face Hub docs, <https://huggingface.co/docs/hub/en/gguf-llamacpp>.
- Deterministic hashing: NIST FIPS 180-4 Secure Hash Standard, <https://csrc.nist.gov/pubs/fips/180-4/upd1/final>, and Python `hashlib`, <https://docs.python.org/3/library/hashlib.html>.
- Template syntax reference: Jinja documentation, <https://jinja.palletsprojects.com/>.
- Online learning: River upstream, <https://github.com/online-ml/river>, and River paper, <https://arxiv.org/abs/2012.04740>.
- Streaming dataflows: Bytewax dataflow programming docs, <https://docs.bytewax.io/v0.20.1/guide/concepts/dataflow-programming.html>.
- Lightweight zero-shot NER: GLiNER upstream, <https://github.com/urchade/GLiNER>, and GLiNER paper, <https://arxiv.org/abs/2311.08526>.
- Game simulation/search lab anchors: OpenSpiel paper, <https://arxiv.org/abs/1908.09453>; MCTS survey record, <https://cris.maastrichtuniversity.nl/en/publications/a-survey-of-monte-carlo-tree-search-methods/>; AlphaZero self-play paper, <https://arxiv.org/abs/1712.01815>.
- Automation timers: systemd timer manual, <https://www.freedesktop.org/software/systemd/man/latest/systemd.timer.html>.
- Ontology representation: W3C OWL overview, <https://www.w3.org/OWL>, and PostgreSQL constraints docs, <https://www.postgresql.org/docs/current/ddl-constraints.html>.
- Abductive reasoning and uncertainty: Stanford Encyclopedia of Philosophy on abduction, <https://plato.stanford.edu/archives/spr2024/entries/abduction/>, and Tversky/Kahneman via PubMed, <https://pubmed.ncbi.nlm.nih.gov/17835457/>.
- OSINT / self-sovereignty safety anchors: Berkeley Protocol, <https://digitallibrary.un.org/record/3973652>; NIST SP 800-86, <https://www.nist.gov/publications/guide-integrating-forensic-techniques-incident-response>; EFF Surveillance Self-Defense, <https://ssd.eff.org/about-surveillance-self-defense>; NIST Privacy Framework, <https://www.nist.gov/privacy-framework>.

## Evidence Ranking

1. Runnable command output / receipt / hash.
2. Current source code or schema.
3. Machine registry / JSON policy.
4. Local historical source material.
5. External standard / official documentation / paper.
6. Operator statement captured with date/context.
7. Inference from the above, labeled as inference.

An RFC MUST distinguish evidence from inference.
