# Knowledge Card: Sydsec Syd

- ID: `sydsec_syd`
- Authority class: `candidate_tool`
- Source: https://gitlab.com/sydsec1/Syd
- Local clone: `01_REPOS/sydsec_syd/`
- Cloned commit: `39321b3ac4b5ad8c90c874ef449a1d8561f2b6f4`
- License: MIT (`01_REPOS/sydsec_syd/LICENSE`)

## What it is

Syd is an offline, local RAG assistant for analyzing security scan artifacts (Nmap XML, BloodHound JSON, Volatility output). Its strongest LUCIDOTA lesson is not the GUI or the large model download path; it is the **facts-first membrane**:

1. Deterministic parsers extract structured facts from source artifacts.
2. Retrieval/LLM layers are downstream explanation helpers, not source-of-truth creators.
3. Generated answers are validated against extracted facts and rejected when they mention unsupported entities.

## Learned pattern for LUCIDOTA

- Put a metadata/fact gate before slow inference. Route packets by structured facts, not raw vibes.
- Cache small fastlane facts with hashes, stage/lane metadata, and receipts; flush batches into slowlane analysis when thresholds are met.
- Keep local/offline defaults. Heavy model setup is optional and must be probed before depending on it.
- Separate UI shell from parser/core: Syd's `syd.py` is a useful warning that one giant GUI file accumulates bloat, while the fact extractors/analyzers are reusable boundaries.
- Validate outbound prose against known facts whenever a slowlane model gets involved.

## Repository map

- `syd.py` — tkinter GUI/orchestrator; large (~4.6k lines), useful as a bloat caution.
- `nmap_fact_extractor.py` — deterministic Nmap fact extraction.
- `bloodhound_fact_extractor.py` — deterministic BloodHound fact extraction and answer validation.
- `volatility_fact_extractor.py` — deterministic Volatility fact extraction and answer validation.
- `bloodhound_analyzer.py`, `volatility_analyzer.py`, `rag_engine/nmap_advice.py` — domain analyzers/advice.
- `chunk_and_embed_*.py`, `fix_all_faiss_indexes.py` — FAISS embedding/index maintenance scripts.

## LUCIDOTA integration stance

Keep Syd as a candidate-tool/reference pattern, not a runtime dependency. Adopt the facts-first/validate-outbound shape immediately in LUCIDOTA metadata gates. Do not import offensive tool execution paths or model download setup into the core runtime.

Immediate no-brainer committed here: `scripts/fast_slow_lane_gate.py` implements a metadata-only fastlane/slowlane gate with cache-to-slowlane flush receipts and no model/network/graph writes.

## Verification / blockers

Passed in this session:

- Clone exists at `01_REPOS/sydsec_syd`.
- `python3 -m py_compile` passed for `nmap_fact_extractor.py`, `bloodhound_fact_extractor.py`, `volatility_fact_extractor.py`, `bloodhound_analyzer.py`, and `volatility_analyzer.py`.
- Line-count audit identified the bloat seam: `syd.py` is ~4.6k lines, while parser/analyzer modules are clearer reusable units.

Not done:

- No dependencies installed.
- No model downloaded.
- No security tool execution performed.
- No GUI launch performed.
