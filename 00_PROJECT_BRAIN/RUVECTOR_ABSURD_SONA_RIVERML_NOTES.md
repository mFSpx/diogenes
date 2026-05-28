# Ruvector Concept Capture for LUCIDOTA ABSURD / RiverML / SONA

Date: 2026-05-20
Status: CAPTURED FOR LATER — NOT WIRED INTO LIVE SUITE

## Operator intent captured

RiverML/Treelite and SONA serve different layers and must not call each other directly.

- RiverML/Treelite is the **math and logic judge**: tabular streaming metrics, queue failure prediction, Chrono anomaly detection, claim-packet vector anomaly flags, and fast compiled verdicts.
- SONA/MicroLoRA is the **neural/embedding surgeon**: model adaptation, GLiNER/local-LLM extraction correction, feedback trajectories, and forgetting-safe neural repairs.
- ABSURD/Postgres is the durable workflow substrate: all work-order state, claim verdicts, dead letters, witness records, and promotion decisions land in Postgres-backed ABSURD tables.
- No live wiring performed in this pass. This file is the capture note for later implementation.

## Sources captured

- RuVector README: https://github.com/ruvnet/RuVector
- SONA README: https://github.com/ruvnet/RuVector/blob/main/crates/sona/README.md
- RuVector MinCut README: https://github.com/ruvnet/RuVector/blob/main/crates/ruvector-mincut/README.md
- RVF README: https://github.com/ruvnet/RuVector/blob/main/crates/rvf/README.md
- RVF crypto README: https://github.com/ruvnet/RuVector/blob/main/crates/rvf/rvf-crypto/README.md
- Residual VQ research gist: https://gist.github.com/ruvnet/cadf124e2e8220682452c268210b09a0

## 1. Residual Vector Quantization (RVQ) — use later for compressed claim-vector baselines

What to jack conceptually:

- Encode each full-dimensional embedding through sequential codebooks.
- Each codebook quantizes the residual error left by the previous codebook.
- Stage 0 captures coarse structure; later stages refine the miss.
- Search can use asymmetric distance computation (ADC): precompute query/codebook tables once, then score candidates with bounded code lookups.
- Keep exact rerank over a small candidate set when recall matters.

LUCIDOTA fit:

- Store claim-packet embedding fingerprints compactly for anomaly baselines.
- RiverML/Treelite can judge tabular features derived from RVQ residual error: reconstruction error, cluster distance, codebook entropy, rerank disagreement, and sudden residual drift.
- ABSURD dead-letter taxonomy can receive typed flags such as `ERR_VECTOR_CLUSTER_ANOMALY`, `ERR_RVQ_RECONSTRUCTION_DRIFT`, and `ERR_RERANK_DISAGREEMENT`.
- Do not use RVQ to mutate GLiNER or SONA. It is a judge-side compression/anomaly feature generator.

Later implementation note:

- Put Rust implementation candidates under `01_REPOS/lucidota_etl` or a new bounded Rust crate after reviewing licenses and API stability.
- Emit only small JSONL/WAL facts into `05_OUTPUTS/absurd/` and ABSURD Postgres tables.

## 2. Dynamic Minimum Cut — use later for workflow graph weak-link detection

What to jack conceptually:

- Maintain a dynamic graph of workers, queues, schemas, claim types, source bundles, and promotion gates.
- Update cuts incrementally as failures, retries, schema mismatches, or anomaly flags arrive.
- The cut identifies the smallest/weakest boundary whose failure disconnects correctness from output.

LUCIDOTA fit:

- Judge the ABSURD queue spine as a graph: queue node -> worker harness -> schema -> claim packet -> ledger -> promotion gate.
- Use min-cut to identify bottlenecks and failure isolation boundaries before running expensive repairs.
- For dead-letter healing, choose the minimum intervention: schema migration, parser fallback, LoRA correction, retry delay, or quarantine.
- Use this as a supervisory metric, not a resident service. It should run as an ephemeral worker over bounded edge batches.

Later implementation note:

- Input: Postgres query of ABSURD job/event/dead-letter graph plus claim lineage graph.
- Output: `absurd_min_cut_report` with cut edges, score, suggested intervention, evidence refs, and no graph mutation.

## 3. Cryptographic Witness Chain — use later for ABSURD job and claim provenance

What to jack conceptually:

- Every operation becomes a deterministic record whose hash links to the previous record.
- Segment/content identity is content-addressed.
- Optional signatures prove who/what produced the segment.
- Verification walks the chain and fails if any operation changed.

LUCIDOTA fit:

- ABSURD job transitions should form a witness chain: queued -> leased -> running -> succeeded/failed/dead_lettered.
- Claim packets should carry witness ancestry: source bundle hash -> parser span hash -> GLiNER extraction hash -> Treelite verdict hash -> dead-letter/repair hash -> promotion decision hash.
- This strengthens the current proof-hoard doctrine without paving the jungle.

Later implementation note:

- Start with SHA-256/SHAKE-compatible local hashes in Postgres; add Ed25519 only after key policy is explicit.
- Add a `prev_witness_hash` column or sidecar table rather than rewriting existing receipt blobs.
- Verification must be a bounded, ephemeral worker.

## 4. Git-like Copy-on-Write (COW) — use later for safe model/vector/claim branches

What to jack conceptually:

- Child branches share parent state and copy only changed segments.
- Branches are cheap enough for what-if edits and rollback.
- Parent/child lineage is explicit and cryptographically tied.

LUCIDOTA fit:

- Do not overwrite original claim packets, vector baselines, GLiNER outputs, or SONA adapter deltas.
- Use COW branch records for Dead-Letter Necromancer repairs: original candidate remains immutable; repaired candidate is a child branch with witness lineage.
- Use COW for SONA experiments: proposed MicroLoRA correction is a branch, evaluated by RiverML/Treelite, then promoted only if gates pass.

Later implementation note:

- Represent COW first in Postgres: `parent_artifact_hash`, `child_artifact_hash`, `delta_ref`, `membership_visibility`, `promotion_status`.
- Do not vendor RVF format or wire a runtime until a separate implementation order.

## 5. Straight-jacket strategy for LUCIDOTA

1. ABSURD worker extracts or receives a claim packet.
2. RiverML/Treelite judge runs over bounded numeric features: payload size, source trust, RVQ residual features, Chrono timing, schema deltas, worker history.
3. Judge emits a typed verdict into ABSURD/Postgres.
4. If the verdict is a neural extraction failure, Dead-Letter Necromancer creates a COW repair branch.
5. SONA/MicroLoRA receives only the approved neural correction payload, not raw queue state.
6. Witness chain records every step.
7. Promotion gate sees only verified claim candidates with evidence refs and witness continuity.

## Hard boundary

No direct RiverML/Treelite -> SONA call.
No direct SONA -> ABSURD queue mutation.
No direct neural output -> canonical graph.
Everything crosses via ABSURD/Postgres records, typed verdicts, evidence refs, and witness hashes.
