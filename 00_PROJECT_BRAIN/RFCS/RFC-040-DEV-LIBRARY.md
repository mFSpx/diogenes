# RFC-040: Dev Library / Reusable Parts Without Authority Leakage

Status: DRAFT  
Subject ID: `dev_library`  
Normative role: defines how LUCIDOTA keeps a huge live-coding toolbox without mistaking the toolbox for production truth.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

The LUCIDOTA Dev Library is necessary because the operator live-codes while operating, and reuse must be faster than reinvention. But the Dev Library is an access layer, not an authority layer. Indexed artifacts are parts, references, experiments, labs, corpses, or candidates. They are not production truth merely because they are findable.

## 2. Why I Believe This Is True

Local evidence:

- `00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md` now defines the canonical human-facing concept.
- `00_PROJECT_BRAIN/TICKLETRUNK.json` is the current machine manifest of indexed proof-hoard parts.
- `scripts/dev_library_scan.py` wraps the legacy manifest scanner for new workflows.
- `scripts/tickletrunk_scan.py` remains the compatibility implementation and builds JSON/Markdown manifests plus `TOOLS/` access layers.
- `TOOLS/README.md` is part of the generated navigation/access layer.
- `AGENTS.md` now tells agents to use the Dev Library manifest and the preferred wrapper command.
- The ROOT-414 and Ahoy extraction proved why this distinction matters: interesting indexed artifacts had too much active-repo gravity until externalized.

## 3. External Source Anchors

- Blueprint First / Model Second supports explicit contracts over prompt fog; a dev library supports reuse without hiding workflow: <https://arxiv.org/abs/2508.02721>.
- RFC 2119 supports making Dev Library rules normative where needed: <https://datatracker.ietf.org/doc/rfc2119/>.
- W3C PROV-O supports preserving origin and lineage; copied/adapted production parts must not erase where they came from: <https://www.w3.org/TR/prov-o/>.
- The local PocketFlow clone and upstream repository are the simplicity mirror for small explicit workflow parts: <https://github.com/The-Pocket/PocketFlow>.

## 4. Classification Law

Every library artifact SHOULD be classifiable as one of:

- `sovereign_original`
- `reference_material`
- `sandbox_experiment`
- `reusable_prior`
- `active_runtime`
- `legacy_compatibility`
- `paused_lab`
- `corpse_preserved_for_hashes`
- `external_repo`
- `generated_receipt`

Unclassified artifacts may exist during discovery. They MUST NOT gain production authority by accident.

## 5. Promotion Path

```text
indexed artifact
  -> candidate for reuse
  -> copied/adapted into production lane
  -> component contract
  -> tests/checks/receipts
  -> active runtime registration only if needed
```

The original remains preserved unless the operator explicitly names it for mutation.

## 6. Whole-System Interaction

- **Slop Law:** Dev Library search prevents reinvention.
- **Main Spine:** promotion requires authority and mutation class.
- **ETL/Krampus:** old artifacts can be preserved/indexed without active runtime status.
- **Runtime Organs:** named organs are backed by library evidence when weird names appear.
- **RFC Program:** subject RFCs cite library artifacts as evidence but distinguish source from inference.

## 7. Benefit to the Whole System

The Dev Library lets LUCIDOTA stay creative and fast without becoming a script swamp. It gives the operator a parts bin for live coding and repair. It also lets archived strangeness remain findable without forcing the active system to be conventional or sterile.

## 8. Correctness Argument

The Dev Library model is correct because LUCIDOTA has two competing needs:

1. preserve the jungle,
2. keep active execution predictable.

A single production tree cannot satisfy both. Therefore the system needs a findable library plus promotion boundaries. The manifest can be large and weird because its role is discovery, while production code can remain small and guarded because it consumes only copied/adapted/hardened parts. That split is what prevents the operator's hard-won historical work from being deleted while also preventing old experiments from pretending to be current runtime.

The current rename posture is also correct: the old implementation name remains compatibility until a mechanical, receipt-backed rename can happen. The serious human-facing name is Dev Library, but stability matters more than performative renaming.

## 9. Operational Rule

The operational rule for agents is: search first, then write. A search result is not an order to use the artifact; it is an obligation to consider whether the artifact already solves part of the problem. If it does, copy/adapt the smallest useful part, add tests/receipts, and leave the original alone. If it does not, the RFC or receipt should say why reuse was rejected. This converts the Dev Library from a passive index into an anti-reinvention gate.

The rule also protects creativity. Experimental, disconnected, or strange artifacts can stay indexed because indexing does not impose production standards. The production copy receives the standard. That is how LUCIDOTA keeps both a wild proof hoard and a predictable execution spine.

## 10. Falsifiers

This RFC is wrong if:

- library search is slower than writing from scratch,
- classification overhead prevents live operation,
- production code cannot be separated from sovereign originals,
- the scanner produces enough false gravity that it causes more confusion than reuse.

Current corrective action is the human-facing wrapper and boundary verifier, not deletion of the library.

## 11. Filesystem / Runtime Consequences

- Use `scripts/dev_library_scan.py` in new operator-facing instructions.
- Keep legacy scanner filenames until a mechanical rename is safe.
- Do not add new doctrine with the old vibe-name except as compatibility reference.
- Keep `TOOLS/` as generated navigation, not production authority.
- Promotion receipts should cite original library path and copied/adapted destination.
