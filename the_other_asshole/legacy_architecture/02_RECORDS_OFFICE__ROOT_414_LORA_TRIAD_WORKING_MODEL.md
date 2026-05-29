# ROOT-414 LoRA Triad Working Model

## Status

This is a working concept note, not the canonical ROOT-414 table.

The actual 414 primitives remain creator-canonical and immutable at public-release base-functionality level. Until northern.strike provides the real primitive list, examples in this file are scaffolding for intent and architecture only.

## Concept

Each ROOT-414 anchor can eventually map to a hot-swappable knowledge/training cartridge:

- one anchor.
- one LoRA or equivalent adapter target.
- three supporting source streams per anchor.
- 414 anchors times 3 sources = 1,242 source slots.

The triad pattern is useful because it forces each anchor to carry more than a label:

- theory: formal or explanatory grounding.
- practice: operational application.
- shadow data: adversarial, messy, or field-derived material.

This matches the LUCIDOTA pattern: source material is ingested, hashed, embedded, graph-linked, validated, and made available as cartridges without pretending Markdown manifests are executable software.

## Architecture Implication

ROOT-414 cartridge records should eventually include:

- canonical root id.
- canonical primitive label.
- creator-defined meaning.
- triad source slots.
- ingest provenance.
- license/usage status.
- embedding set ids.
- graph ids.
- LoRA/adapter ids where applicable.
- activation policy.
- audit notes.

Activation is not just "load a model." It is coordinated focus:

- retrieve the relevant source cartridge.
- expose graph neighborhood and claims.
- load or select adapters if available.
- bind current workflow/tools to the active ontology focus.
- record what was activated and why.

## Noncanonical Example Shape

The user-provided Apex 10 examples demonstrate the desired density and flavor:

- anchors have memorable adapter names.
- each anchor carries theory, practice, and shadow-data slots.
- sources may be books, manuals, datasets, archives, or other symbols.
- not every source is automatically lawful, obtainable, useful, or shippable.

VIBESCONTROL rule: accept the structure, audit the details.

## Anti-Slop Requirements

- Do not fabricate the remaining 404 anchors.
- Do not treat guessed labels as canonical.
- Do not download copyrighted books or datasets without a lawful path.
- Do not train or distribute adapters from material without tracking provenance and rights.
- Do not let this become a giant static bibliography disconnected from ingest, graph, retrieval, and runtime activation.

## Next Work

1. Define the cartridge schema.
2. Add a placeholder ROOT-414 table with empty canonical slots.
3. Wait for creator-canonical primitive input.
4. Build importer paths that can attach source triads to anchors without hardcoding the examples.
