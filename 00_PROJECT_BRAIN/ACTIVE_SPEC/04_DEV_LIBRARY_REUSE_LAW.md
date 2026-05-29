# LUCIDOTA Dev Library Reuse Law

Status: ACTIVE SPEC SOURCE — reuse before reinvention and toolbox sovereignty  
Purpose: replace the old vibe-name concept with a serious rule: the operator has a live dev library for reuse while operating.

## 1. Name

Canonical human name: **LUCIDOTA Dev Library**.

Legacy implementation names may still appear in scripts, generated files, and old docs. The old manifest/scanner name is compatibility only until a safe rename pass happens.

## 2. What It Is

The Dev Library is the indexed workbench of reusable parts:

- scripts,
- tools,
- algorithms,
- schemas,
- workflows,
- models,
- LoRAs,
- scrapers,
- skills,
- plugins,
- services,
- surfaces,
- books/reference assets,
- runtime fixtures,
- strange useful-later experiments.

It exists because the operator live-codes while operating. Reuse must be fast enough to beat reinvention.

## 3. What It Is Not

The Dev Library is not production truth. A file being indexed does not mean it is active, safe, current, complete, canonical, or allowed to mutate anything.

The Dev Library is an access layer, not an authority layer.

## 4. Reuse Law

Before writing a new tool, script, workflow, schema, model hook, scraper, plugin, or service:

1. search the Dev Library,
2. identify existing candidates,
3. copy/adapt/reuse if useful,
4. leave sovereign originals intact unless the operator names the exact target,
5. harden the production copy through contracts/tests/receipts.

## 5. Promotion Path

```text
library artifact
  -> candidate for reuse
  -> copied/adapted into production lane
  -> component contract
  -> tests/checks/receipts
  -> active runtime registration only if needed
```

Do not production-gate the whole jungle. Promote only the hardened copy.

## 6. Classification

Every Dev Library item should be classifiable as one of:

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

Misclassification is allowed during discovery; unlabeled authority is not.

## 7. Current Implementation Sources

Current compatibility sources:

- `00_PROJECT_BRAIN/TICKLETRUNK.json`
- `00_PROJECT_BRAIN/TICKLETRUNK.md`
- `scripts/dev_library_scan.py` — preferred human-facing CLI
- `scripts/tickletrunk_scan.py` — legacy implementation CLI
- `TOOLS/`
- `05_OUTPUTS/tickletrunk/`

Use `python3 scripts/dev_library_scan.py --query <topic>` in new operator-facing workflows. Any deeper rename of legacy filenames should be mechanical, receipt-backed, and separate from doctrine consolidation.
