# Model Source Intelligence Map

## Status

Working note for later model/runtime sourcing. This is not an endorsement list and not a download manifest.

Use it to find leads for small, fast, local models, distills, quantizations, pruning work, adapter experiments, and runtime tricks. Every candidate still needs provenance, license, benchmark, safety, and operational fit review before it enters LUCIDOTA.

## Primary Hubs

### Reddit: r/LocalLLaMA

Use for:

- new local model drops.
- quantization reports.
- pruning/distillation discussions.
- inference/runtime troubleshooting.
- comparative user benchmarks.

Audit posture:

- good discovery source.
- verify claims against model cards, code, evals, and reproducible local tests.
- expect hype cycles and benchmark cherry-picking.

### Hugging Face Model Pages

Use for:

- model cards.
- license/provenance checks.
- weights and quant variants.
- community issue threads.
- maintainer notes.

Candidate creator/feed names mentioned by user:

- `failspy`
- `mlabonne`
- `chuanli11`
- `bartowski`

Audit posture:

- Hugging Face is the primary artifact registry, but a hosted model is not automatically legal, safe, useful, or well-documented.
- capture exact repo, revision/hash, license, base model, training data statement, quantization method, and runtime requirements.

### Discord Communities

User-mentioned targets:

- Nous Research.
- TheBloke / LocalLLaMA adjacent communities.
- SillyTavern / Chub.ai adjacent communities.

Use for:

- fast troubleshooting.
- early model chatter.
- synthetic-data and fine-tuning workflow notes.
- roleplay/coherence stress-testing reports for small models.

Audit posture:

- high signal possible, low archival stability.
- convert useful claims into dated notes with links/screenshots only when lawful and necessary.
- do not let Discord lore become architecture without reproduction.

### 4chan `/g/` Local Model General

Use for:

- early experimental leads.
- weird scripts.
- compression/derestriction chatter.
- anonymous field reports.

Audit posture:

- volatile, abrasive, and unverifiable by default.
- never trust a binary, script, or weight drop without sandboxing, hashes, provenance, and local inspection.
- treat as lead generation only.

## Candidate Intake Checklist

For every model, LoRA, quant, runtime patch, or training script:

- source URL.
- artifact hash.
- license.
- base model lineage.
- claimed training data.
- parameter count.
- quantization format.
- VRAM/RAM footprint.
- tok/s on local target hardware when tested.
- context length.
- tool/function-calling behavior.
- refusal/compliance behavior as observed, not advertised.
- benchmark claims.
- local smoke result.
- security review result.
- decision: reject, watch, test, adopt, or archive.

## LUCIDOTA Runtime Fit

Immediate target categories:

- ultra-small fast routers/classifiers.
- cheap intent and command parsers.
- small tool-calling models.
- medium local assistant model.
- larger distilled model with LoRA headroom.
- adapters aligned to ROOT-414 / source-cartridge activation.

The model stable is supposed to be replaceable. Do not hardwire product behavior to one vendor, one forum celebrity, or one model family.

## VIBESCONTROL Rule

Find aggressively. Trust slowly. Reproduce locally. Record why.
