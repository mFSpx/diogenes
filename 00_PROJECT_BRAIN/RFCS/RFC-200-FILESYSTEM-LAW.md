# RFC-200: Filesystem Organization / Archive / Promotion Law

Status: DRAFT  
Subject ID: `filesystem_law`  
Normative role: defines the filesystem as an authority map: active doctrine, runtime, receipts, vault, schemas, source, external labs, and archived references must live where their role is true.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

In LUCIDOTA, filesystem placement is not cosmetic. Placement communicates authority. If a board-game sim sits in the active repo, agents may treat it as production gravity. If ROOT-414 primitives sit beside active doctrine, agents may confuse historical reference with active ontology. If receipts are mixed into source, proof becomes clutter. If docs sprawl, active specs become unreadable and agents start inventing document roles.

The filesystem must therefore encode reality:

```text
active specs -> 00_PROJECT_BRAIN/ACTIVE_SPEC/ five scoped specs
RFC arguments -> 00_PROJECT_BRAIN/RFCS
schemas/contracts -> 06_SCHEMA
runtime state/logs -> 04_RUNTIME
receipts/outputs -> 05_OUTPUTS
vault/evidence/CAS -> 03_VAULT
reusable algorithms -> ALGOS
active scripts -> scripts
external board-game labs -> /home/mfspx/BOARD_GAMES
external archived ROOT-414 -> /home/mfspx/Documents/ROOT_414
```

## 2. Sources

### Local sources

- `AGENTS.md` — startup law: read Dev Library manifests and blueprint pseudolaw, search manifest, prefer reuse, protect sovereign toolbox originals, index the jungle rather than pave it.
- `00_PROJECT_BRAIN/RFCS/RFC-000-MASTER-THESIS-PROGRAM.md` — organization law listing active scoped specs, RFCs, registries, contracts, receipts, runtime state, external ROOT-414, and external Ahoy.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md` — identity, claim-state, proof-hoard, and anti-slop law.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md` — Dev Library is access layer, not authority layer; promotion path is copy/adapt/harden.
- `05_OUTPUTS/filesystem_organization/*` — receipts for ROOT-414 and Ahoy extraction.
- `scripts/oracle_scope_enforcer.py` — independent filesystem scope oracle for implementation slices.
- `scripts/lucidota_cas_journal.py` — durable local CAS ingest journal for dual-write recovery.
- `scripts/lucidota_markdown_ingest_archive.py` — legacy Markdown archive/promotion path with staged graph approval by default.

### External Source Anchors

- W3C PROV-O supports keeping generated artifacts, activities, agents, and derivations traceable; filesystem roles should preserve that traceability: <https://www.w3.org/TR/prov-o/>.
- NIST SP 800-86 supports collection/examination/analysis/reporting separation for digital forensic work; filesystem roles mirror that separation: <https://www.nist.gov/publications/guide-integrating-forensic-techniques-incident-response>.
- NIST FIPS 180-4 and Python `hashlib` anchor SHA-256/CAS identity practices used in vault/journal/custody paths: <https://csrc.nist.gov/pubs/fips/180-4/upd1/final> and <https://docs.python.org/3/library/hashlib.html>.
- Blueprint-first design supports explicit workflow artifacts and receipts over hidden prompt/path drift: <https://arxiv.org/abs/2508.02721>.

## 3. Authority Map

Filesystem classes:

1. **Active specs:** short, scoped, human-readable instructions in `00_PROJECT_BRAIN/ACTIVE_SPEC/`.
2. **RFCs:** long arguments, source-backed proofs, falsifiers, consequences.
3. **Registries:** machine-readable policy/status/ontology/model manifests.
4. **Schemas/contracts:** enforce storage/runtime shapes.
5. **Scripts/services:** active executable mechanisms.
6. **ALGOS:** reusable deterministic primitives.
7. **Vault:** source/custody/evidence/CAS/reference data.
8. **Runtime:** logs, PIDs, local state, volatile plans.
9. **Outputs:** receipts, reports, dashboards, generated artifacts.
10. **External labs:** valuable but not active production gravity.
11. **External archives:** preserved historical/reference material.

## 4. Promotion / Archive Law

Promotion path:

```text
sovereign original / lab / dev-library artifact
  -> candidate
  -> copied/adapted into production lane
  -> contract/test/receipt
  -> active runtime registration only if needed
```

Archive path:

```text
active tree artifact with wrong authority gravity
  -> classify
  -> move to vault/external archive/external lab
  -> leave pointer or registry update if needed
  -> receipt
  -> verifier boundary check
```

Never mutate sovereign originals unless the operator names the exact target. Do not production-gate the jungle. Promote hardened copies.

## 5. Whole-System Interaction

- **Dev Library:** indexes the jungle without granting authority.
- **RFC program:** explains why each major subsystem belongs where it belongs.
- **KORPUS/ETL:** vault and CAS paths preserve evidence; outputs record receipts.
- **Board-game lab:** external folder prevents lab gravity from becoming production truth.
- **Ontology:** ROOT-414 external archive prevents ontology confusion; GO remains active.
- **Automation:** logs/PIDs/output dashboards stay in runtime/output paths.
- **Artifact templates:** generated artifacts stay in outputs until promoted/exported.
- **Graph:** graph mutation uses schemas/gates/journals, not markdown placement.

## 6. Benefit to the Whole System

Filesystem law reduces LLM influence by making the repo self-explanatory. Agents are less likely to reinvent, misclassify, or over-promote when placement tells the truth. The operator can live-code aggressively because the project has clear lanes for experiments, runtime, evidence, receipts, doctrine, and external labs.

This is how the system remains a LEGO kit rather than a junk drawer: every brick can be weird, but its connector and authority are visible.

## 7. Correctness Argument

I believe this RFC is correct because the current repository now has both scoped specs and receipts matching the law. The five active spec files exist in `00_PROJECT_BRAIN/ACTIVE_SPEC/`. The RFC directory exists. ROOT-414 and Ahoy have been externalized with receipts. The Dev Library keeps proof-hoard indexing separate from production authority. The verifier detects boundary violations for ROOT-414/Ahoy active paths. Scope/oracle/CAS/archive scripts show filesystem role enforcement mechanisms already exist.

The correctness property is not “the filesystem is perfectly clean.” The property is stronger and more useful: every important artifact has or can be assigned an authority class and correct home, and boundary violations can be detected and corrected with receipts.

## 8. Falsifiers

This RFC is wrong if:

- active repo again contains ROOT-414 primitives or Ahoy lab trees,
- active specs grow into a huge unreadable doc pile,
- receipts are absent for major moves,
- agents cannot tell lab/reference/runtime/output/source roles apart,
- dev-library artifacts are treated as production truth merely because indexed,
- promotion happens without copy/adapt/harden/test/receipt.

## 9. Filesystem / Runtime Consequences

- Keep `00_PROJECT_BRAIN/RFCS/` as the long-form argument home.
- Keep exactly the five active spec files short, scoped, and authority-mapped; do not call any arbitrary file “the manual.”
- Keep `/home/mfspx/Documents/ROOT_414/` and `/home/mfspx/BOARD_GAMES/AHOY/` outside active repo.
- Maintain `rfc_program_check.py` boundary checks.
- Use receipts under `05_OUTPUTS/filesystem_organization/` for major moves.
- Use scope/oracle/CAS/archive tooling when changing filesystem authority.
