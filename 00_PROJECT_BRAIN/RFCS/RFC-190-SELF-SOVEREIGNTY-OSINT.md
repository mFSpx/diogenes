# RFC-190: Self-Sovereignty / OSINT Domain Argument

Status: DRAFT  
Subject ID: `self_sovereignty_osint`  
Normative role: defines LUCIDOTA's domain as lawful, local-first, evidence-centered self-sovereignty and OSINT, not surveillance theater, doxxing, coercion, or unbounded offensive intelligence.


## 0. Claim Discipline / ABBA3^5 Gate

ABBA3^5 is a local operator instruction, not an established external domain term. In this RFC it means: do not let confident prose outrun receipts. Every material CLAIM in this RFC must survive these five checks before it is treated as system guidance:

1. **Claim-state:** classify the statement as CLAIM, observation, inference, hypothesis, suggestion, or confidence-rated candidate; factual CLAIMs need receipts or cited sources.
2. **Provenance-count-and-reason:** state which operator instruction, repo file/code, receipt, or external source backs the claim; if only a few docs were consulted, name the count and why those docs were chosen.
3. **Naming-integrity:** preserve names as integral to their statement/context; label local/metaphorical names as local instead of pretending they are established terms.
4. **Reuse-before-reinvention:** search the LUCIDOTA Dev Library and known established concepts before proposing new code; reinvent only for sovereignty, objective superiority, necessity, or explicit operator intent.
5. **Operational-proportionality:** security, logging, tests, audits, and refusal gates are slop if they freeze work, flood storage, or waste operator time without proportional risk/truth benefit.

## 1. Thesis

LUCIDOTA is asymmetric warfare for self-sovereignty in the sense of capability asymmetry: one operator, constrained hardware, local data sovereignty, receipt-backed workflows, graph memory, OSINT discipline, and artifact production that can punch above its weight. The domain is not “hurt people with intelligence.” The domain is: preserve evidence, understand systems, map power, protect privacy, investigate lawfully, prepare reports, and keep the operator honest.

This is why the system must be FOSS/local-first, custody-first, graph-backed, and no-accusation by default. Self-sovereignty without evidence discipline becomes paranoia. OSINT without safety/privacy doctrine becomes harm. LUCIDOTA must be both capable and constrained.

## 2. Sources

### Local sources

- `00_PROJECT_BRAIN/ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md` — defines LUCIDOTA as a local sovereign exocortex for intelligence work, investigation, reporting, evidence handling, and self-checking; includes no-accusation doctrine.
- `00_PROJECT_BRAIN/ACTIVE_SPEC/05_COMPONENT_AUTHORITY_MAP.md` — maps local model fabric, KORPUS, Indy, language membrane, artifact templates, and investigation organs.
- `06_SCHEMA/018_investigation_artifact.sql` — local-first investigative case/artifact workflow with stable case folders, SHA-256 provenance, locked CAS bytes, sidecars, entity pivots, GO anchors, capability registry, artifact/case/tag tables.
- `00_PROJECT_BRAIN/INDY_READS_POLYCAREER_WORKFLOW_WIZARD/ARCHITECTURE.md` — safety boundary for lawful OSINT/research, evidence intake, fraud/case support, journalism-style verification, legal-clerk organization without legal advice, and explicit out-of-scope harms.
- Dev Library query evidence for `investigation`, `graph`, `archive`, and `filesystem` — relevant existing artifacts include investigation schema, graph promotion tools, CAS journal, and archive utilities.

### External Source Anchors

- The Berkeley Protocol identifies standards for digital open source investigations, including gathering, analyzing, preserving digital information professionally, legally, and ethically: <https://digitallibrary.un.org/record/3973652>.
- NIST SP 800-86 provides forensic process guidance around collection, examination, analysis, and reporting for incident response/troubleshooting contexts: <https://www.nist.gov/publications/guide-integrating-forensic-techniques-incident-response>.
- EFF Surveillance Self-Defense is a public guide to protecting against electronic surveillance; it supports the self-sovereignty/privacy side: <https://ssd.eff.org/about-surveillance-self-defense>.
- NIST Privacy Framework frames privacy risk management as protecting individuals while enabling products/services: <https://www.nist.gov/privacy-framework>.
- W3C PROV-O supports provenance chains for evidence and reports: <https://www.w3.org/TR/prov-o/>.

## 3. Domain Boundary

In scope:

- lawful OSINT and public-interest research,
- personal data sovereignty and local-first evidence storage,
- evidence intake, hashing, metadata extraction, sidecars, and case files,
- relationship/power/network mapping with source labels,
- contradiction handling and no-accusation reporting,
- privacy-preserving exports and redaction-aware artifacts,
- strategy and advocacy planning within lawful/ethical bounds,
- protective risk assessment and self-checking.

Out of scope:

- unauthorized access,
- credential theft,
- doxxing or harassment,
- covert manipulation or impersonation,
- tactical harm or evasion,
- illegal surveillance,
- publishing unverified claims as fact,
- pretending to provide legal counsel.

## 4. Whole-System Interaction

- **KORPUS/ETL:** preserves raw artifacts before interpretation.
- **GO graph:** models entities, claims, evidence, hypotheses, events, and relationships.
- **ABSURD/Diogenes:** controls workflows, commands, and authority boundaries.
- **Local LLM fabric:** drafts/extracts locally where possible, preserving data sovereignty.
- **Indy_READs:** routes polycareer workflows and produces briefs within safety boundaries.
- **Artifact templates:** generate reports/letters/case packets with source refs.
- **ABBA63:** keeps investigations abductive but non-credulous.
- **Filesystem law:** keeps sensitive vault/runtime/output material scoped and external labs separated.

## 5. Benefit to the Whole System

This domain argument explains why the whole project exists. The system is not a generic app; it is a local sovereign intelligence workstation. It benefits the operator by turning scarce resources into disciplined capability: ingest, remember, map, reason, report, learn, and protect.

It also disciplines the engineering. If a feature does not improve self-sovereignty, evidence quality, lawful OSINT, privacy, operator safety, or local capability, it is probably slop or a lab artifact.

## 6. Correctness Argument

I believe this RFC is correct because the doctrine, schemas, and architecture all converge on the same domain. `ACTIVE_SPEC/01_IDENTITY_AND_CLAIM_STATE_LAW.md` says local sovereign exocortex and no-accusation. `018_investigation_artifact.sql` implements cases, artifacts, SHA-256/CAS provenance, tags, and capability registry including OSINT and investigation. The Indy polycareer architecture names lawful OSINT, research, evidence organization, journalism-style verification, legal-clerk organization without legal advice, and explicit safety exclusions.

The correctness property is lawful evidence-centered empowerment. LUCIDOTA should increase the operator's ability to know, preserve, explain, and act responsibly — not increase harm or unlabeled certainty.

## 7. Falsifiers

This RFC is wrong if:

- core workflows require remote SaaS or uncontrolled third-party data processing,
- OSINT workflows lack source/custody/provenance records,
- reports publish unverified claims as fact,
- tooling enables doxxing/harassment/unauthorized access,
- privacy/redaction/export boundaries are absent,
- the system cannot produce evidence-backed artifacts from local cases.

## 8. Filesystem / Runtime Consequences

- Sensitive evidence belongs in vault/case/runtime/output paths, not doctrine docs.
- Exports require secret/privacy scans where applicable.
- Investigation schemas and receipts must preserve provenance.
- External scraping/enrichment must be source-policy gated.
- Generated reports should separate fact, source claim, hypothesis, and operator decision.
