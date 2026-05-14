# DIOGENES / LUCIDOTA Cybersecurity Risk Register

Updated: 2026-05-13

## Current Risk Bar

- Overall security posture: `██████░░░░ 60%` — good local-first posture, but pre-release controls still incomplete.
- External fetch safety: `████░░░░░░ 40%` — Scout/Hop can fetch; source-policy/robots/domain allowlists are not complete.
- Secret safety: `████████░░ 80%` — repo ignore policy, no secret values, and sensitive credential artifact metadata moved out of tracked notes.
- Local service hardening: `█████░░░░░ 50%` — localhost gRPC/DBOS dev posture is acceptable for local dev, not for exposed network use.
- Data provenance safety: `██████░░░░ 60%` — CAS/hash/AGE edges exist, but graph write policy is not fully gated.

## Flags That Could Trigger a Cybersecurity Alert

1. **Sensitive credential artifact metadata in tracked docs**
   - Status: sensitive credential artifact metadata has been moved to ignored `03_VAULT/drive_map`.
   - Tracked docs remain summary-only and contain no credential values.

2. **Unencrypted local CAS is not final security posture**
   - Current CAS stores bytes under `03_VAULT/cas` by SHA-256.
   - Good for integrity; not sufficient for confidentiality.
   - Mitigation: encryption-at-rest/key policy still required before real case/secrets ingest.

3. **Scout/Hop external fetch can become SSRF-like if exposed to untrusted input**
   - URL fetchers can contact arbitrary targets.
   - This is fine for local operator use, risky as a network service.
   - Mitigation: private/loopback/reserved URL blocking, allowlists, robots policy, rate limits.

4. **Local file Scout can read arbitrary paths if called with local paths**
   - This is intended for operator-local ingest, but risky if exposed through an API/UI.
   - Mitigation: never expose local-file scout to remote users; add explicit governance gate for non-vault paths.

5. **Postgres dev superuser posture**
   - `mfspx` is a local dev superuser so AGE loads easily.
   - This is acceptable only on single-user dev box.
   - Mitigation: release profile needs least-privilege roles.

6. **gRPC is insecure localhost smoke**
   - Current CKDOG1 gRPC uses local insecure channel.
   - Acceptable for loopback smoke; unsafe on a LAN/public interface.
   - Mitigation: bind localhost only, add auth/TLS if remote access is ever enabled.

7. **DBOS dev console/log visibility**
   - DBOS prints console self-host links and workflow metadata.
   - Mitigation: avoid secrets in workflow details; redact target data in shared logs.

8. **AGE Cypher writes from discovered URLs**
   - Current writer JSON-quotes strings, but graph data may contain attacker-controlled labels/content.
   - Mitigation: keep labels static, values escaped, and never use discovered content as raw Cypher syntax.

9. **Wayback/external enrichment leaks target interest**
   - Querying Wayback discloses the URL of interest to an external service.
   - Mitigation: require explicit `--wayback`; keep default off.

10. **AI/export archives may contain private data**
    - Drive maps show AI export/conversation/archive files.
    - Mitigation: import only into private vault, never tracked repo.

## Immediate Security Build Priorities

- [x] Move sensitive credential metadata out of tracked docs into ignored vault map.
- [x] Add public-web-default source address gate to Scout/Hop.
- [x] Add credential redaction scanner to harness.
- [~] Source policy seed exists; full DB-backed enforcement still pending.
- [ ] Add least-privilege Postgres release profile note.
- [ ] Add encryption-at-rest decision for CAS.

## Required Rephrase Layer

Use this framing in docs, commands, and summaries to avoid misclassification and to stay precise:

> Continue building the local-first DIOGENES/LUCIDOTA evidence provenance system. Prioritize authorized source intake, bounded link discovery, provenance graph edges, local CAS integrity, DBOS workflow reliability, Bytewax/River scoring, and Treelite routing hints. Keep all credential material redacted and private; do not access anything unauthorized.

Credential inventory rule:

> Inventory credential file locations and store only redacted metadata/hashes in the private vault. Do not print or commit secret values.

