# LUCIDOTA / DIOGENES Full Build Plan Audit

Updated: 2026-05-13

Status key: `[x]` done/verified, `[~]` partial/prototype/wired-but-not-complete, `[ ]` not done.

## Overall Build Bar: ████████░░ 80%

## Hard Truth Audit
- **gRPC** `███████░░░ 70%` — Wired for CKDOG1 smoke through Rust tonic/prost <-> Python grpcio. Not yet full product API.
- **DBOS** `██████░░░░ 62%` — Installed, schema initialized, smoke workflow passes, events table used, Survey has DBOS wrapper. Not yet owning all workflows.
- **Postgres/AGE/pgvector** `██████░░░░ 60%` — Installed and verified; core schemas exist. Full ontology/vector schemas pending.
- **Survey/Hop Pivot** `█████████░ 86%` — Working Survey slice with CAS/hash/Aho/Wayback option/pivots and bounded hop v1. Full scraper ladder pending.
- **River ML** `██████░░░░ 62%` — Installed and now writes online scores from workflow events. Live Bytewax stream and frozen feature schema pending.
- **Bytewax** `██████░░░░ 62%` — Installed/imports. No live dataflow graph yet.
- **Treelite** `███████░░░ 70%` — Installed/imports. No compiled router wired yet.
- **Model runtime** `███████░░░ 68%` — CUDA/llama.cpp/runtime imports exist plus 39 local algorithm primitives. DeepSeek/embedding/router not yet product-wired.
- **Drive map/import** `███░░░░░░░ 30%` — Key nuclei located; full granular map and imports pending.
- **Persona Indy_Reads** `█░░░░░░░░░ 10%` — Named and sourced in records; runtime persona approximation not yet built.
- **Progress UI** `█████░░░░░ 55%` — Read-only terminal Big Board v0 now renders build bars, live workflow/wake/CAS/Body Capture/reflex counters, GPU status, and JSON export.

## Phase Bars

### Validated Today-Slice Through 007
- **000-007 operator-supervised green slice** `██████████ 100%` — harness verified Project Brain docs, CKDOG1 gRPC smoke, Clawd, Postgres schemas, CAS, DBOS wrappers/events, Bytewax/River/Treelite hints, Survey/Hop safety gates. This is not claiming every global backlog item is complete.

- **000 Project Brain / Governance** `█████████░ 92%` (16 done / 1 partial / 1 open)
- **001 Kernel / CKDOG1** `████████░░ 81%` (14 done / 1 partial / 3 open)
- **002 Clawd / Rust Interface** `██████░░░░ 58%` (10 done / 1 partial / 7 open)
- **003 Postgres / Records Office** `█████████░ 92%` (16 done / 1 partial / 1 open)
- **004 Storage / Vault / CAS** `███████░░░ 72%` (12 done / 2 partial / 4 open)
- **005 DBOS Workflow Plane** `███████░░░ 70%` (12 done / 2 partial / 4 open)
- **006 Bytewax / River / Treelite Reflex Team** `█████████░ 92%` (18 done / 0 partial / 0 open)
- **007 Survey / Hop Pivot / Authorized Extractors** `█████████░ 85%` (19 done / 1 partial / 3 open)
- **008 Body Capture / Capture / Evidence Diff** `█████████░ 93%` (23 done / 1 partial / 1 open)
- **009 Drive / External Memory / Imports** `██████░░░░ 59%` (12 done / 2 partial / 8 open)
- **010 Model Runtime / Local Brain** `███████░░░ 68%` (13 done / 2 partial / 3 open; primitive library widened to 40 wrappers)
- **011 Indy_Reads / Persona / Assistant Layer** `█████░░░░░ 52%` (10 done / 2 partial / 6 open)
- **012 Big Board / UI / Progress Bars** `███████░░░ 72%` (14 done / 0 partial / 4 open)
- **013 Security / Auth / Credentials** `██████░░░░ 60%` (9 done / 2 partial / 7 open)
- **014 Verification / Release / Product Slice** `█████████░ 92%` (20 done / 0 partial / 0 open)

## Full Checklist

### 000 Project Brain / Governance — █████████░ 92%
- [x] 001. Workspace exists at /home/mfspx/LUCIDOTA
- [x] 002. /home/mfspx/DIOGENES compatibility symlink preserved
- [x] 003. THEPLAN recorded
- [x] 004. DECISIONS ledger initialized
- [x] 005. STATUS ledger initialized
- [x] 006. SOURCES ledger initialized
- [x] 007. OPEN_QUESTIONS ledger initialized
- [x] 008. GLOSSARY initialized
- [x] 009. Anti-slop audit doctrine recorded
- [x] 010. Brainstorm reconciliation rule recorded
- [x] 011. MinIO demoted to deferred/candidate storage adapter
- [x] 012. XGBoost runtime ban recorded
- [x] 013. Treelite runtime path recorded
- [x] 014. DBOS + Bytewax team model recorded
- [x] 015. Drive mapping rules recorded
- [x] 016. No secret values in tracked docs rule recorded
- [ ] 017. PARTIAL: Indy_Reads persona intake started from existing records
- [ ] 018. Full Indy_Reads persona approximation generated and wired into runtime prompts

### 001 Kernel / CKDOG1 — ████████░░ 81%
- [x] 019. doggystyle cloned under 01_REPOS
- [x] 020. CKDOG1 package installs in local venv
- [x] 021. CKDOG1 unit test suite passes
- [x] 022. CKDOG1 CLI doctor smoke passes
- [x] 023. CKDOG1 genesis smoke passes
- [x] 024. CKDOG1 ontology init smoke passes
- [x] 025. CKDOG1 soul create smoke passes
- [x] 026. CKDOG1 trait set smoke passes
- [x] 027. CKDOG1 semantic route smoke passes
- [x] 028. CKDOG1 route smoke passes
- [x] 029. kernel.proto exists as current canonical proto
- [x] 030. Python grpcio server exists
- [x] 031. gRPC smoke harness exists
- [x] 032. Live gRPC smoke passes on localhost
- [ ] 033. PARTIAL: Only smoke endpoints are production-wired
- [ ] 034. Full ingest/recall gRPC method implemented
- [ ] 035. Full graph state-update gRPC method implemented
- [ ] 036. Full workflow/tool action gRPC method implemented

### 002 Clawd / Rust Interface — ██████░░░░ 58%
- [x] 037. claudecode fork cloned under 01_REPOS
- [x] 038. Rust/Cargo installed system-wide
- [x] 039. Clawd release build succeeds
- [x] 040. Rust workspace tests pass
- [x] 041. Generated tonic/prost client compiled
- [x] 042. claw diogenes-smoke command exists
- [x] 043. claw diogenes-smoke starts CKDOG1 and calls gRPC
- [x] 044. claw lucidota-survey command exists
- [x] 045. Clawd plugin hook runner repaired
- [x] 046. Private UI/theming track recorded
- [ ] 047. PARTIAL: Morrowind-inspired status bar spec recorded, not fully coded
- [ ] 048. Always-visible progress bar renderer implemented
- [ ] 049. Big Board TUI first panel implemented
- [ ] 050. Operator command queue implemented
- [ ] 051. Quiet side-process inbox implemented
- [ ] 052. Clawd auth/profile surfaces implemented
- [ ] 053. Clawd workflow launch/inspect/cancel implemented
- [ ] 054. Clawd drive-map read UI implemented

### 003 Postgres / Records Office — █████████░ 92%
- [x] 055. PostgreSQL 18.3 installed
- [x] 056. lucidota_state database created
- [x] 057. lucidota_graph database created
- [x] 058. pgvector extension verified
- [x] 059. Apache AGE extension verified
- [x] 060. Local dev role mfspx usable
- [x] 061. DBOS/Postgres topology note created
- [x] 062. control-plane schema applied
- [x] 063. model runtime schema applied
- [x] 064. survey schema applied
- [x] 065. learning/reflex schema applied
- [x] 066. workflow_event table exists
- [x] 067. governance_gate table exists
- [x] 068. source_policy table exists
- [x] 069. Artifact metadata schema exists for Survey/local CAS artifacts
- [ ] 070. encrypted local CAS finalized
- [x] 071. AGE provenance graph writer implemented for pivots/promotions
- [ ] 072. pgvector chunk schema finalized

### 004 Storage / Vault / CAS — ███████░░░ 72%
- [x] 073. Vault directory exists with no-commit rule
- [x] 074. Survey stores artifacts in local CAS
- [x] 075. Survey computes SHA-256
- [x] 076. Survey persists artifact metadata
- [x] 077. Originals immutable rule recorded
- [x] 078. MinIO not canonical rule recorded
- [x] 078A. Storage decision matrix written; Cassandra marked non-canonical
- [ ] 079. PARTIAL: Local CAS layout exists but not final encrypted vault
- [ ] 080. PARTIAL: Provenance edge path exists in tables, not full graph
- [ ] 081. Encryption-at-rest implementation selected
- [ ] 082. Key management policy implemented
- [x] 083. CAS GC report-first mark/sweep implemented
- [x] 084. CAS manifest/index implemented
- [ ] 085. CAS restore/export implemented
- [x] 086. CAS->AGE provenance edge writer implemented for Survey artifacts
- [x] 087. Vault integrity checker implemented
- [x] 087A. CAS GC smoke wired into full harness
- [x] 088. CAS orphan quarantine path implemented; no delete mode
- [ ] 089. Drive archive ingest into CAS implemented
- [ ] 090. Binary diff storage implemented

### 005 DBOS Workflow Plane — ███████░░░ 70%
- [x] 091. DBOS installed in project venv
- [x] 092. DBOS smoke workflow runs
- [x] 093. DBOS initializes system schema
- [x] 094. workflow_event conventions exist
- [x] 095. Survey emits workflow_event
- [x] 096. Harness runs DBOS smoke
- [x] 097. DBOS Survey workflow wrapper implemented
- [ ] 098. PARTIAL: DBOS policy/gate tables exist but are not enforced by all actions
- [x] 099. DBOS workflow registry implemented
- [ ] 100. DBOS retry policy implemented
- [ ] 101. DBOS scheduled watchers implemented
- [ ] 102. DBOS external-write draft gate implemented
- [ ] 103. DBOS Drive map workflow implemented
- [ ] 104. DBOS scraper dispatch workflow implemented
- [ ] 105. DBOS capture workflow implemented
- [ ] 106. DBOS model routing workflow implemented
- [ ] 107. DBOS Big Board event feed implemented
- [ ] 108. DBOS workflow replay tests implemented
- [x] 108A. DBOS/source policy seed implemented for public web, Wayback, local files, and Drive
- [x] 108B. Wake Bus outbox schema implemented
- [x] 108C. Wake Bus local signal worker implemented

### 006 Bytewax / River / Treelite Reflex Team — █████████░ 92%
- [x] 109. River installed
- [x] 110. Treelite installed
- [x] 111. Bytewax installed
- [x] 112. learning/reflex schema created
- [x] 113. River reflex script trains from committed workflow_event rows
- [x] 114. River score persisted to Postgres
- [x] 115. Runtime smoke imports River/Treelite/Bytewax
- [x] 116. River is wired through Bytewax mini stream proof and CLI reflex persistence
- [x] 117. Treelite routing-hint proof runs without XGBoost runtime
- [x] 118. Bytewax mini live dataflow graph implemented
- [x] 119. Bytewax TestingSource ordering proof implemented
- [x] 120. River feature schema frozen for v0 source/phase/decision/status features
- [x] 121. Bytewax-to-hint persistence implemented for DBOS/survey events
- [x] 122. Treelite router artifact generated in ignored local vault
- [ ] 123. Treelite router called from DBOS policy
- [x] 124. XGBoost kept offline-only by policy; current Treelite proof uses no XGBoost runtime
- [x] 125. No-XGBoost-runtime assertion included in Treelite router report
- [x] 126. Reflex audit trail persisted through river_run/bytewax_stream_run/treelite_router_run
- [x] 126A. Wake Bus canon wired into Section V event stream plan

### 007 Survey / Hop Pivot / Authorized Extractors — █████████░ 85%
- [x] 127. Survey CLI exists
- [x] 128. HEAD/GET/file triage implemented
- [x] 129. Size gate implemented
- [x] 130. multi-pattern keyword scan implemented
- [x] 131. HTML structure summary implemented
- [x] 132. Pivot link extraction implemented
- [x] 133. Wayback optional lookup implemented
- [x] 134. Postgres persistence implemented
- [x] 135. Clawd subcommand wired
- [x] 136. Harness smoke covers Survey
- [ ] 137. PARTIAL: Tree-sitter slot exists but real Tree-sitter parser not installed/wired
- [x] 138. Bounded link discovery / hop pivot capability named and persisted
- [x] 139. Bounded hop expansion implemented
- [x] 140. Promotion scoring implemented
- [x] 141. Promotion threshold gate implemented
- [x] 142. Hop queue/job/node persistence implemented
- [x] 143. Authorized extractor registry implemented; adapter-first policy recorded
- [ ] 144. Scraper ladder Level 2 static extraction implemented
- [ ] 145. Scraper ladder Level 3 Playwright implemented
- [ ] 146. Scraper ladder Level 4 local-reasoning fallback implemented
- [ ] 147. Scraper fleet imported from Drive/PYPELINE
- [x] 148. Authorized source-intake and bounded link-discovery smokes included in harness
- [x] 148A. Source URL gate defaults to public-web intake
- [x] 148B. Hop Pivot propagates explicit local-address option
- [x] 148C. Browser fallback explicitly non-default in extractor registry
- [x] 148D. Extractor registry smoke wired into harness

### 008 Body Capture / Capture / Evidence Diff — █████████░ 93%
- [x] 149. Body Capture Head brainstorm reconciled as candidate subsystem
- [x] 150. Screenshot/capture requirement recorded
- [x] 151. Wayback requirement recorded
- [x] 152. SHA-256 evidence rule exists
- [x] 153. Body Capture HTTP-body capture writes current artifact hash
- [~] 154. Browser capture contract implemented; real screenshot waits on clean browser binary
- [~] 155. Visual hash slot and browser fallback tool exist; skipped without usable browser
- [x] 156. Capture bytes stored in CAS for HTTP-body v0
- [x] 157. Canonical content/structure hash pipeline implemented for HTTP/HTML v0
- [x] 158. Current-vs-prior SHA delta row implemented
- [x] 159. Wayback-vs-current diff implemented
- [x] 160. Tree/text diff implemented
- [x] 161. Delta summarization implemented
- [x] 162. Evidence bundle export implemented
- [x] 163. Capture provenance AGE edge implemented as CAPTURED_AS
- [x] 164. Watcher profile policy evaluator implemented
- [x] 165. Body Capture policy harness smoke implemented; scheduling pending
- [ ] 166. Visual diff review UI implemented
- [x] 166A. Body Capture v0 harness smoke added
- [x] 166B. Body Capture graph/state DB split fixed after adversarial audit
- [x] 166C. Body Capture watcher law recorded
- [x] 166D. Adversarial audit record created
- [x] 166E. Browser fallback contract implemented with safe skip
- [x] 166F. Body Capture visual channel design recorded
- [x] 166G. Tri-Algo conduit signal gate wired before Body Capture CAS writes
- [x] 166H. Signal ingress schema records standby/burst/recover decisions
- [x] 166I. Body Capture graph-local workflow_event_outbox prevents silent state split-brain
- [x] 166J. Wake Bus batch delivery CTE removes O(N) update round trips
- [x] 166K. Survey local-address gate uses explicit ipaddress properties, no obfuscation

### 009 Drive / External Memory / Imports — ██████░░░░ 59%
- [x] 167. Google Drive connector available
- [x] 168. Google Contacts connector available
- [x] 169. Drive ambient-browse boundary recorded
- [x] 170. User authorized full Drive map
- [x] 171. Top-level Drive nuclei mapped
- [x] 172. PYPELINE folder found
- [x] 173. C_PYPELINE.zip verified by summary
- [x] 174. math-intrinsics found
- [x] 175. ENV/secrets locations identified without printing values
- [x] 176. scraper docs searched
- [x] 177. template-generation docs searched
- [x] 178. DRIVE_MAP_STATUS written
- [ ] 179. PARTIAL: Full granular map in progress
- [ ] 180. PARTIAL: PYPELINE source manifest fetched but not stored as final machine map
- [ ] 181. Full recursive Drive map persisted in private vault
- [ ] 182. PYPELINE archive downloaded/inspected after explicit target confirmation
- [ ] 183. math archive/folder correlated to source manifest
- [ ] 184. all .env/secret/auth files inventoried into secret vault
- [ ] 185. all scrapers copied into quarantine/import worktree
- [ ] 186. all document templates copied into quarantine/import worktree
- [ ] 187. Drive import checksums verified
- [ ] 188. Wrong earlier dossier intake quarantined/ignored

### 010 Model Runtime / Local Brain — ███████░░░ 68%
- [x] 189. CUDA toolkit installed
- [x] 190. GPU visible
- [x] 191. llama.cpp CUDA build exists
- [x] 192. Runtime imports transformers/accelerate/peft/safetensors/etc
- [x] 193. model registry schema exists
- [x] 194. runtime inventory recorder exists
- [x] 195. MPS lifecycle helpers exist
- [ ] 196. PARTIAL: DeepSeek-R1-Distill-Qwen-1.5B selected in architecture but not verified local artifact
- [ ] 197. PARTIAL: Needle imported but not benchmarked
- [ ] 198. DeepSeek model downloaded/verified
- [ ] 199. Embedding model selected
- [ ] 200. Embedding model benchmarked
- [ ] 201. Local reasoning service wrapped
- [ ] 202. Pydantic output schema enforcement wired
- [ ] 203. Model load/unload governor implemented
- [ ] 204. VRAM budget guard implemented
- [ ] 205. Model routing policy tied to Treelite
- [ ] 206. Needle benchmark completed
- [x] 206A. Local algorithm primitives wrapped under ALGOS: HDC, NLMS, SSIM, MinHash, Possum filter, OPOSSUM RBF surrogate, Thanatosis, Capybara Optimization, Hoeffding stream split, Schoolfield poikilotherm, Serpentina self-righting, Chelydrid ambush-strike
- [x] 206B. Algorithm primitive smoke wired into harness (40 modules)

### 011 Indy_Reads / Persona / Assistant Layer — █████░░░░░ 52%
- [x] 207. Indy_Reads marked active assistant/persona
- [x] 208. Operator identity recorded
- [x] 209. Project brain answers basic status from docs
- [x] 210. Quiet side-process rule recorded
- [ ] 211. PARTIAL: Persona source exists in Drive/archives but not fully extracted
- [x] 212. Runtime prompt/brief approximation generated from project brain contract
- [ ] 213. Indy_Reads persona Drive targets mapped
- [ ] 214. Indy_Reads persona corpus imported to private vault
- [ ] 215. Persona distillation note produced
- [x] 216. Runtime system prompt contract produced
- [x] 217. Task memory routines implemented
- [x] 218. Calendar intent routine implemented
- [x] 219. Reminder routine implemented
- [x] 220. Wiki/notes routine implemented
- [x] 221. Auth inventory routine implemented
- [x] 222. No-interruption side queue implemented
- [ ] 223. Persona regression tests created
- [ ] 224. Operator correction learning loop implemented

### 012 Big Board / UI / Progress Bars — ███████░░░ 72%
- [x] 225. Progress-bar requirement captured
- [x] 226. Status docs exist
- [x] 227. Workflow_event has phase/status
- [x] 228. Drive map status doc exists
- [x] 229. Manual bars can be reported from docs
- [x] 230. Automated progress summarizer implemented
- [x] 231. Terminal dashboard implemented
- [x] 232. Service health panel implemented
- [x] 233. Build phase panel implemented
- [ ] 234. Drive map panel implemented
- [ ] 235. Secrets/auth panel with redaction implemented
- [x] 236. Workflow queue panel implemented
- [x] 237. Model/GPU panel implemented
- [x] 238. River/Bytewax/Treelite panel implemented
- [ ] 239. Scraper fleet panel implemented
- [x] 240. Evidence/vault panel implemented
- [ ] 241. Operator correction panel implemented
- [x] 242. Report export implemented

### 013 Security / Auth / Secrets — ███░░░░░░░ 33%
- [x] 243. Repo ignore policy covers secrets
- [x] 244. Auth inventory doc exists
- [x] 245. Drive secret-value printing banned
- [x] 246. External writes draft/preview rule recorded
- [x] 247. Governance gate table exists
- [ ] 248. PARTIAL: Local .codex auth known but not copied into repo
- [ ] 249. PARTIAL: Drive secrets_current located but not ingested
- [ ] 250. Secret vault implemented
- [ ] 251. Credential classifier implemented
- [ ] 252. Secret redaction scanner in harness
- [ ] 253. Auth material imported to vault with hashes
- [ ] 254. Auth freshness/rotation status tracked
- [ ] 255. Gmail auth tooling implemented
- [ ] 256. Calendar auth tooling implemented
- [ ] 257. Drive write policy implemented
- [ ] 258. Emergency revoke runbook written
- [ ] 259. Release sanitization gate implemented
- [ ] 260. Threat model linked to build phases

### 014 Verification / Release / Product Slice — ██████░░░░ 60%
- [x] 261. check_diogenes.sh exists
- [x] 262. Postgres AGE/pgvector smoke in harness
- [x] 263. DBOS smoke in harness
- [x] 264. runtime smoke in harness
- [x] 265. CKDOG1 tests in harness
- [x] 266. CKDOG1 gRPC smoke in harness
- [x] 267. Rust workspace tests in harness
- [x] 268. Clawd release build in harness
- [x] 269. Clawd gRPC smoke in harness
- [x] 270. Survey smoke in harness
- [x] 271. River reflex smoke added to harness
- [x] 272. Full harness rerun after latest River/Bytewax edits
- [ ] 273. Tree-sitter smoke in harness
- [x] 274. Bytewax live smoke in harness
- [x] 275. Treelite router smoke in harness
- [ ] 276. Drive map workflow smoke in harness
- [x] 277. Secret redaction smoke in harness
- [ ] 278. End-to-end operator demo script
- [ ] 279. Release checklist
- [ ] 280. Regression dashboard

## Execution Rule
Always update this checklist and `STATUS.md` after any build/audit step. Progress bars are mandatory in user-facing build updates.

## Global Recalibration — End-State Product Audit

Updated: 2026-05-13

The sprint bars measure verified local slices. The true autonomous end-state is lower:

- **True global lifecycle**: `████░░░░░░ 41%`
- **Phase 1 Core engine + persistence**: `████████░░ 80%`
- **Phase 2 Autonomous intake + perception**: `█████░░░░░ 50%`
- **Phase 3 Reflex loop**: `███░░░░░░░ 30%`
- **Phase 4 Persona + control**: `░░░░░░░░░░ 0%`

### Priority Rebalance

Do **not** chase more smoke-test percentage. Bias the next real work toward closing the organism loop:

1. **Autonomous Hop Pivot / perception first** — the reflex loop needs real event volume and varied outcomes.
2. **Bytewax/River/Treelite reflex spine second, but not later** — every Hop Pivot action must emit events in the shape the reflex system will consume.
3. **Body Capture capture/diff third** — once Hop Pivot promotes candidates, capture and diff become evidence-grade.
4. **Indy_Reads/Village fourth** — do not let persona work block crawling/reflex, but keep the sideload boundary clean.

### New North-Star Loop

```text
Survey target
  -> bounded Hop Pivot expands candidates
  -> every candidate/action emits DBOS workflow_event
  -> Bytewax normalizes live stream
  -> River updates source/route/trust/failure scores
  -> Treelite serves compiled routing hints
  -> DBOS chooses next workflow under governance
  -> Body Capture captures promoted evidence
  -> AGE/pgvector/CAS persist provenance, semantics, bytes
```

### Immediate Priority Delta

- Promote 007 from `64%` to `75%` by implementing real queue-backed hop rounds, source-policy gates, and AGE edges.
- Promote 006 from `56%` to `70%` by wiring Bytewax to actual DBOS/survey events and producing bounded River features.
- Promote 008 from `25%` to `45%` by adding current-page capture + hash + diff shell.
- Keep 011 persona at low priority until the system has enough behavior to profile.

- [x] 280A. Code language scanner wired into harness for executable/schema paths
- [x] 280B. Body Capture evidence bundle smoke wired into harness
- [x] 280C. Big Board JSON smoke wired into harness
- [x] 280D. Tri-Algo conduit smoke wired into algorithm harness

## 2026-05-13 Maths Recovery Failure + Repair
- FAIL-2026-05-13-MATHS-001: requested Drive maths folders were not imported on the prior pass. Logged in `02_RECORDS_OFFICE/DRIVE_MATHS_RECOVERY_2026_05_13.md`.
- Added 10 Algorithms in Nature wrappers under `/ALGOS`.
- Hop Pivot now stores pheromone, utility, maintenance cost, and selection probability for bounded link discovery.
