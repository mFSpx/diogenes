# DIOGENES Status

## Current State

- Umbrella workspace renamed to `/home/mfspx/LUCIDOTA`; `/home/mfspx/DIOGENES` remains a compatibility symlink.
- `LUCIDOTA` is the private umbrella project/repo. `doggystyle` remains the kernel-only repo.
- `claudecode` is now treated as the LUCIDOTA-owned interface fork, not an external dependency.
- Active long-term goal recorded: get the working sovereign local system done, with verified code over vibes.
- Repo-level ignore policy covers secrets, credentials, runtime state, vault data, databases, model weights, logs, and build outputs.
- VIBESCONTROL now lives in the repo as project-brain/audit/self-improvement source, not build code.
- Project brain initialized.
- `doggystyle` cloned into `01_REPOS`.
- `claudecode` cloned into `01_REPOS`.
- GitLab CLI installed system-wide.
- Codex global config includes project working directives.
- `THEPLAN.md` saved in project brain.
- Anti-slop audit doctrine added.
- Linux dev toolkit installed from Pop/Ubuntu repos.
- NVIDIA GTX 1650 is visible, idle, and not driving display workload (`Disp.A Off`, 2 MiB VRAM observed).
- CKDOG1 Python package installs in local venv.
- CKDOG1 test suite passes: 45 tests.
- CKDOG1 CLI smoke passes: doctor, genesis, ontology init, soul create, trait set, semantic route, route.
- CKDOG1 gRPC bridge exists and passes live smoke on `127.0.0.1:50051`.
- Rust/Cargo installed system-wide.
- Clawd release build succeeds.
- Clawd exposes `diogenes-smoke`, which starts CKDOG1 locally and calls it through generated Rust tonic/prost gRPC bindings.
- Rust workspace tests pass, including doc-tests.
- Combined LUCIDOTA/DIOGENES harness passes after the Rust tonic bridge and workspace rename.
- Rustdoc dynamic linker path repaired system-wide for the distro Rust package.
- Clawd plugin hook runner now tolerates hook scripts that exit without reading stdin.
- Private GitLab remote created and pushed: `https://gitlab.com/mfspx/LUCIDOTA`.
- Google account is connected for Drive/Contacts as `maroonedpilot@gmail.com`; Gmail and Calendar tools are not exposed in this session yet.
- DBOS/Postgres topology note created.
- PostgreSQL 18.3 installed from PGDG; `lucidota_state` and `lucidota_graph` databases created.
- `lucidota_graph` has verified `vector` 0.8.2 and `age` 1.7.0 extension behavior.
- Local Postgres role `mfspx` is a dev superuser so AGE can be loaded from normal dev commands; this is local dev posture, not release policy.
- DBOS 2.21.0 smoke workflow runs against `lucidota_state` and initializes DBOS system schema.
- Private Morrowind-inspired Clawd UI track recorded; functionality/code/layouts can live in the private dev repo, while literal copyrighted game assets stay local/ignored and are not public release material.
- Drive mapping is in progress; PYPELINE/math/secrets/scraper/template nuclei located. Earlier wrong target-dossier intake is not canonical and must not be used unless reauthorized.
- Survey Protocol v0 product slice exists: `claw lucidota-survey <target>` performs URL/file triage, local CAS write, multi-pattern keyword scan, structural HTML summary, pivot extraction, Wayback lookup option, and Postgres persistence.
- UI/image tooling installed for private theme work: `chafa`, ImageMagick, `optipng`, `pngquant`, `caca-utils`.
- Indy_Reads runtime brief exists: `claw indy-brief` reads local project brain + Postgres only, reports cited next moves, active memory, quiet side queue counts, and redacted auth inventory.

## Next Verification

- Begin DBOS/Postgres storage design after the kernel/interface bridge stays green.
- Add operational Gmail/Calendar tooling; current Google access is not enough for email/calendar assistant work.
- Body Capture v0: `█████████░ 93%` — HTTP-body capture to CAS now gated by Tri-Algo conduit, SHA delta, content/structure hashes, watcher decisions, Wayback/text diff, evidence bundle export, AGE CAPTURED_AS edge, harness smoke.

- Wake Bus: `████████░░ 80%` — Postgres outbox refs + local signal worker; truth remains DBOS/Postgres.
- CAS GC: `████████░░ 80%` — report-first mark/sweep, durable run/candidate tables, optional quarantine-only apply path, no delete mode.
- Storage decision matrix: `██████████ 100%` — Postgres/pgvector/AGE/local CAS locked; Cassandra non-canonical; MinIO deferred adapter only.
- Validated 000-007 green slice: `██████████ 100%` — full harness green across docs/control, CKDOG1, Clawd, Postgres, CAS, DBOS, Bytewax/River/Treelite hints, and Survey/Hop safety gates. Global backlog remains separately tracked.
- Overall product: `████████░░ 80%`
- gRPC bridge: `███████░░░ 70%` — Rust tonic/prost to Python grpcio smoke works; full API pending.
- DBOS workflow plane: `███████░░░ 70%` — installed/smoked/schema/events plus Survey DBOS wrapper; workflow registry and source policies seeded; full workflow ownership pending.
- Postgres/AGE/pgvector: `█████████░ 92%` — installed/verified/core schemas; final ontology/vector/vault schemas pending.
- Survey/hop-pivot: `████████░░ 80%` — working slice plus bounded hop v1; scraper ladder pending.
- River ML: `█████████░ 92%` — River scores workflow events; live Bytewax stream pending.
- Bytewax: `█████████░ 92%` — installed/imports; mini TestingSource graph emits persisted hints.
- Treelite: `█████████░ 92%` — installed/imports; Treelite artifact + advisory route run persisted; DBOS policy call-in pending.
- Drive map/import: `███░░░░░░░ 30%` — nuclei mapped; full private granular map/imports pending.
- Indy_Reads persona: `█████░░░░░ 52%` — runtime contract, local brief, task memory, reminders/calendar-intent queue, and redacted auth inventory now exist; Drive persona corpus and real Gmail/Calendar adapters pending.
- Algorithm primitive library: `██████████ 98%` — 40 wrappers smoked, including Possum/OPOSSUM/Thanatosis/Capybara/Hoeffding and Chelydra/Serpentina/Schoolfield additions.
- Progress UI: `██████░░░░ 60%` — checklist progress printer plus read-only Big Board v0 with live workflow/wake/CAS/Body Capture/reflex/GPU counters and JSON export.

Full audited checklist: `00_PROJECT_BRAIN/BUILD_PLAN_AUDIT.md` (280+ line-items). Latest full harness passed after Body Capture evidence bundles, Big Board v0, and 23-module algorithm smoke wiring.

- math.zip top-level recovery: `██████████ 100%` — `/home/mfspx/Downloads/math.zip` copied, extracted, manifested, and converted into discrete ALGOS modules; full system hardwire audit written.
- Ruthless audit fixes: Wake Bus CTE batch delivery, Body Capture graph-local workflow-event outbox, and Survey explicit IP gate landed.
