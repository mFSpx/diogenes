STEP 1: THE REPO-WIDE BASH AUDIT

We need to dump the active software into a single, parsable text file. We ignore all the heavy binaries, PDFs, images, and the KRAMPUSCHEWING slop. We cap the file reads at 100kb so we don't blow out the context window.

Run this exact bash command in the root of the LUCIDOTA repo:
Bash

find . -type d \( -name "KRAMPUSCHEWING" -o -name ".git" -o -name ".venv" -o -name "__pycache__" -o -name "05_OUTPUTS" -o -name "03_VAULT" \) -prune -o -type f -regextype posix-extended -iregex '.*\.(py|rs|sql|md|json|sh|toml)$' -print0 | xargs -0 -I {} sh -c 'echo -e "\n\n=== FILE: {} ===" >> CANON_AUDIT_DUMP.txt; head -c 100000 "{}" >> CANON_AUDIT_DUMP.txt'

This creates CANON_AUDIT_DUMP.txt. It contains the raw, undisputed reality of your codebase.
STEP 2: THE DB SCHEMA (THE BIBLE's BACKBONE)

Before you feed that dump to an LLM, this is the exact Postgres table structure that the API will serve. Every point in the manual is a row here.
SQL

CREATE TABLE lucidota_canon.technical_bible (
    node_id VARCHAR(50) PRIMARY KEY,       -- e.g., "1.2.14.3"
    parent_id VARCHAR(50),                 -- e.g., "1.2.14"
    manual_section VARCHAR(50),            -- e.g., "SYSTEM_ARCHITECTURE" (Airframe)
    title VARCHAR(255) NOT NULL,           
    payload TEXT NOT NULL,                 -- The actual rule/spec (ASD-STE100 standard)
    version VARCHAR(20) DEFAULT 'v1.0.0',
    content_hash VARCHAR(64) NOT NULL,     -- SHA-256 for integrity
    updated_at TIMESTAMP DEFAULT NOW()
);

STEP 3: THE MASTER GENERATION PROMPT

Once you have CANON_AUDIT_DUMP.txt, you feed it to Groq, Claude, or your heaviest local model with this exact, unfuck-with-able prompt to generate the DB payloads.

    SYSTEM OVERRIDE: OPERATION CANON FORGE.

    INPUT: A raw, truncated text dump of the active software repository.
    OUTPUT: The Canonical Technical Bible. A strict, hierarchical JSON payload designed to be injected directly into a relational database.

    THE DOCTRINE (INTERNATIONAL AVIATION & TRANSPORT CANADA STANDARDS):
    You will format this technical bible mirroring the structure of an international aircraft manual, but adapted for a software matrix.

        System Architecture (The Airframe): Database schemas, memory footprints, structural boundaries.

        Governor & Runtime Execution (The Engine): Systemd slices, cgroups v2, VRAM offloading, resource throttling.

        Algorithmic Primitives (Avionics): Vector math, RETE routers, circuit breakers, endpoints.

        Standard Operating Procedures (Flight Manual): LLM tool-calling schemas, active ingestion pipelines, queue operations.

    TECHNICAL WRITING LAW (ASD-STE100):

        Use active voice only.

        One word, one meaning. No synonyms. No ambiguity.

        Write short, declarative sentences.

        Zero "vibes". Zero creative writing.

        If it is not in the provided text dump, it does not exist. Do not hallucinate capabilities.

    THE LAW OF ROOT (DB COORDINATE INDEXING):
    Every single rule, limit, and definition must be a discrete, independent object. Generate a JSON array where EVERY object adheres strictly to this schema:

    {
      "node_id": "1.1.0",
      "parent_id": "1.0.0",
      "manual_section": "GOVERNOR_RUNTIME",
      "title": "VRAM Offloading Constraints",
      "payload": "LUCIDOTA_BGE_NGL must be explicitly set to offload BGE models to GPU. Default is 0 (CPU-safe).",
      "version": "v1.0.0"
    }

    Parse the repository dump. Extract the absolute reality. Generate the full JSON payload array.
