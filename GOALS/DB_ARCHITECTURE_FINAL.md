# Final Database Architecture: Spreadsheet vs Database

**Override Instruction:** AI are not in charge. They are respected peers.
Old traditional MATHS rules. DB is designed before any ingestion proceeds.

---

## The Vision

```
SPREADSHEET VIEW                    DATABASE VIEW
(what humans see)                    (what Postgres stores)
                                    
┌─────────────────┐                ┌──────────────────┐
│ Cell A1: value   │    maps to     │ SELECT * FROM     │
│ Cell B2: value   │  ──────────→  │   table WHERE     │
│ Cell C3: value   │                │   row = condition │
└─────────────────┘                └──────────────────┘
                                    
Every spreadsheet cell = a database query.
Every database row = spreadsheet row in a normalized table.
```

---

## Core Tables (The Database View)

### 1. KRAMPUSCHEWING Hypertimeline — `lucidota_korpus.krampus_hypertimeline`

**Purpose:** Every single one of 35,443 files as a row. This is the source of truth for "what existed and when."

| Column | Type | Spreadsheet Analogy |
|--------|------|-------------------|
| `entry_id` | UUID PK | Row ID |
| `file_path` | TEXT | Column A: File path |
| `file_name` | TEXT | Column B: File name |
| `file_ext` | TEXT | Column C: Extension |
| `file_size_bytes` | BIGINT | Column D: Size |
| `sha256` | TEXT (UNIQUE) | Column E: SHA256 hash |
| `file_mtime` | TIMESTAMPTZ | Column F: Last modified |
| `ingested_at` | TIMESTAMPTZ | Column G: When ingested |
| `timeline_bucket` | TEXT | Column H: deprecated/archived/active/corpse |
| `status` | TEXT | Column I: pending/scanned/hashed/ingested/error |
| `lore_note` | TEXT | Column J: Human-readable note |
| `source_context` | TEXT | Column K: Where it came from |
| `parent_entry_id` | UUID → self | Column L: Parent file |
| `graph_promoted` | BOOLEAN | Column M: In graph? |
| `graph_promoted_at` | TIMESTAMPTZ | Column N: When promoted |
| `receipt_path` | TEXT | Column O: Receipt location |

**Indexes:** bucket+mtime, ext+bucket, sha256, status

---

### 2. BRAG Cells — `lucidota_korpus.brag_cell`

**Purpose:** Content chunks from ALL documents (books, manuals, everything).

| Column | Type | Spreadsheet Analogy |
|--------|------|-------------------|
| `cell_id` | UUID PK | Row ID |
| `chunk_id` | TEXT UNIQUE | Chunk identifier |
| `source` | TEXT | Source document path |
| `pass_name` | TEXT | GO-25 / O-75 / ROOT-414 |
| `ontology_tags` | TEXT[] | Ontology categories |
| `percyphon_slot` | INT | Percyphon 128-slot coordinate |
| `text` | TEXT | The actual content |
| `token_estimate` | INT | Token count |
| `sha256` | TEXT | Content hash |
| `xhash` | TEXT | Integrity chain hash |
| `rete_decision` | JSONB | RETE bandit routing metadata |
| `metadata` | JSONB | LTC intensity, temporal ordering |
| `embedding` | vector(384) | Vector embedding (future) |
| `created_at` | TIMESTAMPTZ | When chunked |

---

### 3. Percyphon Villagers — `lucidota_korpus.percyphon_villager`

**Purpose:** 128 deterministic identities per seed. The "village" of procedural entities.

| Column | Type | Spreadsheet Analogy |
|--------|------|-------------------|
| `villager_id` | UUID PK | Row ID |
| `slot` | INT (0-127) | 128-slot coordinate |
| `seed` | TEXT | Deterministic seed |
| `name` | TEXT | Generated name |
| `persona` | TEXT | Archetype persona |
| `identity_hash` | TEXT | xxhash128 identity |
| `slot_type` | TEXT | fixed_identity / procedural |
| `created_at` | TIMESTAMPTZ | When generated |

---

### 4. Canonical Graph Edges — `lucidota_canon.graph_edge`

**Purpose:** The GO-25 edge topology connecting everything. This is the GRAPH the user wants.

| Column | Type | Spreadsheet Analogy |
|--------|------|-------------------|
| `edge_uuid` | UUID PK | Row ID |
| `source_uuid` | UUID → any table | Source node |
| `target_uuid` | UUID → any table | Target node |
| `edge_type` | TEXT | Relationship type |
| `relationship_family` | TEXT | intimate/interpersonal/vector/etc |
| `status` | TEXT | located/staged/approved/rejected/etc |
| `valid_from` | TIMESTAMPTZ | When edge became valid |
| `valid_to` | TIMESTAMPTZ | When edge expired (NULL = current) |
| `evidence` | JSONB | Supporting evidence refs |
| `ontology_tags` | TEXT[] | GO-25 ontology tags |

---

## The Spreadsheet View (PostgREST)

Every table above is accessible via PostgREST as a RESTful spreadsheet:

```
# All KRAMPUSCHEWING files as a spreadsheet
GET /krampus_hypertimeline?select=file_path,file_name,file_size_bytes,sha256,timeline_bucket,status

# All BRAG chunks for a specific book
GET /brag_cell?source=like.*The_Prince*&select=chunk_id,ontology_tags,token_estimate,xhash

# Graph edges for a specific file
GET /graph_edge?source_uuid=eq.<file_uuid>&select=edge_type,target_uuid,relationship_family,status

# The FULL hypertimeline spreadsheet (for export to CSV)
GET /krampus_hypertimeline?limit=1000&order=file_mtime.desc
```

---

## Data Flow: File → Graph

```
KRAMPUSCHEWING file on disk
  │
  ▼
krampus_hypertimeline row        ← file metadata + SHA256 + timeline bucket
  │
  ├──→ brag_cell rows            ← content chunks (if text/parseable)
  │       │
  │       └──→ ontology_tags     ← GO-25/O-75 ontology
  │
  ├──→ percyphon_villager row    ← 128-slot identity scaffold
  │
  └──→ graph_edge rows           ← GO-25 edge topology
          │
          ├──→ source = krampus_hypertimeline.entry_id
          ├──→ target = brag_cell.cell_id (or another file)
          └──→ edge_type = ontology_tags join
```

---

## Old Traditional MATHS

Every hash is SHA256 (deterministic, no randomness).
Every row has a UNIQUE constraint (no duplicates).
Every timestamp is TIMESTAMPTZ (timezone-aware).
Every join is via UUID (no string matching).
Every status has a CHECK constraint (no invalid states).
Every edge is directed and typed (GO-25 ontology).

The database is the source of truth.
The spreadsheet is a view of the database.
AIs query the database. They do not own it.
Maths rule. Determinism wins. Chaos is for creativity, not for data.
