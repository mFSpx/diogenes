# 06_SCHEMA — Postgres SQL Contract Layer

Numbered SQL files (`NNN_*.sql`) that define the database contract layer.

## Key Rules

- **Numbered in order.** New schemas get the next number. Check `ls *.sql | tail -1` first.
- **Immutable once applied.** Schema changes use new migrations, not edits to old files.
- **Two databases:** `lucidota_state` (workflow/runtime/control) and `lucidota_storage` (graph items, edges, layers).
- **Every table must have a primary key** (UUID preferred).
- **Every table must have `created_at` with `now()` default.**
- **Every mutation table needs an audit trail** (event/dead_letter/journal pattern).

## Conventions

- `lucidota_control.*` — workflow/runtime/control tables
- `lucidota_go.*` — graph ontology tables
- `lucidota_canon.*` — canonical/routing tables
- `lucidota_korpus.*` — corpus/ingestion tables
