# SCHEMA AGENT STARTUP LAW

1. Read root `CLAUDE.md` + `AGENTS.md` first.
2. Never edit an existing numbered SQL file after it's been applied.
3. Create a new `NNN_description.sql` with the next number.
4. Test with `psql -f <file>` before declaring done.
5. Document the purpose of each table in a comment at the top of the file.
6. New tables need entries in `OFFICIAL_ONTOLOGY.json`.
