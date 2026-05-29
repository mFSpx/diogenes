# TICKLETRUNK TOOLS ACCESS LAYER

TICKLETRUNK is the canonical manifest and access layer for the operator's proof hoard: every sovereign tool, algo, model, workflow, book, LoRA, scraper, skill, plugin, service, surface, schema, reusable fragment, and weird experimental instrument in the filesystem.

Machine manifest: `00_PROJECT_BRAIN/TICKLETRUNK.json`
Human manifest: `00_PROJECT_BRAIN/TICKLETRUNK.md`
Regenerate: `python3 scripts/tickletrunk_scan.py --execute`

Hard laws:
- Check TICKLETRUNK before writing new tools.
- Copy/adapt from the proof hoard when useful.
- Do not mutate sandbox originals without explicit operator instruction.
- Missing imports are not evidence of uselessness.
- Production gates apply to production copies, not sovereign proof-hoard originals.

This directory is for navigation. It uses symlinks when safe and README references otherwise.
