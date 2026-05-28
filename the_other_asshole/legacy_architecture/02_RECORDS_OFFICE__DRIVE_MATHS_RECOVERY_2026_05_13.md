# Drive Maths Recovery — 2026-05-13

## Operator order
Recover the maths / algorithm material from Drive and make it available to the build.

## Failure record
- FAIL-2026-05-13-MATHS-001: prior pass did not import or locally register the requested PYPELINE/MATHS and PYPELINE/MATHS_CORE material. This is now logged as a build failure and recovery item.

## Drive search results
- Exact folder search: `MATHS_CORE` returned no folder hit.
- PYPELINE folder found: `PYPELINE` (`1mxuyGIspG6zPFbLUcIFt3lF62rZk-92V`). Listing showed `SELF_DOX_2026-05-01`, `legal_authority_system`, and `.env`; no visible `MATHS` or `MATHS_CORE` child folder in that listing.
- Maths folder found: `math-intrinsics` (`1uB8FG2jRYdGpH-u-G6yym11AuyUghfN6`). Listing showed `constants`, `.github`, and `test`; those child folders listed empty through the connector.
- Math document found and fetched through Drive connector: `01_MATH_PATTERNS_TRAC_TITLES.md` (`1oDasNdr1h5owJA1dQZ5ie_piHvXFXACQ`). Duplicate found: `1NAdep1RLx5r28WxPRTAPd7yvF7uckHKh`.

## Local build actions
- Added local algorithm wrappers under `/ALGOS` for the Algorithms in Nature primitives supplied by the operator.
- Extended `scripts/lucidota_algos_smoke.py` to prove all local algorithm wrappers import and execute.
- Wired pheromone-weighted bounded link selection into Hop Pivot.

## Remaining recovery item
Find the exact `PYPELINE/MATHS` and `PYPELINE/MATHS_CORE` source folders or archive path if they exist outside the visible PYPELINE folder listing. Do not treat the current local `/ALGOS` folder as a complete substitute for the missing Drive folders.
