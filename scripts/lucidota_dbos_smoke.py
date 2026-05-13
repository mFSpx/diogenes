from __future__ import annotations

import json
import os

from dbos import DBOS, DBOSConfig


@DBOS.step()
def normalize_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace(" ", "_")


@DBOS.workflow()
def smoke_workflow(symbol: str) -> dict[str, str]:
    normalized = normalize_symbol(symbol)
    return {"input": symbol, "normalized": normalized}


def main() -> int:
    config: DBOSConfig = {
        "name": "lucidota-dbos-smoke",
        "system_database_url": os.environ.get("DBOS_SYSTEM_DATABASE_URL"),
    }
    DBOS(config=config)
    DBOS.launch()
    result = smoke_workflow("northern strike")
    print(json.dumps({"ok": True, "dbos": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
