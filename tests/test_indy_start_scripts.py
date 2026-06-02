from __future__ import annotations

from pathlib import Path


def test_indy_polycareer_watch_runs_in_own_session():
    script = Path("scripts/lucidota_start_indy_polycareer_watch.sh").read_text(encoding="utf-8")

    assert "setsid bash -c" in script
