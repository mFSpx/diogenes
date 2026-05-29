#!/usr/bin/env python3
from pathlib import Path
import py_compile

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    ROOT / "scripts" / "lucidota_river_reflex.py",
    ROOT / "scripts" / "lucidota_bytewax_mini.py",
    ROOT / "scripts" / "lucidota_stream_river_worker.sh",
]


def test_root_river_bytewax_scripts_exist_compile_and_stay_small():
    for path in SCRIPTS:
        assert path.exists(), f"missing root runtime script: {path}"
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 100
        if path.suffix == ".py":
            py_compile.compile(str(path), doraise=True)


def test_stream_worker_does_not_swallow_tick_failures():
    source = (ROOT / "scripts" / "lucidota_stream_river_worker.sh").read_text(encoding="utf-8")
    assert "|| true" not in source
    assert "lucidota_bytewax_mini.py" in source
    assert "lucidota_river_reflex.py" in source
