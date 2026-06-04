from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "talkie_moe_build.py"
MATH = ROOT / "04_RUNTIME" / "RUNPOD_ACCEL" / "TALKIE_4X_MOE_MATH.md"


def load_build_module():
    assert SCRIPT.exists(), f"missing build helper: {SCRIPT}"
    spec = importlib.util.spec_from_file_location("talkie_moe_build", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_talkie_moe_build_parses_exact_4x_math(tmp_path: Path) -> None:
    build = load_build_module()
    receipt = tmp_path / "talkie_moe_router_build.json"

    rc = build.main(["--math", str(MATH), "--out", str(receipt)])

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["schema"] == "lucidota.runpod.talkie_moe_router_build.v1"
    assert payload["status"] == "PASS"
    assert payload["math_path"] == str(MATH)
    assert payload["full_4x_bf16_bytes"] == 106242061768
    assert payload["shared_base_plan"] == "one frozen base + four adapter banks + macro router + micro router"
