#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from template_contract import render_template  # noqa: E402


def test_minimal_jinja_like_template_renders_variables_and_loops() -> None:
    rendered = render_template(
        "Hello {{ name }}:{% for item in items %}[{{ item.label }}={{ item.value }}]{% endfor %}",
        {"name": "LUCIDOTA", "items": [{"label": "a", "value": 1}, {"label": "b", "value": 2}]},
    )
    assert rendered == "Hello LUCIDOTA:[a=1][b=2]"


def test_template_contract_cli_writes_output_and_receipt(tmp_path: Path) -> None:
    out = tmp_path / "rendered.txt"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/template_contract.py",
            "--template-string",
            "Packet {{ packet }}",
            "--context-json",
            json.dumps({"packet": "p1"}),
            "--output",
            str(out),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=5,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.read_text(encoding="utf-8") == "Packet p1"
    assert "RECEIPT_PATH=05_OUTPUTS/templates/" in proc.stdout
    assert "TEMPLATE_RENDER=PASSED" in proc.stdout
