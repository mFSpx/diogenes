from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "07_SURFACES" / "sidecars" / "promptflow_canvas.py"


def load_app():
    spec = importlib.util.spec_from_file_location("luci_flow_app", APP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flow_app_renders_real_canvas_regions():
    app = load_app()
    html = app.render_html()
    assert "LUCI /flow" in html
    assert "palette" in html
    assert "canvas" in html
    assert "Inspector" in html
    assert "console" in html
    for control in ["Save Flow", "Stage", "Run", "Validate", "Promote", "Rollback"]:
        assert control in html


def test_flow_smoke_writes_flow_json_and_receipt(tmp_path):
    app = load_app()
    app.os.environ["LUCI_FLOW_DISABLE_DB_WRITE"] = "1"
    payload = app.smoke(tmp_path)
    assert payload["status"] == "PASS"
    spec_path = ROOT / payload["spec_path"] if not Path(payload["spec_path"]).is_absolute() else Path(payload["spec_path"])
    receipt_path = ROOT / payload["receipt_path"] if not Path(payload["receipt_path"]).is_absolute() else Path(payload["receipt_path"])
    assert spec_path.exists()
    assert receipt_path.exists()
    spec = json.loads(spec_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    assert spec["flow_id"]
    assert isinstance(spec["nodes"], list) and spec["nodes"]
    assert isinstance(spec["edges"], list)
    assert receipt["action"] == "save"
    assert receipt["output_hash"]


def test_card_index_uses_refs_not_file_body_dump():
    app = load_app()
    cards = app.build_card_index(limit_per_root=2)
    assert cards
    assert any(card["type"] in {"SCRIPT", "PROMPT", "DATA", "ONTOLOGY", "CAPABILITY", "MUTATION"} for card in cards)
    assert all("body" not in card and "content" not in card for card in cards)
    assert any(card.get("ref", {}).get("path") == "scripts/indy_reads.py" for card in cards)
