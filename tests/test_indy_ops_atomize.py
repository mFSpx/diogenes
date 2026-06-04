from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from absurd_queue_spine import ALLOWED_JOB_KINDS, run_job
from indy_ops import handle_atomize_csv_file, handle_atomize_json_file


def test_atomize_job_kinds_are_registered():
    assert {"intake.atomize_json", "intake.atomize_csv"} <= ALLOWED_JOB_KINDS


def test_handle_atomize_json_file_atomizes_top_level_objects(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    path.write_text(json.dumps({"alpha": 1, "bravo": {"nested": True}}), encoding="utf-8")

    result = handle_atomize_json_file(path)

    assert result["schema"] == "lucidota.indy_ops.atomize_json.result.v1"
    assert result["outcome"] == "succeeded"
    assert result["record_count"] == 2
    assert result["records"][0]["record_key"] == "alpha"
    assert result["records"][1]["record_key"] == "bravo"


def test_handle_atomize_csv_file_atomizes_rows(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("name,score\nalpha,10\nbravo,20\n", encoding="utf-8")

    result = handle_atomize_csv_file(path)

    assert result["schema"] == "lucidota.indy_ops.atomize_csv.result.v1"
    assert result["outcome"] == "succeeded"
    assert result["record_count"] == 2
    assert result["header_fields"] == ["name", "score"]
    assert result["records"][0]["record_value"]["name"] == "alpha"


def test_absurd_queue_spine_dispatches_atomize_json(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    path.write_text(json.dumps([{"id": 1}, {"id": 2}]), encoding="utf-8")

    ok, result, err = run_job("intake.atomize_json", {"source_path": str(path)})

    assert ok is True
    assert err == ""
    assert result["schema"] == "lucidota.indy_ops.atomize_json.result.v1"
    assert result["record_count"] == 2
    assert result["job_kind"] == "intake.atomize_json"


def test_absurd_queue_spine_dispatches_atomize_csv(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("name,score\nalpha,10\n", encoding="utf-8")

    ok, result, err = run_job("intake.atomize_csv", {"source_path": str(path)})

    assert ok is True
    assert err == ""
    assert result["schema"] == "lucidota.indy_ops.atomize_csv.result.v1"
    assert result["record_count"] == 1
    assert result["job_kind"] == "intake.atomize_csv"
