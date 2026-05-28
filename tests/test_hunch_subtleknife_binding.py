from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_no_ui_suggestion_law_is_durable() -> None:
    doc = (ROOT / "00_PROJECT_BRAIN" / "PROJECT_2501_CORE_CONTRACT.md").read_text(encoding="utf-8")
    assert "Agents must not suggest UI features, dashboard tiles, or screen changes" in doc
    assert "Only the operator orders UI changes." in doc


def test_hunch_and_subtleknife_binding_sources_exist_and_are_hashed() -> None:
    binding_path = ROOT / "04_RUNTIME" / "indy_percyphon_hunch_subtleknife_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))

    assert binding["schema"] == "lucidota.runtime.indy_percyphon_hunch_subtleknife_binding.v1"
    assert binding["applies_to"] == ["INDY_READs", "PercyphonAI"]
    assert binding["authority"] == "operator_hunch_signal_and_cut_seal_protocol_not_canonical_truth"

    for source in binding["sources"]:
        path = ROOT / source["path"]
        assert path.exists(), source["path"]
        assert source["sha256"] == sha256(path)


def test_binding_rules_preserve_evidence_and_subtleknife_seal() -> None:
    binding = json.loads((ROOT / "04_RUNTIME" / "indy_percyphon_hunch_subtleknife_binding.json").read_text(encoding="utf-8"))
    rules = binding["rules"]

    assert "operator_labeled_hunches_are_high_priority_signals" in rules
    assert "hunches_require_evidence_paths_before_truth_promotion" in rules
    assert "subtle_knife_cuts_require_receipt_or_log_seal" in rules
    assert "percyphon_outputs_remain_procedural_scaffold_candidates_not_truth" in rules
    assert "indy_reads_must_not_flatten_hunch_record_into_generic_speculation" in rules
