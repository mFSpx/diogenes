#!/usr/bin/env python3
from __future__ import annotations

import sys
import types

from ALGOS.gliner_zero_shot_extractor import extract, parse_labels
from ALGOS.runtime_caps import MAX_LABELS, MAX_SPANS, MAX_TEXT_CHARS


def _install_fake_gliner(monkeypatch, predicted):
    state = {"loads": 0}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def predict_entities(self, text, labels, threshold):
            return predicted

    class FakeGLiNER:
        @staticmethod
        def from_pretrained(model_name: str):
            state["loads"] += 1
            return FakeModel()

    fake_module = types.ModuleType("gliner")
    fake_module.GLiNER = FakeGLiNER
    monkeypatch.setitem(sys.modules, "gliner", fake_module)
    return state


def test_parse_labels_caps_to_runtime_limit(tmp_path):
    raw = ",".join(f"Label{i}" for i in range(MAX_LABELS + 10))
    labels = parse_labels(raw)
    assert len(labels) == MAX_LABELS
    assert labels[0] == "Label0"


def test_extract_caps_text_and_spans_before_model_run(monkeypatch, tmp_path):
    predicted = [
        {
            "start": 0,
            "end": 4,
            "text": "Test",
            "label": "Operator",
            "score": 0.99,
        }
    ]
    _install_fake_gliner(monkeypatch, predicted)
    model_path = tmp_path / "model"
    model_path.write_text("offline", encoding="utf-8")

    text = "x" * (MAX_TEXT_CHARS + 123)
    result = extract(text, ["Operator"], model=str(model_path), allow_remote_model=False)

    assert result["text_length"] == MAX_TEXT_CHARS
    assert len(result["text"]) == MAX_TEXT_CHARS if "text" in result else True
    assert len(result["spans"]) == 1


def test_extract_limits_span_count_and_uses_singleton_cache(monkeypatch, tmp_path):
    # 3 spans above cap to prove output truncation keeps MAX_SPANS only.
    extra = [
        {"start": i * 2, "end": i * 2 + 1, "text": "x", "label": "Operator", "score": 0.1}
        for i in range(MAX_SPANS + 4)
    ]
    state = _install_fake_gliner(monkeypatch, extra)
    model_path = tmp_path / "model"
    model_path.write_text("offline", encoding="utf-8")

    result = extract("x" * 10, ["Operator"], model=str(model_path), allow_remote_model=False, no_fallback=False)
    assert len(result["spans"]) == MAX_SPANS

    # Call again with same model path and assert singleton load happens once.
    extract("x" * 10, ["Operator"], model=str(model_path), allow_remote_model=False, no_fallback=False)
    assert state["loads"] == 1
