from __future__ import annotations


def test_vibe_sequencer_run_groq_with_no_key_is_skip_marker(monkeypatch):
    import scripts.vibe_sequencer as seq

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert seq.run_groq("check my prompt") == "SKIP: GROQ_API_KEY is not set"


def test_vibe_sequencer_groq_job_is_skipped_without_key(monkeypatch, tmp_path):
    import scripts.vibe_sequencer as seq

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    target = tmp_path / "skip.txt"
    result = seq.execute_job(
        {
            "label": "safe-skip",
            "prompt": "run groq only if key exists",
            "model": "groq",
            "target_file": str(target),
        }
    )

    assert result["status"] == "skipped"
    assert result["error"] == "SKIP: GROQ_API_KEY is not set"
    assert not target.exists()


def test_ocr_image_without_groq_key_is_safe_noop(monkeypatch):
    import scripts.corpus_groq_extractor as cge

    monkeypatch.setattr(cge, "GROQ_API_KEY", "", raising=False)
    assert cge.ocr_image(b"not-a-real-image", "image/png") == ""
