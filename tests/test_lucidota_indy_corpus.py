from __future__ import annotations


def test_indy_corpus_builds_from_current_project_brain_docs():
    import scripts.lucidota_indy_corpus as corpus

    artifact = corpus.build_corpus()

    assert artifact["ok"] is True
    assert artifact["unit_count"] >= 20
    assert artifact["artifact_sha256"]
    assert all(src["exists"] for src in artifact["sources"])


def test_indy_brief_corpus_summary_keeps_distillation():
    import scripts.lucidota_indy_brief as brief

    summary = brief.corpus_summary()

    assert summary["unit_count"] >= 20
    assert summary["distilled"]["runtime_jobs"]
