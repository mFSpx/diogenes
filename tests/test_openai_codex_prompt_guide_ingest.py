import json


def test_extract_markdown_from_notebook_and_laws():
    from scripts.openai_codex_prompt_guide_ingest import extract_markdown, prompt_laws

    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Guide\n", "Prefer rg and apply_patch.\n"]},
            {"cell_type": "code", "source": ["print('skip')"]},
            {"cell_type": "markdown", "source": ["Persist until complete; avoid upfront plan preambles.\n"]},
        ]
    }
    markdown = extract_markdown(json.dumps(nb))
    laws = prompt_laws(markdown)
    assert "Prefer rg" in markdown
    assert "print('skip')" not in markdown
    assert {law["id"] for law in laws} >= {"prefer_fast_search", "persist_to_verified_work", "suppress_upfront_preamble"}


def test_prompt_policy_artifact_is_lucidota_typed():
    from scripts.openai_codex_prompt_guide_ingest import build_policy

    policy = build_policy("Persist until complete. Use tools. Preserve dirty worktree.", source_url="https://developers.openai.com/cookbook/examples/gpt-5/codex_prompting_guide")
    assert policy["schema"] == "lucidota.prompting.codex_guide_policy.v1"
    assert policy["source_url"].startswith("https://developers.openai.com/")
    assert policy["canonical_graph_writes_performed"] is False
    assert policy["model_calls_performed"] is False
