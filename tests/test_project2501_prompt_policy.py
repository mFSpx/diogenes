from pathlib import Path


def test_project2501_admin_prompt_references_codex_prompting_policy():
    text = Path("00_PROJECT_BRAIN/PROJECT_2501_ADMIN_PROMPT.md").read_text(encoding="utf-8")
    assert "CODEX_PROMPTING_GUIDE_LUCIDOTA_POLICY.json" in text
    assert "working code, not just a plan" in text.lower()
