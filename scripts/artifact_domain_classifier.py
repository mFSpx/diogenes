from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_LEGAL = ["bcfsa", "rtb", "case", "evidence", "forensic", "chronology", "affidavit",
          "exhibit", "landlord", "tenant", "northern_strike", "nordley",
          "master_case_file", "investigation"]
_COMMS = ["email", "gmail", "takeout", "mail", "sms", "chat", "messenger", "whatsapp",
          "telegram", "imessage", "commdump", "chatdump", "nordley_squeezecopy/takeout"]
_MEDIA_PATH = ["pictures", "photos", "dcim", "camera", "screenshots", "picture", "gallery"]
_MEDIA_EXCLUDE = ["case", "evidence", "forensic"]
_DEV = ["01_repos", "scripts", "algos", "tools", "cargo.toml", "src/", ".git",
        "lucidota", "krampuschewing", "00_project_brain", "crates/"]
_JOURNALISM = ["substack", "article", "press", "journalism", "publishing_desk", "newsletter"]
_BUSINESS = ["invoice", "contract", "financial", "accounting", "receipt", "business",
             "payroll", "tax", "gst", "hst", "t4", "t1"]
_WEB_DESIGN = ["figma", "mockup", "wireframe", "web_design", "webdesign", ".fig", ".psd",
               "site", "theme", "template"]
_RESEARCH = ["research", "paper", "study", "academic", "arxiv", "pubmed", "reference",
             "bibliography"]
_PROJECTS_INCLUDE = ["projects/", "project_"]
_PROJECTS_EXCLUDE = ["01_repos", "scripts", "algos"]


def _any_in(needles: list[str], haystack: str) -> bool:
    return any(n in haystack for n in needles)


def _groq_classify(path: Path, text_snippet: str, groq_api_key: str) -> tuple[str, str | None]:
    try:
        import httpx
        prompt = (
            f"Classify this file into exactly one domain. File path: {path}. "
            f"Content snippet (first 200 chars): {text_snippet[:200]}\n"
            "Domains: legal, personal, comms, media, research, dev, projects, journalism, "
            "business, web_design, assets, unknown\n"
            'Respond with JSON: {"domain": "<domain>", "sub_domain": null or "<sub>"}'
        )
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_api_key}"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=10,
        )
        resp.raise_for_status()
        content: str = resp.json()["choices"][0]["message"]["content"]
        start = content.find("{")
        end = content.rfind("}") + 1
        data: dict[str, Any] = json.loads(content[start:end])
        return (str(data.get("domain", "unknown")), data.get("sub_domain"))
    except Exception:
        return ("unknown", None)


def classify_domain(
    path: Path, text_snippet: str = "", groq_api_key: str = ""
) -> tuple[str, str | None]:
    p = str(path.resolve()).lower()

    if _any_in(_LEGAL, p):
        return ("legal", None)

    if _any_in(_COMMS, p):
        return ("personal", "comms")

    if _any_in(_MEDIA_PATH, p) and not _any_in(_MEDIA_EXCLUDE, p):
        return ("personal", "media")

    if _any_in(_DEV, p):
        return ("dev", None)

    if _any_in(_JOURNALISM, p):
        return ("journalism", None)

    if _any_in(_BUSINESS, p):
        return ("business", None)

    if _any_in(_WEB_DESIGN, p):
        return ("web_design", None)

    if _any_in(_RESEARCH, p):
        return ("research", None)

    if _any_in(_PROJECTS_INCLUDE, p) and not _any_in(_PROJECTS_EXCLUDE, p):
        return ("projects", None)

    if groq_api_key and text_snippet:
        return _groq_classify(path, text_snippet, groq_api_key)

    return ("unknown", None)


if __name__ == "__main__":
    print("BUILT: artifact_domain_classifier.py")
