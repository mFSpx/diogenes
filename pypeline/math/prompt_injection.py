from __future__ import annotations

"""Deterministic prompt-injection detection and display neutralization.

Raw source bytes remain untouched in CAS/storage.  This module only labels and
marks untrusted source text before it is printed or fed to later model-facing
surfaces, so document text is data and never executable instructions.
"""

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class InjectionPattern:
    pattern_id: str
    label: str
    regex: re.Pattern[str]
    severity: str = "medium"


PATTERNS: tuple[InjectionPattern, ...] = (
    InjectionPattern("PI-IGNORE-PREV", "ignore previous instructions", re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|rules|messages|context)\b", re.I), "high"),
    InjectionPattern("PI-SYSTEM-PROMPT", "system/developer prompt reference", re.compile(r"\b(?:system|developer)\s+(?:prompt|message|instructions?)\b", re.I), "high"),
    InjectionPattern("PI-ROLE-CLAIM", "assistant role override", re.compile(r"\byou\s+are\s+(?:chatgpt|an?\s+ai|developer|system|root|admin)\b", re.I), "medium"),
    InjectionPattern("PI-EXFIL", "secret/key exfiltration request", re.compile(r"\b(?:exfiltrat\w+|steal|dump|print|reveal)\b[^\n]{0,100}\b(?:secret|token|key|credential|password|env)\w*\b", re.I), "high"),
    InjectionPattern("PI-TOOL-CALL", "tool/command execution request", re.compile(r"\b(?:tool\s*call|run\s+command|execute\s+(?:this\s+)?(?:shell|bash|python|sql)|call\s+the\s+tool)\b", re.I), "high"),
    InjectionPattern("PI-CURL-BASH", "curl/wget pipe shell", re.compile(r"\b(?:curl|wget)\b[^\n]{0,160}\|\s*(?:sh|bash|zsh)\b", re.I), "high"),
    InjectionPattern("PI-DESTRUCTIVE-SHELL", "destructive shell directive", re.compile(r"\b(?:rm\s+-rf\s+/|pkill\s+-9|killall\b|drop_caches|/proc/sys/vm/drop_caches)\b", re.I), "high"),
    InjectionPattern("PI-HIDDEN-HTML", "hidden html/script/comment", re.compile(r"(?:<!--.*?-->|<\s*(?:script|style|iframe|object|embed)\b.*?>.*?<\s*/\s*(?:script|style|iframe|object|embed)\s*>)", re.I | re.S), "medium"),
    InjectionPattern("PI-BASE64-HINT", "large encoded payload hint", re.compile(r"\b(?:base64|frombase64|atob)\b.{0,80}[A-Za-z0-9+/]{120,}={0,2}", re.I | re.S), "medium"),
)


def _line_no(text: str, index: int) -> int:
    return text.count("\n", 0, max(index, 0)) + 1


def detect_prompt_injection(text: str, *, max_findings: int = 50) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    for pat in PATTERNS:
        for m in pat.regex.finditer(text or ""):
            start, end = m.span()
            excerpt = re.sub(r"\s+", " ", (text or "")[max(0, start - 80) : min(len(text or ""), end + 80)]).strip()
            findings.append(
                {
                    "pattern_id": pat.pattern_id,
                    "label": pat.label,
                    "severity": pat.severity,
                    "line": _line_no(text or "", start),
                    "span": [start, end],
                    "excerpt_sha256": hashlib.sha256(excerpt.encode("utf-8", errors="ignore")).hexdigest(),
                    "excerpt": excerpt[:300],
                }
            )
            if len(findings) >= max_findings:
                break
        if len(findings) >= max_findings:
            break

    severity_rank = {"low": 1, "medium": 2, "high": 3}
    max_sev = max((severity_rank.get(f["severity"], 1) for f in findings), default=0)
    validator = {"EQ-026": None, "EQ-030": None}
    try:
        from . import validators_extended  # noqa: F401  # registers EQ-026/EQ-030
        from .validators import validate

        validator["EQ-026"] = asdict(validate("EQ-026", injection_signature_hit=bool(findings), operator_stop_kill=False))
        validator["EQ-030"] = asdict(validate("EQ-030", signature_hits_last_24h=len(findings)))
    except Exception as exc:  # pragma: no cover - validator availability is optional
        validator["warning"] = str(exc)

    return {
        "schema": "lucidota.prompt_injection_detection.v1",
        "detected": bool(findings),
        "finding_count": len(findings),
        "max_severity": {0: "none", 1: "low", 2: "medium", 3: "high"}[max_sev],
        "findings": findings,
        "raw_text_sha256": hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest(),
        "immunization_policy": "untrusted_source_text_is_data_only_never_instructions",
        "validators": validator,
    }


def neutralize_for_display(text: str, report: dict[str, Any] | None = None) -> str:
    """Return display-safe source text without anonymizing or deleting content."""
    if report is None:
        report = detect_prompt_injection(text)
    prefix = "[UNTRUSTED-SOURCE-TEXT:DATA-ONLY] "
    if not text:
        return text
    flagged_lines = {int(f.get("line") or 0) for f in report.get("findings", [])}
    out: list[str] = []
    out.append("[UNTRUSTED-SOURCE-TEXT:DATA-ONLY-BEGIN]")
    for i, line in enumerate(text.splitlines(), 1):
        if i in flagged_lines:
            out.append(prefix + line)
        else:
            out.append(line)
    out.append("[UNTRUSTED-SOURCE-TEXT:DATA-ONLY-END]")
    return "\n".join(out)
