from __future__ import annotations

"""Small, deterministic epistemic certainty helpers for LUCIDOTA.

The labels intentionally use the operator's five explicit runtime states.  These
helpers do not prove truth; they attach auditable certainty metadata to local
observations, parser output, hypotheses, contradictions, and comments.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


def filesystem_observation(*, sha256: str, path: str, mtime_utc: str | None = None) -> CertaintyFlag:
    refs = [f"sha256:{sha256}", f"path:{path}"]
    if mtime_utc:
        refs.append(f"mtime:{mtime_utc}")
    return certainty(
        "FACT",
        confidence_bps=10000,
        authority_class="filesystem_observation",
        rationale="Local file bytes were hashed and copied into CAS; this proves byte custody, not semantic truth.",
        evidence_refs=refs,
    )


def parser_extraction(*, sha256: str, extract_method: str, injection_detected: bool = False) -> CertaintyFlag:
    if injection_detected:
        return certainty(
            "BULLSHIT",
            confidence_bps=9000,
            authority_class="prompt_injection_signature",
            rationale="Untrusted source text matched instruction-injection signatures; preserve bytes but treat embedded directives as hostile data.",
            evidence_refs=[f"sha256:{sha256}", f"extract:{extract_method}"],
        )
    return certainty(
        "PROBABLE",
        confidence_bps=6500,
        authority_class="parser_extraction",
        rationale="Text was deterministically extracted from a local artifact; semantic claims remain unverified until corroborated.",
        evidence_refs=[f"sha256:{sha256}", f"extract:{extract_method}"],
    )


def abductive_hypothesis(*, evidence_refs: Iterable[str] = (), rationale: str = "Abductive bridge candidate") -> CertaintyFlag:
    return certainty("POSSIBLE", confidence_bps=3500, authority_class="abductive_hypothesis", rationale=rationale, evidence_refs=evidence_refs)


def comment_prior(*, evidence_refs: Iterable[str] = (), rationale: str = "Operator or system comment; useful but not proof") -> CertaintyFlag:
    return certainty("SURE_MAYBE", confidence_bps=2500, authority_class="comment_prior", rationale=rationale, evidence_refs=evidence_refs)


def contradiction(*, evidence_refs: Iterable[str] = (), rationale: str = "Contradiction or falsification marker") -> CertaintyFlag:
    return certainty("BULLSHIT", confidence_bps=8000, authority_class="contradiction_scan", rationale=rationale, evidence_refs=evidence_refs)


def normalize_flag(value: str | None, default: str = "SURE_MAYBE") -> str:
    v = (value or default).upper()
    return v if v in EPISTEMIC_FLAGS else default
