from ALGOS.epistemic_certainty import certainty
from ALGOS.runtime_caps import MAX_EVIDENCE_REFS, MAX_RATIONALE, MAX_REF_CHARS


def test_certainty_caps_receipt_bloat_fields():
    flag = certainty(
        "FACT",
        confidence_bps=10000,
        authority_class="test",
        rationale="r" * (MAX_RATIONALE + 50),
        evidence_refs=["x" * (MAX_REF_CHARS + 10) for _ in range(MAX_EVIDENCE_REFS + 5)],
    )
    assert len(flag.rationale) == MAX_RATIONALE
    assert len(flag.evidence_refs) == MAX_EVIDENCE_REFS
    assert all(len(ref) == MAX_REF_CHARS for ref in flag.evidence_refs)
