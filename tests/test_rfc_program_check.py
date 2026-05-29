import subprocess
import sys


def test_rfc_program_check_passes_and_counts_subjects():
    result = subprocess.run(
        [sys.executable, "scripts/rfc_program_check.py"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RFC_PROGRAM_CHECK=PASS" in result.stdout
    assert "SUBJECTS=20" in result.stdout
    assert "MISSING_EVIDENCE=0" in result.stdout
    assert "BOUNDARY_VIOLATIONS=0" in result.stdout


def test_rfc_program_check_enforces_draft_section_families():
    result = subprocess.run(
        [sys.executable, "scripts/rfc_program_check.py"],
        text=True,
        capture_output=True,
    )
    assert "DRAFT_SECTION_VIOLATIONS=0" in result.stdout, result.stdout + result.stderr


def test_rfc_program_check_rejects_placeholder_subjects_and_missing_rfc_files():
    result = subprocess.run(
        [sys.executable, "scripts/rfc_program_check.py"],
        text=True,
        capture_output=True,
    )
    assert "RFC_FILES_MISSING=0" in result.stdout, result.stdout + result.stderr
    assert "SEEDED_OR_PLACEHOLDER_SUBJECTS=0" in result.stdout, result.stdout + result.stderr


def test_rfc_program_check_enforces_depth_and_source_coverage():
    result = subprocess.run(
        [sys.executable, "scripts/rfc_program_check.py"],
        text=True,
        capture_output=True,
    )
    assert "RFC_DEPTH_VIOLATIONS=0" in result.stdout, result.stdout + result.stderr
    assert "RFC_SOURCE_COVERAGE_VIOLATIONS=0" in result.stdout, result.stdout + result.stderr


def test_rfc_program_check_enforces_claim_discipline_and_doc_authority():
    result = subprocess.run(
        [sys.executable, "scripts/rfc_program_check.py"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RFC_CLAIM_DISCIPLINE_VIOLATIONS=0" in result.stdout
    assert "PROJECT_BRAIN_DOC_AUTHORITY_VIOLATIONS=0" in result.stdout
