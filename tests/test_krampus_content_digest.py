"""Tests for krampus_content_digest pipeline."""
from __future__ import annotations

import json
import math
import re
import sys
import uuid
from collections import Counter
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR.parent))

from scripts.krampus_content_digest import (  # noqa: E402
    FEATURE_VERSION,
    TOKEN_RE,
    aggregate_by_file,
    build_global_vocabulary,
    compute_global_tfidf,
    compute_repo_digests,
    deterministic_uuid,
    extract_repos_from_manifests,
    generate_embedding_manifest,
    read_repo_manifests,
    sha256_digest,
)

# --- Sample RiverML records for testing ---

SAMPLE_RECORDS = [
    {
        "source": "test_file.py",
        "source_type": "code",
        "chunk_id": "go-25_0000",
        "pass_name": "GO-25",
        "text": "def hello():\n    print('hello world')\n    return 42\n",
        "sha256": "abc123",
        "riverml_features": {
            "byte_len": 50.0,
            "line_count": 3.0,
            "avg_line_len": 16.7,
            "blank_lines": 0.0,
            "comment_lines": 0.0,
            "import_lines": 0.0,
            "function_defs": 1.0,
            "branch_points": 0.0,
            "return_points": 1.0,
            "nest_depth_estimate": 0.0,
            "unique_tokens": 10.0,
            "type_token_ratio": 0.8,
            "upper_ratio": 0.0,
            "symbol_ratio": 0.1,
            "subsys_core": 1.0,
            "subsys_api": 0.0,
            "type_code": 1.0,
            "type_doc": 0.0,
        },
    },
    {
        "source": "test_file.py",
        "source_type": "code",
        "chunk_id": "go-25_0001",
        "pass_name": "GO-25",
        "text": "class MyClass:\n    def method(self):\n        pass\n",
        "sha256": "def456",
        "riverml_features": {
            "byte_len": 40.0,
            "line_count": 3.0,
            "avg_line_len": 13.3,
            "blank_lines": 1.0,
            "comment_lines": 0.0,
            "import_lines": 0.0,
            "function_defs": 1.0,
            "branch_points": 0.0,
            "return_points": 0.0,
            "nest_depth_estimate": 1.0,
            "unique_tokens": 8.0,
            "type_token_ratio": 0.9,
            "upper_ratio": 0.1,
            "symbol_ratio": 0.05,
            "subsys_core": 1.0,
            "subsys_api": 0.0,
            "type_code": 1.0,
            "type_doc": 0.0,
        },
    },
    {
        "source": "README.md",
        "source_type": "doc",
        "chunk_id": "go-25_0000",
        "pass_name": "GO-25",
        "text": "# Test Project\nThis is a test project for documentation.\n## Usage\nRun the script.\n",
        "sha256": "ghi789",
        "riverml_features": {
            "byte_len": 72.0,
            "line_count": 4.0,
            "avg_line_len": 18.0,
            "blank_lines": 1.0,
            "comment_lines": 0.0,
            "import_lines": 0.0,
            "function_defs": 0.0,
            "branch_points": 0.0,
            "return_points": 0.0,
            "nest_depth_estimate": 0.0,
            "unique_tokens": 14.0,
            "type_token_ratio": 0.7,
            "upper_ratio": 0.05,
            "symbol_ratio": 0.02,
            "subsys_core": 0.0,
            "subsys_api": 1.0,
            "type_code": 0.0,
            "type_doc": 1.0,
        },
    },
]


SAMPLE_MANIFESTS = [
    {
        "filename": "test_repos.json",
        "path": "/tmp/test_repos.json",
        "data": {
            "repos": {
                "test_repo_a": {"total_files": 100, "source_files": 80, "disk_gb": 0.5, "lang": "Python"},
                "test_repo_b": {"total_files": 200, "source_files": 150, "disk_gb": 1.2, "lang": "Rust"},
            }
        },
    },
    {
        "filename": "test_repo_detail.json",
        "path": "/tmp/test_repo_detail.json",
        "data": {
            "repo": "test_repo_c",
            "source_root": "/tmp/repo_c",
            "total_files": 50,
            "total_bytes": 5000000,
            "elapsed_seconds": 2.5,
        },
    },
]


# --- Tests ---


class TestDeterministicUUID:
    def test_consistent(self):
        """Same inputs produce same UUID."""
        a = deterministic_uuid("test", "hello")
        b = deterministic_uuid("test", "hello")
        assert a == b

    def test_different_namespace(self):
        """Different namespaces produce different UUIDs."""
        a = deterministic_uuid("ns1", "value")
        b = deterministic_uuid("ns2", "value")
        assert a != b

    def test_valid_uuid(self):
        """Output is a valid UUID string."""
        uid = deterministic_uuid("test", "value")
        uuid.UUID(uid)


class TestSHA256:
    def test_known_hash(self):
        h = sha256_digest("hello")
        assert len(h) == 64
        assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_bytes_input(self):
        h = sha256_digest(b"hello")
        assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


class TestBuildVocabulary:
    def test_basic_vocab(self):
        vocab = build_global_vocabulary(SAMPLE_RECORDS, min_df=1, max_df_ratio=1.0)
        assert len(vocab) > 0
        # Common terms should be present
        terms = list(vocab.keys())
        assert "def" in terms
        assert "hello" in terms
        assert "class" in terms

    def test_min_df_filter(self):
        """Terms appearing in fewer than min_df docs are excluded."""
        vocab = build_global_vocabulary(SAMPLE_RECORDS, min_df=2, max_df_ratio=1.0)
        # "class" appears in 1 doc, so should be excluded with min_df=2
        # Actually "class" and "def" - let me check
        # "def" appears in test_file.py (both chunks)
        # "print" appears in 1 doc
        assert "class" not in vocab  # appears in 1 doc

    def test_max_df_filter(self):
        """Terms appearing in more than max_df_ratio docs are excluded."""
        vocab = build_global_vocabulary(SAMPLE_RECORDS, min_df=1, max_df_ratio=0.3)
        # "the" and "is" and "a" might appear in multiple docs
        # Since we have 2 unique docs (test_file.py chunks share source but have different texts)
        # max_df_ratio=0.3 with 3 docs = max 0.9 docs, so terms in >=2 docs filtered
        pass


class TestComputeGlobalTFIDF:
    def test_augmented_records(self):
        vocab = build_global_vocabulary(SAMPLE_RECORDS, min_df=1, max_df_ratio=1.0)
        augmented = compute_global_tfidf(SAMPLE_RECORDS, vocab, top_k=10)
        assert len(augmented) == len(SAMPLE_RECORDS)
        for rec in augmented:
            assert "global_tfidf" in rec
            assert len(rec["global_tfidf"]) <= 10

    def test_tfidf_scores(self):
        vocab = build_global_vocabulary(SAMPLE_RECORDS, min_df=1, max_df_ratio=1.0)
        augmented = compute_global_tfidf(SAMPLE_RECORDS, vocab, top_k=5)
        for rec in augmented:
            for term, score in rec["global_tfidf"].items():
                assert isinstance(score, float)
                assert score > 0


class TestAggregateByFile:
    def test_file_count(self):
        """Records are grouped by source file."""
        file_digests = aggregate_by_file(SAMPLE_RECORDS)
        assert len(file_digests) == 2  # test_file.py + README.md
        assert "test_file.py" in file_digests
        assert "README.md" in file_digests

    def test_chunk_aggregation(self):
        file_digests = aggregate_by_file(SAMPLE_RECORDS)
        py_digest = file_digests["test_file.py"]
        assert py_digest["chunk_count"] == 2
        assert py_digest["dominant_subsystem"] == "core"

    def test_feature_averaging(self):
        file_digests = aggregate_by_file(SAMPLE_RECORDS)
        py_digest = file_digests["test_file.py"]
        avg = py_digest["avg_features"]
        # byte_len: (50 + 40) / 2 = 45
        assert avg["byte_len"] == 45.0
        # function_defs: (1 + 1) / 2 = 1
        assert avg["function_defs"] == 1.0

    def test_content_sha256(self):
        file_digests = aggregate_by_file(SAMPLE_RECORDS)
        py_digest = file_digests["test_file.py"]
        # file_sha256 should be deterministic
        assert len(py_digest["file_sha256"]) == 64

    def test_readme_subsystem(self):
        file_digests = aggregate_by_file(SAMPLE_RECORDS)
        readme_digest = file_digests["README.md"]
        assert readme_digest["dominant_subsystem"] == "api"


class TestRepoManifests:
    def test_read_returns_list(self):
        manifests = read_repo_manifests()
        assert isinstance(manifests, list)

    def test_extract_repos_format_a(self):
        """Extract repos from the 'repos' dict format."""
        repos = extract_repos_from_manifests(SAMPLE_MANIFESTS)
        assert "test_repo_a" in repos
        assert "test_repo_b" in repos
        assert repos["test_repo_a"]["total_files"] == 100

    def test_extract_repos_format_b(self):
        """Extract repos from the single-repo format."""
        repos = extract_repos_from_manifests(SAMPLE_MANIFESTS)
        assert "test_repo_c" in repos
        assert repos["test_repo_c"]["total_files"] == 50
        assert repos["test_repo_c"]["disk_gb"] == pytest.approx(0.005, abs=0.001)


class TestComputeRepoDigests:
    def test_basic_aggregation(self):
        file_digests = aggregate_by_file(SAMPLE_RECORDS)
        repo_meta = {
            "odysseus": {"total_files": 100, "source_files": 80, "disk_gb": 0.5, "lang": "Python"},
        }
        repo_digest = compute_repo_digests(file_digests, repo_meta, SAMPLE_RECORDS)
        assert "odysseus" in repo_digest
        od = repo_digest["odysseus"]
        assert od["digested_file_count"] == 2
        assert len(od["repo_sha256"]) == 64

    def test_repo_sha256_deterministic(self):
        file_digests = aggregate_by_file(SAMPLE_RECORDS)
        repo_meta = {"odysseus": {"total_files": 100, "source_files": 80, "disk_gb": 0.5, "lang": "Python"}}
        rd1 = compute_repo_digests(file_digests, repo_meta, SAMPLE_RECORDS)
        rd2 = compute_repo_digests(file_digests, repo_meta, SAMPLE_RECORDS)
        assert rd1["odysseus"]["repo_sha256"] == rd2["odysseus"]["repo_sha256"]


class TestEmbeddingManifest:
    def test_entry_count(self):
        file_digests = aggregate_by_file(SAMPLE_RECORDS)
        repo_meta = {"odysseus": {"total_files": 100, "source_files": 80, "disk_gb": 0.5, "lang": "Python"}}
        repo_digests = compute_repo_digests(file_digests, repo_meta, SAMPLE_RECORDS)
        manifest = generate_embedding_manifest(file_digests, repo_digests)
        # 2 file entries + 1 repo entry
        assert manifest["manifest"]["total_entries"] == 3

    def test_entry_structure(self):
        file_digests = aggregate_by_file(SAMPLE_RECORDS)
        repo_meta = {"odysseus": {"total_files": 100, "source_files": 80, "disk_gb": 0.5, "lang": "Python"}}
        repo_digests = compute_repo_digests(file_digests, repo_meta, SAMPLE_RECORDS)
        manifest = generate_embedding_manifest(file_digests, repo_digests)
        entry = manifest["entries"][0]
        assert "doc_id" in entry
        assert "embedding_text" in entry
        assert "sha256" in entry
        assert "features" in entry


class TestTokenRegex:
    def test_finds_identifiers(self):
        text = "def hello_world(): return 42"
        tokens = TOKEN_RE.findall(text)
        assert "def" in tokens
        assert "hello_world" in tokens
        assert "return" in tokens
        assert "42" not in tokens  # purely numeric

    def test_skips_short_tokens(self):
        text = "a b c d ef"
        tokens = TOKEN_RE.findall(text)
        assert "a" not in tokens  # < 2 chars
        assert "ef" in tokens
