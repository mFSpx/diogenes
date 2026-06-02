from __future__ import annotations

from scripts.lucidota_ingestion_quality_audit import (
    classify_readability,
    embedding_quality_sql_where,
)


def test_readability_passes_plain_json_and_code_material():
    plain = classify_readability(
        "This is a real readable paragraph about a tenancy file with enough words to embed safely.",
        mime="text/plain",
        source_path="KRAMPUSCHEWING/example.md",
    )
    structured = classify_readability(
        '{"messages": [{"sender_name": "A", "content": "this is readable exported chat text"}]}',
        mime="text/json",
        source_path="C_ARCHIVE.zip!facebook-export/message_1.json",
    )
    code = classify_readability(
        "def hello_world():\n    return 'readable code with identifiers and strings'\n",
        mime="text/py",
        source_path="repo/script.py",
    )

    assert plain["status"] == "pass"
    assert structured["status"] == "pass"
    assert code["status"] == "pass"


def test_readability_blocks_raw_email_and_binaryish_chunks():
    raw_email = classify_readability(
        "Content-Transfer-Encoding: quoted-printable\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        "<div class=3D\"protonmail_signature_block\">hello=C2=A0world</div>",
        mime="text/eml",
        source_path="C_ARCHIVE.zip!Inbox.zip!bad.eml",
    )
    binaryish = classify_readability("\x00\x01\x02" * 40, mime="text/plain", source_path="bad.txt")

    assert raw_email["status"] == "block"
    assert "raw_email_headers_or_html" in raw_email["reasons"]
    assert binaryish["status"] == "block"
    assert "nonprintable_text" in binaryish["reasons"]


def test_embedding_quality_sql_where_excludes_audit_blocked_rows():
    where = embedding_quality_sql_where()

    assert "embedding IS NULL" in where
    assert "embedding_quality_status" in where
    assert "block" in where
