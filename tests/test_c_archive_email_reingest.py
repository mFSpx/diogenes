from __future__ import annotations

from email.message import EmailMessage

from scripts.lucidota_c_archive_email_reingest import chunk_text, extract_email_text


def test_extract_email_text_decodes_quoted_printable_body_without_raw_headers():
    msg = EmailMessage()
    msg["Subject"] = "Hello"
    msg["From"] = "sender@example.test"
    msg["Date"] = "Mon, 01 Jun 2026 00:00:00 +0000"
    msg.set_content("This is a readable quoted printable body with nonbreaking space: café.", cte="quoted-printable")

    text = extract_email_text(msg.as_bytes())

    assert "Subject: Hello" in text
    assert "Content-Transfer-Encoding" not in text
    assert "=C2" not in text
    assert "readable quoted printable body" in text


def test_extract_email_text_keeps_short_valid_messages():
    msg = EmailMessage()
    msg["Subject"] = "Ack"
    msg["From"] = "sender@example.test"
    msg["Date"] = "Mon, 01 Jun 2026 00:00:00 +0000"
    msg.set_content("OK", cte="quoted-printable")

    text = extract_email_text(msg.as_bytes())

    assert text
    assert "Subject: Ack" in text
    assert "\n\nOK" in text


def test_chunk_text_preserves_source_order():
    chunks = chunk_text("abcdef", max_chars=2)

    assert chunks == ["ab", "cd", "ef"]
