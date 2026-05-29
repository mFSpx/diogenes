#!/usr/bin/env python3
"""md_to_pdf — lean Markdown → PDF via reportlab (FOSS).

Mutation class: read_only (filesystem product only; emits one PDF).
Receipt scope: LOCAL_FILE_PRODUCT.

Handles: # .. #### headings, paragraphs, - / * bullets, N. numbered lists,
fenced ``` code blocks, --- rules, > blockquotes, **bold**, `inline code`.
Deliberately minimal — no tables. Usage: md_to_pdf.py IN.md OUT.pdf [title]
"""
import re
import sys
import html

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, HRFlowable, ListFlowable, ListItem,
)


def _inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font face="Courier" color="#9b1d64">\1</font>', text)
    return text


def build(md_path: str, pdf_path: str, title: str | None = None) -> None:
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10, leading=14, spaceAfter=6)
    h = {
        1: ParagraphStyle("h1", parent=ss["Heading1"], fontSize=20, leading=24, spaceBefore=18, spaceAfter=8, textColor=HexColor("#111111")),
        2: ParagraphStyle("h2", parent=ss["Heading2"], fontSize=15, leading=19, spaceBefore=14, spaceAfter=6, textColor=HexColor("#6a1b4d")),
        3: ParagraphStyle("h3", parent=ss["Heading3"], fontSize=12, leading=16, spaceBefore=10, spaceAfter=4, textColor=HexColor("#333333")),
        4: ParagraphStyle("h4", parent=ss["Heading4"], fontSize=11, leading=15, spaceBefore=8, spaceAfter=3, textColor=HexColor("#555555")),
    }
    code = ParagraphStyle("code", parent=ss["Code"], fontSize=8, leading=10, backColor=HexColor("#f3f0f4"),
                          borderPadding=6, leftIndent=4, spaceBefore=4, spaceAfter=8, textColor=HexColor("#1a1a1a"))
    quote = ParagraphStyle("quote", parent=body, leftIndent=16, textColor=HexColor("#444444"), fontName="Helvetica-Oblique")

    flow = []
    if title:
        flow += [Paragraph(_inline(title), h[1]), HRFlowable(width="100%", color=HexColor("#6a1b4d")), Spacer(1, 8)]

    lines = open(md_path, encoding="utf-8").read().splitlines()
    i, n = 0, len(lines)
    bullets: list = []
    numbered: list = []

    def flush():
        nonlocal bullets, numbered
        if bullets:
            flow.append(ListFlowable([ListItem(Paragraph(_inline(b), body), leftIndent=14) for b in bullets], bulletType="bullet", start="•"))
            bullets = []
        if numbered:
            flow.append(ListFlowable([ListItem(Paragraph(_inline(b), body)) for b in numbered], bulletType="1"))
            numbered = []

    while i < n:
        ln = lines[i]
        if ln.strip().startswith("```"):
            flush(); i += 1; buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            flow.append(Preformatted("\n".join(buf) or " ", code)); i += 1; continue
        m = re.match(r"(#{1,4})\s+(.*)", ln)
        if m:
            flush(); flow.append(Paragraph(_inline(m.group(2)), h[len(m.group(1))])); i += 1; continue
        if re.match(r"^\s*---+\s*$", ln):
            flush(); flow.append(Spacer(1, 4)); flow.append(HRFlowable(width="100%", color=HexColor("#cccccc"))); flow.append(Spacer(1, 4)); i += 1; continue
        mb = re.match(r"^\s*[-*]\s+(.*)", ln)
        if mb:
            if numbered: flush()
            bullets.append(mb.group(1)); i += 1; continue
        mn = re.match(r"^\s*\d+\.\s+(.*)", ln)
        if mn:
            if bullets: flush()
            numbered.append(mn.group(1)); i += 1; continue
        if ln.strip().startswith(">"):
            flush(); flow.append(Paragraph(_inline(ln.strip()[1:].strip()), quote)); i += 1; continue
        if ln.strip() == "":
            flush(); i += 1; continue
        flush(); flow.append(Paragraph(_inline(ln), body)); i += 1

    flush()
    SimpleDocTemplate(pdf_path, pagesize=LETTER, topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                      leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                      title=title or "LUCIDOTA").build(flow)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: md_to_pdf.py IN.md OUT.pdf [title]")
    build(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    print(f"PDF written: {sys.argv[2]}")
