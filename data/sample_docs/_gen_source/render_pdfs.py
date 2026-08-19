import sys
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

SOURCE_DIR = Path(__file__).parent
OUTPUT_DIR = SOURCE_DIR.parent

styles = getSampleStyleSheet()
title_style = ParagraphStyle("DocTitle", parent=styles["Heading1"], fontSize=14, spaceAfter=4)
subtitle_style = ParagraphStyle("DocSubtitle", parent=styles["Normal"], fontSize=9, textColor="#555555", spaceAfter=16)
heading_style = ParagraphStyle("SectionHeading", parent=styles["Heading2"], fontSize=12, spaceBefore=14, spaceAfter=6)
body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=10)


def render(txt_path: Path) -> Path:
    lines = txt_path.read_text().splitlines()
    title, subtitle, *rest = [line for line in lines if line.strip() != "" or line == ""]
    # first two non-empty lines are title + subtitle; blank lines separate paragraphs
    paragraphs = txt_path.read_text().strip().split("\n\n")
    title_text = paragraphs[0].splitlines()[0]
    subtitle_text = paragraphs[0].splitlines()[1] if len(paragraphs[0].splitlines()) > 1 else ""

    story = [Paragraph(title_text, title_style), Paragraph(subtitle_text, subtitle_style)]
    for para in paragraphs[1:]:
        para = para.strip()
        if not para:
            continue
        first_line, *remaining = para.splitlines()
        is_heading = first_line.isupper() and len(first_line) < 80
        if is_heading:
            story.append(Paragraph(first_line, heading_style))
            body_text = " ".join(remaining).strip()
            if body_text:
                story.append(Paragraph(body_text, body_style))
        else:
            story.append(Paragraph(" ".join(para.splitlines()), body_style))

    out_path = OUTPUT_DIR / (txt_path.stem + ".pdf")
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        topMargin=0.9 * inch,
        bottomMargin=0.9 * inch,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
    )
    doc.build(story)
    return out_path


if __name__ == "__main__":
    for name in [
        "leave_policy.txt",
        "remote_and_expense_policy.txt",
        "conduct_and_safety_policy.txt",
        "it_security_policy.txt",
        "compensation_and_offboarding_policy.txt",
    ]:
        out = render(SOURCE_DIR / name)
        print(f"wrote {out}")
